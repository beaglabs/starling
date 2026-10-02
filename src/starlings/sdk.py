"""Evidence-aware local SDK. No network, generation, or autonomous CUI release authority."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint
from .datasets import content_text, pack
from .model import select_device
from .registry import category_task, digest
from .schemas import Candidate, Content, EvaluationResult, Finding, Question, SourceBlock
from .training import decision_logits, environment


class Classifier:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device="auto",
        threshold=0.95,
        calibration: str | Path | None = None,
        max_evaluations=4096,
        deterministic=False,
    ):
        if not math.isfinite(threshold) or not 0.5 <= threshold <= 1:
            raise ValueError("Threshold must be finite and in [0.5,1]")
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        self.device = select_device(device)
        self.model, self.tokenizer, self.registry, self.manifest = load_checkpoint(
            checkpoint, self.device
        )
        if self.manifest["stage"] != "decision":
            raise ValueError(
                "SDK requires a decision-trained checkpoint, not initialized/pretrained weights"
            )
        if deterministic:
            # Process-wide setting. Unsupported kernels fail rather than silently change behavior.
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "mha"):
                torch.backends.mha.set_fastpath_enabled(False)
        self.threshold = threshold
        self.max_evaluations = max_evaluations
        self.categories = {category.id: category for category in self.registry.categories}
        self.calibration = None
        if calibration:
            calibration = json.loads(Path(calibration).read_text())
            if calibration.get("pipeline_hash") != digest(self.manifest["files"]):
                raise ValueError(
                    "Calibration belongs to a different tokenizer/configuration pipeline"
                )
            if calibration.get("model_id") != self.manifest["model_id"]:
                raise ValueError("Calibration belongs to different weights")
            if calibration.get("registry_hash") != self.manifest["registry_hash"]:
                raise ValueError("Calibration belongs to another registry snapshot")
            if calibration.get("environment") != environment(self.device):
                raise ValueError("Calibration environment differs; refit for this deployment")
            temperature = calibration.get("temperature")
            if (
                not isinstance(temperature, (float, int))
                or not math.isfinite(temperature)
                or temperature <= 0
            ):
                raise ValueError("Invalid calibration temperature")
            if not calibration.get("task_signatures"):
                raise ValueError("Calibration contains no evaluated tasks")
            self.calibration = calibration

    @staticmethod
    def normalize_content(content: str | Content | dict) -> Content:
        if isinstance(content, str):
            if not content.strip():
                raise ValueError("Content must not be empty")
            return Content(blocks=[SourceBlock(id="block-1", text=content)])
        return Content.model_validate(content)

    def _score(self, task_id, question, text, state, refs, scope, count) -> Finding:
        count[0] += 1
        if count[0] > self.max_evaluations:
            raise ValueError(
                "Evaluation budget exceeded; narrow scope or raise max_evaluations explicitly"
            )
        signature = digest(question.model_dump())
        calibrated = bool(self.calibration and signature in self.calibration["task_signatures"])
        reasons = []
        try:
            with torch.inference_mode():
                options, logits = decision_logits(
                    self.model, self.tokenizer, text, state, question, self.device
                )
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Model produced nonfinite logits")
                temperature = self.calibration["temperature"] if calibrated else 1.0
                scores = torch.softmax(logits / temperature, dim=-1).cpu().tolist()
            probabilities = dict(zip(options, scores, strict=True))
            outcome = options[max(range(len(scores)), key=lambda i: scores[i])]
            confidence = probabilities[outcome]
        except ValueError as exc:
            if "exceeds context" not in str(exc):
                raise
            # Explicit unresolved outcome; no truncated evidence is accepted as a complete evaluation.
            probabilities = {
                option: 1 / len(question.criteria) for option in sorted(question.criteria)
            }
            outcome, confidence, calibrated = "insufficient_evidence", 0.0, False
            reasons.append(str(exc))
        if not calibrated:
            reasons.append("No matching held-out calibration for this task and deployment")
        if confidence < self.threshold:
            reasons.append("Below configured review threshold")
        if outcome == "insufficient_evidence":
            reasons.append("Insufficient evidence")
        if self.manifest.get("allow_unreviewed") or (
            self.calibration and self.calibration.get("allow_unreviewed")
        ):
            reasons.append("Experiment used unreviewed labels")
        # Category findings always remain marking proposals; this gate is a review recommendation.
        return Finding(
            task_id=task_id,
            block_refs=refs,
            outcome=outcome,
            probabilities=probabilities,
            confidence=confidence,
            calibrated=calibrated,
            requires_review=bool(reasons),
            reasons=reasons,
            scope=scope,
        )

    def _pieces(self, block, state, question):
        # Reserve space for the longest alternative + all state/policy text. Never truncate them.
        overhead = max(
            len(self.tokenizer.encode(pack("", state, question, option)).ids)
            for option in question.criteria
        )
        budget = self.model.config.max_length - overhead - 16
        encoded = self.tokenizer.encode(block.text)
        if budget < 16 or len(encoded.ids) <= budget:
            return [block.text]
        # Overlapping token windows, mapped to exact source substrings via tokenizer offsets.
        pieces = []
        stride = max(1, budget - min(64, budget // 4))
        for start in range(0, len(encoded.ids), stride):
            end = min(start + budget, len(encoded.ids))
            left, right = encoded.offsets[start][0], encoded.offsets[end - 1][1]
            pieces.append(block.text[left:right])
            if end == len(encoded.ids):
                break
        return pieces

    def evaluate(
        self,
        *,
        content: str | Content | dict,
        state: dict[str, Any] | None = None,
        categories: list[str] | str = "all",
        questions: dict[str, Question | dict] | None = None,
        candidates: list[Candidate | dict] | None = None,
    ) -> EvaluationResult:
        document = self.normalize_content(content)
        if state is not None and not isinstance(state, dict):
            raise ValueError("state must be a JSON object")
        state = state or {}
        # JSON validation prevents hidden nonfinite values and makes input identities stable.
        digest(state)
        if categories == "all":
            selected = sorted(self.categories)
        elif isinstance(categories, list) and all(isinstance(c, str) for c in categories):
            selected = sorted(set(categories))
            unknown = set(selected) - set(self.categories)
            if unknown:
                raise ValueError(f"Unknown categories: {sorted(unknown)}")
        else:
            raise ValueError("categories must be 'all' or a list of registry ids")
        normalized_questions = {
            key: Question.model_validate(q) for key, q in (questions or {}).items()
        }
        normalized_candidates = [
            Candidate.model_validate(candidate) for candidate in (candidates or [])
        ]
        if len({c.id for c in normalized_candidates}) != len(normalized_candidates):
            raise ValueError("Duplicate candidate ids")
        blocks = {block.id: block for block in document.blocks}
        # Validate every candidate before any model work; invented references/quotes fail closed.
        for candidate in normalized_candidates:
            if len(set(candidate.evidence_refs)) != len(candidate.evidence_refs):
                raise ValueError("Duplicate candidate evidence references")
            if set(candidate.evidence_refs) - set(blocks):
                raise ValueError(f"Candidate {candidate.id} references missing source blocks")
            if set(candidate.category_hypotheses) - set(self.categories):
                raise ValueError(f"Candidate {candidate.id} names unknown categories")
            referenced = "\n".join(blocks[ref].text for ref in candidate.evidence_refs)
            if candidate.quote and candidate.quote not in referenced:
                raise ValueError(f"Candidate {candidate.id} quote does not match source")
        findings, question_findings, candidate_findings = [], [], {}
        warnings = list(document.extraction_warnings)
        count = [0]
        whole_text = content_text(document)
        for category_id in selected:
            task = category_task(self.categories[category_id])
            for block in document.blocks:
                for piece in self._pieces(block, state, task):
                    findings.append(
                        self._score(category_id, task, piece, state, [block.id], "block", count)
                    )
            # Separate whole-document pass surfaces aggregation/context limits; block scores are not
            # combined into an invented globally calibrated "document is safe" probability.
            findings.append(
                self._score(category_id, task, whole_text, state, list(blocks), "document", count)
            )
        for candidate in normalized_candidates:
            refs = candidate.evidence_refs
            text = "\n".join(f"[{ref}] {blocks[ref].text}" for ref in refs)
            # Hypotheses do not restrict the independent scan or mutate the supplied state.
            candidate_categories = sorted(
                set(candidate.category_hypotheses or selected) & set(selected)
            )
            candidate_findings[candidate.id] = [
                self._score(
                    cid, category_task(self.categories[cid]), text, state, refs, "candidate", count
                )
                for cid in candidate_categories
            ]
        for key, task in sorted(normalized_questions.items()):
            question_findings.append(
                self._score(key, task, whole_text, state, list(blocks), "question", count)
            )
        if not selected:
            warnings.append("No categories were evaluated")
        if set(selected) != set(self.categories):
            warnings.append(
                "Category subset: findings do not establish absence of other CUI categories"
            )
        if not state.get("provenance"):
            warnings.append(
                "Verified source/release/contract context has not been established by the host"
            )
        all_findings = (
            findings + question_findings + [f for fs in candidate_findings.values() for f in fs]
        )
        reasons = [r for finding in all_findings for r in finding.reasons if "exceeds context" in r]
        warnings.extend(sorted(set(reasons)))
        return EvaluationResult(
            model_id=self.manifest["model_id"],
            registry_version=self.registry.version,
            evaluated_categories=selected,
            unevaluated_categories=sorted(set(self.categories) - set(selected)),
            findings=findings,
            question_findings=question_findings,
            candidate_findings=candidate_findings,
            coverage={
                "source_blocks": len(blocks),
                "independent_scan": True,
                "category_block_windows": sum(f.scope == "block" for f in findings),
                "document_context_complete": not bool(reasons),
                "evaluations": count[0],
            },
            requires_review=bool(warnings) or any(f.requires_review for f in all_findings),
            warnings=warnings,
            provenance={
                "input_hash": digest(
                    {
                        "content": document.model_dump(),
                        "state": state,
                        "categories": selected,
                        "questions": {k: v.model_dump() for k, v in normalized_questions.items()},
                        "candidates": [c.model_dump() for c in normalized_candidates],
                    }
                ),
                "pipeline_hash": digest(self.manifest["files"]),
                "registry_hash": self.manifest["registry_hash"],
                "calibration_hash": digest(self.calibration) if self.calibration else None,
                "environment": environment(self.device),
                "threshold": self.threshold,
                "interpretation": "Decision support and marking proposals; no release authorization",
            },
        )

    def classify(self, content, *, state=None, categories="all"):
        return self.evaluate(content=content, state=state, categories=categories)
