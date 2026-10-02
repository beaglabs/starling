import json
import shutil

import pytest
import torch

from starlings import Classifier
from starlings.checkpoint import load_checkpoint
from starlings.datasets import read_rows, write_rows
from starlings.evaluation import evaluate
from starlings.schemas import Question
from starlings.training import row_loss, train


def test_random_pretrain_decision_lifecycle_and_integrity(pipeline, tmp_path):
    initial, _, _, first = load_checkpoint(pipeline / "init")
    trained, _, _, last = load_checkpoint(pipeline / "trained")
    assert first["stage"] == "initialized" and first["random_initialization"]
    assert initial.config.decision_pooling == "option_mean"
    assert last["stage"] == "decision" and last["steps"] == 3
    assert not torch.equal(initial.token_embedding.weight, trained.token_embedding.weight)
    with pytest.raises(ValueError, match="decision-trained"):
        Classifier(pipeline / "init")
    shutil.copytree(pipeline / "trained", tmp_path / "tampered")
    (tmp_path / "tampered/config.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        Classifier(tmp_path / "tampered")


def test_option_pooling_can_learn_all_three_alternatives(pipeline):
    model, tokenizer, registry, _ = load_checkpoint(pipeline / "init", "cpu")
    rows = read_rows(pipeline / "splits/train.jsonl")
    selected = {}
    for row in rows:
        selected.setdefault(row.label, row)
    assert set(selected) == {"billing", "technical", "insufficient_evidence"}
    examples = [selected[key] for key in sorted(selected)]

    def average_loss():
        return torch.stack(
            [row_loss(model, tokenizer, row, registry, torch.device("cpu")) for row in examples]
        ).mean()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    with torch.no_grad():
        initial_loss = float(average_loss())
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = average_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = float(average_loss())
    assert final_loss < 0.8
    assert final_loss < initial_loss - 0.2


def test_heldout_test_report_and_no_leakage(pipeline, tmp_path):
    report = evaluate(
        pipeline / "trained",
        pipeline / "splits/test.jsonl",
        tmp_path / "report.json",
        calibration=pipeline / "calibration.json",
        device="cpu",
    )
    assert report["samples"] == 8
    assert report["calibration_coverage"] == 1
    assert not report["experimental"]
    assert report["unrepresented_categories"] == ["other_category", "test_category"]
    assert 0 <= report["ece"] <= 1
    for bad in ["train", "validation", "calibration"]:
        with pytest.raises(ValueError, match="overlap"):
            evaluate(
                pipeline / "trained",
                pipeline / f"splits/{bad}.jsonl",
                tmp_path / f"{bad}.json",
                calibration=pipeline / "calibration.json",
                device="cpu",
            )


def test_full_scan_candidate_integrity_and_repeatability(pipeline):
    classifier = Classifier(pipeline / "trained", device="cpu", deterministic=True)
    content = {
        "blocks": [
            {"id": "p1-b1", "text": "A specification."},
            {"id": "p2-b1", "text": "A second specification."},
        ]
    }
    kwargs = dict(
        content=content,
        state={"provenance": {"host_verified": True}},
        candidates=[
            {
                "id": "c1",
                "evidence_refs": ["p1-b1"],
                "category_hypotheses": ["test_category"],
                "quote": "A specification.",
            }
        ],
    )
    result = classifier.evaluate(**kwargs)
    assert result.model_dump() == classifier.evaluate(**kwargs).model_dump()
    assert result.evaluated_categories == ["other_category", "test_category"]
    assert len(result.findings) == 6 and len(result.candidate_findings["c1"]) == 1
    assert any(f.block_refs == ["p2-b1"] for f in result.findings)
    assert result.requires_review and all(not f.calibrated for f in result.findings)
    assert len(result.filter_categories(["test_category"])) == 3
    with pytest.raises(ValueError, match="not evaluated"):
        result.filter_categories(["missing"])
    for bad in [
        {"id": "c", "evidence_refs": ["invented"]},
        {"id": "c", "evidence_refs": ["p1-b1"], "quote": "An invented quote"},
    ]:
        with pytest.raises(ValueError):
            classifier.evaluate(content=content, candidates=[bad])
    with pytest.raises(ValueError, match="Unknown categories"):
        classifier.classify("Some text", categories=["typo"])


def test_option_order_and_calibration_coverage(pipeline):
    classifier = Classifier(
        pipeline / "trained", calibration=pipeline / "calibration.json", device="cpu"
    )
    row = read_rows(pipeline / "splits/test.jsonl")[0]
    a = classifier.evaluate(
        content=row.content, state=row.state, categories=[], questions={"route": row.question}
    )
    reverse = Question(
        instruction=row.question.instruction,
        criteria=dict(reversed(list(row.question.criteria.items()))),
    )
    b = classifier.evaluate(
        content=row.content, state=row.state, categories=[], questions={"route": reverse}
    )
    assert a.question_findings[0].probabilities == b.question_findings[0].probabilities
    assert a.question_findings[0].calibrated and b.question_findings[0].calibrated
    changed = reverse.model_copy(update={"instruction": "A different task"})
    c = classifier.evaluate(content=row.content, categories=[], questions={"route": changed})
    assert not c.question_findings[0].calibrated
    assert c.question_findings[0].requires_review


def test_bad_calibration_identity_and_environment(pipeline, tmp_path):
    artifact = json.loads((pipeline / "calibration.json").read_text())
    for key, value in [
        ("pipeline_hash", "wrong"),
        ("model_id", "wrong"),
        ("registry_hash", "wrong"),
        ("environment", {}),
        ("temperature", 0),
    ]:
        bad = {**artifact, key: value}
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            Classifier(pipeline / "trained", calibration=path, device="cpu")


def test_long_context_is_explicit_and_windows_still_scan(pipeline):
    classifier = Classifier(pipeline / "trained", device="cpu")
    result = classifier.classify("word " * 1200, categories=["test_category"])
    assert result.coverage["category_block_windows"] > 1
    assert not result.coverage["document_context_complete"] and result.requires_review
    document = next(f for f in result.findings if f.scope == "document")
    assert document.outcome == "insufficient_evidence" and document.confidence == 0
    assert any("exceeds context" in warning for warning in result.warnings)
    with pytest.raises(ValueError, match="budget"):
        Classifier(pipeline / "trained", max_evaluations=1).classify("text")


def test_training_rejects_unreviewed_and_prior_leaks(pipeline, tmp_path):
    rows = read_rows(pipeline / "splits/train.jsonl")
    rows[0].review.status = "pending"
    write_rows(tmp_path / "unreviewed.jsonl", rows)
    with pytest.raises(ValueError, match="unreviewed"):
        train(
            pipeline / "init",
            tmp_path / "unreviewed.jsonl",
            pipeline / "splits/validation.jsonl",
            tmp_path / "reject",
        )
    with pytest.raises(ValueError, match="prior checkpoint"):
        train(
            pipeline / "trained",
            pipeline / "splits/test.jsonl",
            pipeline / "splits/train.jsonl",
            tmp_path / "leak",
        )
    with pytest.raises(FileExistsError):
        train(pipeline / "init", "does-not-exist", "does-not-exist", pipeline / "trained")
