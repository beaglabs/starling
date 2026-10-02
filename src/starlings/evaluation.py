"""Held-out temperature calibration and category-aware decision metrics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .datasets import content_text, fingerprint, read_rows, row_task, validate_rows, write_json
from .model import select_device
from .registry import digest
from .training import decision_logits, environment


def collect(checkpoint, dataset, device="auto", allow_unreviewed=False):
    selected = select_device(device)
    model, tokenizer, registry, manifest = load_checkpoint(checkpoint, selected)
    if manifest["stage"] != "decision":
        raise ValueError("Calibration/evaluation requires a decision-trained checkpoint")
    rows = read_rows(dataset)
    validate_rows(rows, registry, allow_unreviewed)
    blocked_groups = set(manifest.get("training_groups", [])) | set(
        manifest.get("validation_groups", [])
    )
    blocked_examples = set(manifest.get("training_examples", [])) | set(
        manifest.get("validation_examples", [])
    )
    if any(r.group_id in blocked_groups or fingerprint(r) in blocked_examples for r in rows):
        raise ValueError("Held-out dataset overlaps training or model-selection validation")
    records = []
    with torch.inference_mode():
        for row in rows:
            task = row_task(row, registry)
            options, logits = decision_logits(
                model, tokenizer, content_text(row.content), row.state, task, selected
            )
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite model logits")
            records.append(
                {
                    "row": row,
                    "options": options,
                    "logits": logits.cpu().clone(),
                    "gold": options.index(row.label),
                    "signature": digest(task.model_dump()),
                }
            )
    return records, manifest, registry, selected


def metrics(records, temperature=1.0, threshold=0.95):
    correct = kept = kept_correct = missed = positives = false_positives = negatives = 0
    nll = brier = 0.0
    bins = [[] for _ in range(10)]
    outcomes = defaultdict(
        lambda: {
            "samples": 0,
            "correct": 0,
            "gold_applicable": 0,
            "applicable_hits": 0,
            "applicable_false_negatives": 0,
            "gold_not_applicable": 0,
            "false_positives": 0,
            "abstentions": 0,
        }
    )
    confusion = defaultdict(lambda: defaultdict(int))
    for record in records:
        probabilities = torch.softmax(
            record["logits"] / record.get("temperature", temperature), dim=-1
        )
        pred = int(probabilities.argmax())
        confidence = float(probabilities[pred])
        gold = record["gold"]
        hit = pred == gold
        correct += hit
        nll += -math.log(max(float(probabilities[gold]), 1e-12))
        target = F.one_hot(torch.tensor(gold), len(probabilities)).float()
        brier += float(((probabilities - target) ** 2).sum())
        bins[min(9, int(confidence * 10))].append((confidence, hit))
        if confidence >= threshold and record["options"][pred] != "insufficient_evidence":
            kept += 1
            kept_correct += hit
        row = record["row"]
        key = row.category or "custom_questions"
        outcomes[key]["samples"] += 1
        outcomes[key]["correct"] += hit
        predicted = record["options"][pred]
        confusion[row.label][predicted] += 1
        outcomes[key]["abstentions"] += predicted == "insufficient_evidence"
        if row.category:
            outcomes[key]["gold_applicable"] += row.label == "applicable"
            outcomes[key]["applicable_hits"] += (
                row.label == "applicable" and predicted == "applicable"
            )
            outcomes[key]["applicable_false_negatives"] += (
                row.label == "applicable" and predicted == "not_applicable"
            )
            outcomes[key]["gold_not_applicable"] += row.label == "not_applicable"
            outcomes[key]["false_positives"] += (
                row.label == "not_applicable" and predicted == "applicable"
            )
            positives += row.label == "applicable"
            missed += row.label == "applicable" and predicted == "not_applicable"
            negatives += row.label == "not_applicable"
            false_positives += row.label == "not_applicable" and predicted == "applicable"
    n = len(records)
    ece = sum(
        len(bucket)
        / n
        * abs(sum(c for c, _ in bucket) / len(bucket) - sum(h for _, h in bucket) / len(bucket))
        for bucket in bins
        if bucket
    )
    return {
        "samples": n,
        "accuracy": correct / n,
        "nll": nll / n,
        "brier": brier / n,
        "ece": ece,
        "threshold": threshold,
        "score_threshold_coverage": kept / n,
        "score_threshold_accuracy": kept_correct / kept if kept else None,
        "applicable_mislabeled_not_applicable": missed,
        "false_negative_rate": missed / positives if positives else None,
        "false_positive_rate": false_positives / negatives if negatives else None,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "per_category": {
            k: {
                **v,
                "accuracy": v["correct"] / v["samples"],
                "applicable_recall": v["applicable_hits"] / v["gold_applicable"]
                if v["gold_applicable"]
                else None,
                "abstention_rate": v["abstentions"] / v["samples"],
            }
            for k, v in outcomes.items()
        },
    }


def calibrate(checkpoint, dataset, output, *, device="auto", allow_unreviewed=False):
    if Path(output).exists():
        raise FileExistsError("Calibration output exists; select a fresh path")
    records, manifest, registry, selected = collect(checkpoint, dataset, device, allow_unreviewed)
    # Deterministic scalar search on held-out labels. Temperature changes confidence, not argmax.
    temperatures = torch.logspace(-1, 1.3, 121).tolist()
    best = min(temperatures, key=lambda t: metrics(records, t)["nll"])
    artifact = {
        "format_version": 1,
        "model_id": manifest["model_id"],
        "pipeline_hash": digest(manifest["files"]),
        "registry_hash": manifest["registry_hash"],
        "environment": environment(selected),
        "temperature": best,
        "task_signatures": sorted({r["signature"] for r in records}),
        "groups": sorted({r["row"].group_id for r in records}),
        "examples": sorted({fingerprint(r["row"]) for r in records}),
        "dataset_hash": digest([r["row"].model_dump() for r in records]),
        "allow_unreviewed": allow_unreviewed or manifest.get("allow_unreviewed", False),
        "raw_metrics": metrics(records),
        "fitted_metrics": metrics(records, best),
        "note": "Calibration-fit metrics are not held-out test performance.",
    }
    write_json(output, artifact)
    return artifact


def evaluate(
    checkpoint,
    dataset,
    output,
    *,
    calibration=None,
    device="auto",
    threshold=0.95,
    allow_unreviewed=False,
):
    if not math.isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError("Threshold must be in [0.5,1]")
    if Path(output).exists():
        raise FileExistsError("Evaluation output exists; select a fresh path")
    records, manifest, registry, selected = collect(checkpoint, dataset, device, allow_unreviewed)
    temperature = 1.0
    fitted = None
    if calibration:
        fitted = json.loads(Path(calibration).read_text())
        if fitted.get("pipeline_hash") != digest(manifest["files"]):
            raise ValueError("Calibration input pipeline mismatch")
        if (
            fitted["model_id"] != manifest["model_id"]
            or fitted["registry_hash"] != manifest["registry_hash"]
        ):
            raise ValueError("Calibration identity mismatch")
        if fitted["environment"] != environment(selected):
            raise ValueError("Calibration deployment mismatch")
        if any(
            r["row"].group_id in fitted["groups"] or fingerprint(r["row"]) in fitted["examples"]
            for r in records
        ):
            raise ValueError("Test examples overlap the calibration split")
        temperature = fitted["temperature"]
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid calibration temperature")
    # Per-task calibration coverage is explicit. Never claim unseen tasks are calibrated.
    for record in records:
        record["calibrated"] = bool(fitted and record["signature"] in fitted["task_signatures"])
        record["temperature"] = temperature if record["calibrated"] else 1.0
    report = metrics(records, temperature, threshold)
    report.update(
        {
            "model_id": manifest["model_id"],
            "registry_version": registry.version,
            "threshold_interpretation": "Score-only coverage, not SDK review acceptance or release authorization",
            "calibration_coverage": sum(r["calibrated"] for r in records) / len(records),
            "unrepresented_categories": sorted(
                {c.id for c in registry.categories} - {r["row"].category for r in records}
            ),
            "experimental": allow_unreviewed
            or manifest.get("allow_unreviewed", False)
            or bool(fitted and fitted.get("allow_unreviewed")),
            "environment": environment(selected),
            "dataset_hash": digest([r["row"].model_dump() for r in records]),
        }
    )
    write_json(output, report)
    return report
