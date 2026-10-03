import json
from pathlib import Path

import pytest

import starlings.trainer as trainer_module
from starlings import FactorSpec, Trainer, TrainingConfig
from starlings.schemas import DecisionRow, Question, Review


def write_factor_rows(path: Path, *, labels=None, instruction="Classify technical readiness."):
    labels = labels or ["low", "medium", "high"]
    question = Question(
        instruction=instruction,
        criteria={label: f"The evidence supports {label}." for label in labels},
    )
    row = DecisionRow(
        id=path.stem,
        group_id=f"group-{path.stem}",
        content="Representative technical evidence.",
        state={"program": "demo"},
        question=question,
        label=labels[0],
        review=Review(status="approved", reviewer="reviewer", rationale="Reviewed example."),
        source_kind="operational",
    )
    path.write_text(row.model_dump_json() + "\n")


def test_factor_spec_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="unique"):
        FactorSpec(name="risk", labels=["low", "low"])


def test_trainer_public_api_orchestrates_factor_lifecycle(monkeypatch, tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    test_path = tmp_path / "test.jsonl"
    for path in [train_path, validation_path, calibration_path, test_path]:
        write_factor_rows(path)

    def fake_train(checkpoint, train, validation, output, **kwargs):
        output = Path(output)
        output.mkdir()
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": "model-123",
                    "files": {"model.safetensors": "abc"},
                    "train_digest": "training-digest",
                }
            )
        )
        return {
            "checkpoint": str(output),
            "steps": 7,
            "loss": 0.21,
            "validation_loss": 0.34,
            "device": "cpu",
        }

    def fake_calibrate(checkpoint, dataset, output, **kwargs):
        payload = {"temperature": 1.2, "dataset_hash": "calibration-digest"}
        Path(output).write_text(json.dumps(payload))
        return payload

    def fake_evaluate(checkpoint, dataset, output, **kwargs):
        payload = {
            "model_id": "model-123",
            "dataset_hash": "test-digest",
            "samples": 10,
            "accuracy": 0.9,
            "nll": 0.2,
            "brier": 0.1,
            "ece": 0.03,
            "calibration_coverage": 1.0,
            "experimental": False,
        }
        Path(output).write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(trainer_module, "_train", fake_train)
    monkeypatch.setattr(trainer_module, "_calibrate", fake_calibrate)
    monkeypatch.setattr(trainer_module, "_evaluate", fake_evaluate)

    factor = FactorSpec(
        name="technical-readiness",
        labels=["low", "medium", "high"],
        description="Reusable readiness judgment.",
    )
    result = Trainer(device="cpu").train(
        factor=factor,
        checkpoint=tmp_path / "base",
        train=train_path,
        validation=validation_path,
        calibration=calibration_path,
        test=test_path,
        output=tmp_path / "factor-v1",
        config=TrainingConfig(epochs=3, learning_rate=1e-4, seed=17),
    )

    assert result.factor == factor
    assert result.model_id == "model-123"
    assert result.training_dataset_hash == "training-digest"
    assert result.steps == 7
    assert result.evaluation is not None
    assert result.evaluation.accuracy == 0.9
    assert result.calibration_path == str(tmp_path / "factor-v1.calibration.json")
    assert result.evaluation_path == str(tmp_path / "factor-v1.evaluation.json")
    assert len(result.task_signature) == 64


def test_factor_dataset_must_match_labels_and_one_question_contract(tmp_path):
    factor = FactorSpec(name="risk", labels=["low", "medium", "high"])
    good = tmp_path / "good.jsonl"
    bad_labels = tmp_path / "bad-labels.jsonl"
    bad_instruction = tmp_path / "bad-instruction.jsonl"
    write_factor_rows(good)
    write_factor_rows(bad_labels, labels=["low", "high"])
    write_factor_rows(bad_instruction, instruction="A different task.")

    with pytest.raises(ValueError, match="labels"):
        Trainer._validate_factor_datasets(factor, [good, bad_labels])
    with pytest.raises(ValueError, match="stable question"):
        Trainer._validate_factor_datasets(factor, [good, bad_instruction])
