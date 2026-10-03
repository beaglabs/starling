"""Public training API for reusable decision Factors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .datasets import read_rows
from .evaluation import calibrate as _calibrate
from .evaluation import evaluate as _evaluate
from .registry import digest
from .training import train as _train


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactorSpec(_StrictModel):
    """Stable contract for one reusable learned judgment."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", min_length=1, max_length=128)
    type: Literal["choice"] = "choice"
    labels: list[str] = Field(min_length=2, max_length=32)
    description: str = Field(default="", max_length=4096)

    @model_validator(mode="after")
    def valid_labels(self) -> FactorSpec:
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("Factor labels must be unique")
        if any(not label.strip() for label in self.labels):
            raise ValueError("Factor labels must not be empty")
        return self


class TrainingConfig(_StrictModel):
    """Bounded knobs exposed by the supported training API."""

    epochs: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    grad_accum: int = Field(default=8, ge=1)
    seed: int = 42
    max_steps: int | None = Field(default=None, ge=1)
    allow_unreviewed: bool = False
    freeze_encoder: bool = False
    threshold: float = Field(default=0.95, ge=0.5, le=1.0)


class EvaluationReport(_StrictModel):
    """Typed summary plus the complete immutable evaluation payload."""

    model_id: str
    dataset_hash: str
    samples: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    nll: float = Field(ge=0)
    brier: float = Field(ge=0)
    ece: float = Field(ge=0)
    calibration_coverage: float = Field(ge=0, le=1)
    experimental: bool
    metrics: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EvaluationReport:
        return cls(
            model_id=str(payload["model_id"]),
            dataset_hash=str(payload["dataset_hash"]),
            samples=int(payload["samples"]),
            accuracy=float(payload["accuracy"]),
            nll=float(payload["nll"]),
            brier=float(payload["brier"]),
            ece=float(payload["ece"]),
            calibration_coverage=float(payload["calibration_coverage"]),
            experimental=bool(payload["experimental"]),
            metrics=payload,
        )


class TrainingResult(_StrictModel):
    """Content-addressed identity and metrics for one trained Factor version."""

    factor: FactorSpec
    checkpoint: str
    model_id: str
    pipeline_hash: str
    training_dataset_hash: str
    task_signature: str
    steps: int = Field(ge=0)
    loss: float
    validation_loss: float
    device: str
    calibration_path: str | None = None
    evaluation_path: str | None = None
    calibration: dict[str, Any] | None = None
    evaluation: EvaluationReport | None = None


class Trainer:
    """Supported orchestration API over Starlings training/calibration/evaluation internals.

    The caller supplies an initialized or pretrained Starlings checkpoint. Factor datasets must use
    one custom Question contract consistently across every split; category-registry rows are rejected
    because a Factor is a reusable custom judgment, not a registry scan.
    """

    def __init__(self, *, device: str = "auto") -> None:
        self.device = device

    def train(
        self,
        *,
        factor: FactorSpec,
        checkpoint: str | Path,
        train: str | Path,
        validation: str | Path,
        output: str | Path,
        calibration: str | Path | None = None,
        test: str | Path | None = None,
        config: TrainingConfig | None = None,
        calibration_output: str | Path | None = None,
        evaluation_output: str | Path | None = None,
    ) -> TrainingResult:
        config = config or TrainingConfig()
        split_paths = [train, validation]
        if calibration is not None:
            split_paths.append(calibration)
        if test is not None:
            split_paths.append(test)
        task_signature = self._validate_factor_datasets(factor, split_paths)

        output_path = Path(output)
        low_level = _train(
            checkpoint,
            train,
            validation,
            output_path,
            epochs=config.epochs,
            lr=config.learning_rate,
            grad_accum=config.grad_accum,
            device=self.device,
            seed=config.seed,
            max_steps=config.max_steps,
            allow_unreviewed=config.allow_unreviewed,
            freeze_encoder=config.freeze_encoder,
        )
        manifest = json.loads((output_path / "manifest.json").read_text())

        calibration_artifact = None
        calibration_path = None
        if calibration is not None:
            calibration_path = (
                Path(calibration_output)
                if calibration_output
                else Path(f"{output_path}.calibration.json")
            )
            calibration_artifact = _calibrate(
                output_path,
                calibration,
                calibration_path,
                device=self.device,
                allow_unreviewed=config.allow_unreviewed,
            )

        evaluation_report = None
        evaluation_path = None
        if test is not None:
            evaluation_path = (
                Path(evaluation_output)
                if evaluation_output
                else Path(f"{output_path}.evaluation.json")
            )
            payload = _evaluate(
                output_path,
                test,
                evaluation_path,
                calibration=calibration_path,
                device=self.device,
                threshold=config.threshold,
                allow_unreviewed=config.allow_unreviewed,
            )
            evaluation_report = EvaluationReport.from_payload(payload)

        return TrainingResult(
            factor=factor,
            checkpoint=str(output_path),
            model_id=str(manifest["model_id"]),
            pipeline_hash=digest(manifest["files"]),
            training_dataset_hash=str(manifest["train_digest"]),
            task_signature=task_signature,
            steps=int(low_level["steps"]),
            loss=float(low_level["loss"]),
            validation_loss=float(low_level["validation_loss"]),
            device=str(low_level["device"]),
            calibration_path=str(calibration_path) if calibration_path else None,
            evaluation_path=str(evaluation_path) if evaluation_path else None,
            calibration=calibration_artifact,
            evaluation=evaluation_report,
        )

    def calibrate(
        self,
        *,
        factor: FactorSpec,
        checkpoint: str | Path,
        dataset: str | Path,
        output: str | Path,
        allow_unreviewed: bool = False,
    ) -> dict[str, Any]:
        self._validate_factor_datasets(factor, [dataset])
        return _calibrate(
            checkpoint,
            dataset,
            output,
            device=self.device,
            allow_unreviewed=allow_unreviewed,
        )

    def evaluate(
        self,
        *,
        factor: FactorSpec,
        checkpoint: str | Path,
        dataset: str | Path,
        output: str | Path,
        calibration: str | Path | None = None,
        threshold: float = 0.95,
        allow_unreviewed: bool = False,
    ) -> EvaluationReport:
        self._validate_factor_datasets(factor, [dataset])
        payload = _evaluate(
            checkpoint,
            dataset,
            output,
            calibration=calibration,
            device=self.device,
            threshold=threshold,
            allow_unreviewed=allow_unreviewed,
        )
        return EvaluationReport.from_payload(payload)

    @staticmethod
    def _validate_factor_datasets(factor: FactorSpec, paths: list[str | Path]) -> str:
        expected_labels = set(factor.labels)
        task_signature = None
        for path in paths:
            rows = read_rows(path)
            if not rows:
                raise ValueError(f"Factor dataset is empty: {path}")
            for row in rows:
                if row.category is not None or row.question is None:
                    raise ValueError(
                        "Factor training requires custom-question rows; registry category rows are not "
                        "a Factor dataset"
                    )
                if set(row.question.criteria) != expected_labels:
                    raise ValueError(
                        f"Factor {factor.name} labels do not match dataset question alternatives"
                    )
                current = digest(row.question.model_dump())
                if task_signature is None:
                    task_signature = current
                elif current != task_signature:
                    raise ValueError(
                        "A Factor dataset must use one stable question instruction and criteria contract"
                    )
        if task_signature is None:
            raise ValueError("Factor training requires at least one example")
        return task_signature
