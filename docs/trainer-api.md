# Public training API

Starlings exposes a supported Python training facade for reusable **Factors**: bounded learned judgments with one stable question contract and a fixed set of alternatives.

```python
from starlings import FactorSpec, Trainer, TrainingConfig

factor = FactorSpec(
    name="technical-readiness",
    labels=["low", "medium", "high"],
    description="Reusable technical-readiness judgment.",
)

trainer = Trainer(device="auto")
result = trainer.train(
    factor=factor,
    checkpoint="runs/pretrained",
    train="splits/train.jsonl",
    validation="splits/validation.jsonl",
    calibration="splits/calibration.jsonl",
    test="splits/test.jsonl",
    output="runs/technical-readiness-v1",
    config=TrainingConfig(
        epochs=5,
        learning_rate=3e-4,
        grad_accum=8,
        seed=42,
    ),
)

print(result.model_id)
print(result.pipeline_hash)
print(result.training_dataset_hash)
print(result.evaluation.accuracy if result.evaluation else None)
```

The public facade deliberately does not expose arbitrary training code. It delegates to the same Starlings trainer, calibration, evaluation, checkpoint-integrity, and leakage-protection implementation used by the CLI.

## Factor dataset contract

A Factor dataset uses `DecisionRow.question`, not registry-category rows. Every row in every supplied split must use the same `Question` instruction and criteria descriptions, and the criteria keys must exactly match `FactorSpec.labels`.

This is stricter than merely checking class names: changing the instruction or the meaning of an alternative creates a different task and must be trained/evaluated as a different Factor contract.

## Separate lifecycle operations

The facade also exposes calibration and evaluation independently:

```python
trainer.calibrate(
    factor=factor,
    checkpoint="runs/technical-readiness-v1",
    dataset="splits/calibration.jsonl",
    output="runs/technical-readiness-v1.calibration.json",
)

report = trainer.evaluate(
    factor=factor,
    checkpoint="runs/technical-readiness-v1",
    dataset="splits/test.jsonl",
    calibration="runs/technical-readiness-v1.calibration.json",
    output="runs/technical-readiness-v1.evaluation.json",
)
```

`TrainingResult` returns the model identity, pipeline hash, training-corpus hash, task signature, training metrics, and optional calibration/evaluation artifacts so a host such as Papyrus can record provenance and apply its own promotion or publication policy.

The current Factor API supports choice/classification tasks backed by Starlings' existing decision encoder. Additional encoder backends or task types should be added behind this public contract rather than by having host applications import Starlings internals.
