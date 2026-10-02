"""One-command from-scratch encoder experiment on the bundled CUI fixtures."""

from __future__ import annotations

import math
from pathlib import Path

from .datasets import (
    assert_disjoint,
    content_text,
    pack,
    read_rows,
    row_task,
    validate_rows,
    write_json,
)
from .evaluation import calibrate, evaluate
from .model import select_device
from .preparation import fresh, prepare_corpus
from .registry import load_registry
from .synthetic import export_dataset
from .tokenizer import train_tokenizer
from .training import initialize, pretrain, train


def train_encoder(
    output,
    *,
    dataset=None,
    device="auto",
    seed=42,
    vocab_size=8000,
    hidden_size=128,
    layers=2,
    heads=4,
    epochs=1,
    pretrain_steps=50,
    max_steps=None,
    grad_accum=4,
    lr=3e-4,
):
    # Fail unavailable devices before generating files or spending tokenizer compute.
    selected = select_device(device)
    from .training import check_training_args

    check_training_args(epochs, lr, grad_accum, max_steps)
    if pretrain_steps < 0:
        raise ValueError("pretrain_steps must be nonnegative")
    if hidden_size < 2 or layers < 1 or heads < 1 or hidden_size % heads:
        raise ValueError("Invalid encoder dimensions")
    root = fresh(output)
    if dataset is None:
        export_dataset(root / "data")
        data = root / "data"
    else:
        data = Path(dataset)
    registry_path = data / "registry.json"
    registry = load_registry(registry_path)
    split_rows = {
        name: read_rows(data / f"{name}.jsonl")
        for name in ["train", "validation", "calibration", "test"]
    }
    assert_disjoint(*split_rows.values())
    for rows in split_rows.values():
        validate_rows(rows, registry, allow_unreviewed=True)
    root.mkdir(parents=True, exist_ok=True)
    prepare_corpus(
        root / "corpus.txt", [data / "train.jsonl"], registry_path, include_registry=True
    )
    tokenizer = train_tokenizer(
        (root / "corpus.txt").read_text().splitlines(), root / "tokenizer.json", vocab_size
    )
    # Choose context from train/validation only; no held-out text is used to fit tokenizer,
    # language weights, or select context length. Definitions are the fixed shared policy.
    longest = max(
        len(tokenizer.encode(pack(content_text(r.content), r.state, row_task(r, registry), o)).ids)
        + 2
        for name in ["train", "validation"]
        for r in split_rows[name]
        for o in row_task(r, registry).criteria
    )
    context = max(512, math.ceil((longest + 256) / 128) * 128)
    initialize(
        root / "tokenizer.json",
        root / "random",
        {
            "hidden_size": hidden_size,
            "layers": layers,
            "heads": heads,
            "intermediate_size": hidden_size * 4,
            "max_length": context,
        },
        registry_path,
        seed,
    )
    import json

    print(
        json.dumps(
            {
                "stage": "prepared",
                "context_tokens": context,
                "longest_training_input": longest,
                "device": str(selected),
                "training_rows": len(split_rows["train"]),
            }
        ),
        flush=True,
    )
    starting = root / "random"
    if pretrain_steps:
        pretrain(
            starting,
            root / "corpus.txt",
            root / "pretrained",
            steps=pretrain_steps,
            batch_size=1,
            device=str(selected),
            seed=seed,
            lr=lr,
        )
        starting = root / "pretrained"
    train(
        starting,
        data / "train.jsonl",
        data / "validation.jsonl",
        root / "encoder",
        epochs=epochs,
        lr=lr,
        grad_accum=grad_accum,
        device=str(selected),
        seed=seed,
        max_steps=max_steps,
        allow_unreviewed=True,
    )
    calibrate(
        root / "encoder",
        data / "calibration.jsonl",
        root / "calibration.json",
        device=str(selected),
        allow_unreviewed=True,
    )
    report = evaluate(
        root / "encoder",
        data / "test.jsonl",
        root / "benchmark.json",
        calibration=root / "calibration.json",
        device=str(selected),
        allow_unreviewed=True,
    )
    summary = {
        "checkpoint": str(root / "encoder"),
        "calibration": str(root / "calibration.json"),
        "benchmark": str(root / "benchmark.json"),
        "device": str(selected),
        "context_tokens": context,
        "rows": sum(len(rs) for rs in split_rows.values()),
        "test_rows": report["samples"],
        "accuracy": report["accuracy"],
        "experimental": True,
        "note": "Agreement with unreviewed synthetic labels; not validated operational CUI performance.",
    }
    write_json(root / "run.json", summary)
    return summary
