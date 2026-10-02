from pathlib import Path

import pytest
import torch

from starlings.datasets import read_rows, split_rows, write_json, write_rows
from starlings.evaluation import calibrate
from starlings.preparation import make_demo, prepare_corpus
from starlings.schemas import Category, Registry, Review
from starlings.tokenizer import train_tokenizer
from starlings.training import initialize, pretrain, train


@pytest.fixture(scope="session", autouse=True)
def cpu_budget():
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    registry = Registry(
        version="test-registry",
        retrieved_at="2026-10-02",
        source_url="https://example.test",
        categories=[
            Category(
                id=name,
                name=name,
                group="test",
                description="Test category definition.",
                source_url="https://example.test",
                source_sha256="test",
            )
            for name in ["test_category", "other_category"]
        ],
    )
    write_json(root / "registry.json", registry.model_dump())
    make_demo(root / "demo")
    rows = read_rows(root / "demo/rows.jsonl")
    for row in rows:
        row.review = Review(
            status="approved", reviewer="test-fixture", rationale="Fictional routing fixture only."
        )
    write_rows(root / "reviewed.jsonl", rows)
    split_rows(rows, root / "splits")
    prepare_corpus(root / "corpus.txt", [root / "splits/train.jsonl"], root / "registry.json")
    train_tokenizer((root / "corpus.txt").read_text().splitlines(), root / "tokenizer.json", 512)
    initialize(
        root / "tokenizer.json",
        root / "init",
        {
            "hidden_size": 32,
            "layers": 1,
            "heads": 4,
            "intermediate_size": 64,
            "max_length": 512,
            "dropout": 0,
        },
        root / "registry.json",
        seed=7,
    )
    pretrain(root / "init", root / "corpus.txt", root / "pretrained", steps=2, device="cpu")
    train(
        root / "pretrained",
        root / "splits/train.jsonl",
        root / "splits/validation.jsonl",
        root / "trained",
        max_steps=3,
        grad_accum=2,
        device="cpu",
    )
    calibrate(
        root / "trained", root / "splits/calibration.jsonl", root / "calibration.json", device="cpu"
    )
    return Path(root)


@pytest.fixture(autouse=True)
def restore_runtime_settings():
    deterministic = torch.are_deterministic_algorithms_enabled()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.mha.set_fastpath_enabled(fastpath)
