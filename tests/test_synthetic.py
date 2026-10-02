import gzip
import json
from collections import defaultdict
from importlib.resources import files

import pytest

from starlings.datasets import assert_disjoint, read_rows, write_json
from starlings.encoder_demo import train_encoder
from starlings.registry import load_registry
from starlings.schemas import Registry
from starlings.synthetic import export_dataset, generate


def test_4000_bundle_all_category_all_split_coverage_and_pairing(tmp_path):
    manifest = export_dataset(tmp_path / "demo")
    assert manifest["rows"] == 4000
    assert {name: s["rows"] for name, s in manifest["splits"].items()} == {
        "train": 2488,
        "validation": 504,
        "calibration": 504,
        "test": 504,
    }
    categories = {c.id for c in load_registry().categories}
    splits = [read_rows(tmp_path / f"demo/{name}.jsonl") for name in manifest["splits"]]
    assert_disjoint(*splits)
    groups = defaultdict(list)
    for subset in splits:
        assert {r.category for r in subset} == categories
        by_category = defaultdict(set)
        for row in subset:
            by_category[row.category].add(row.label)
            groups[row.group_id].append(row)
            assert row.review.status == "pending" and row.review.reviewer is None
            assert "example.invalid" in row.state["provenance"]["source_uri"]
            assert row.state["provenance"]["fictional"] is True
            assert "label" not in row.state and "rationale" not in row.state
            assert row.state["facts"]["contract"]["relationship"]
        assert all(
            labels == {"applicable", "not_applicable", "insufficient_evidence"}
            for labels in by_category.values()
        )
    for rows in groups.values():
        controlled = next(r for r in rows if "/controlled:" in r.review.rationale)
        released = next(r for r in rows if "/released:" in r.review.rationale)
        assert controlled.content == released.content
        assert controlled.label != released.label
        assert controlled.state["facts"]["handling"]["controls_in_force"] is True
        assert released.state["facts"]["handling"]["controls_in_force"] is False
    assert len(read_rows(tmp_path / "demo/examples.jsonl")) == 4000
    with pytest.raises(FileExistsError):
        export_dataset(tmp_path / "demo")


def test_reproducible_generator_and_explicit_unknowns():
    a, manifest, _ = generate(seed=42)
    b, other, _ = generate(seed=42)
    assert manifest == other
    assert [r.model_dump() for r in a["test"]] == [r.model_dump() for r in b["test"]]
    _, changed, _ = generate(seed=7)
    assert changed["sha256"] != manifest["sha256"]
    missing = next(r for r in a["train"] if "/missing_source:" in r.review.rationale)
    assert missing.state["facts"]["source_type"] == "unknown"
    assert missing.state["facts"]["handling"]["controls_in_force"] is None
    assert missing.label == "insufficient_evidence"
    with pytest.raises(ValueError, match="supports"):
        generate(count=10)
    packed = files("starlings.data").joinpath("cui-demo-4000.jsonl.gz").read_bytes()
    assert len(gzip.decompress(packed).splitlines()) == 4000


def test_train_encoder_end_to_end_on_small_complete_registry(tmp_path):
    registry = load_registry()
    small = Registry(
        **{**registry.model_dump(), "categories": [c.model_dump() for c in registry.categories[:2]]}
    )
    write_json(tmp_path / "registry.json", small.model_dump())
    export_dataset(
        tmp_path / "data", count=64, registry_path=tmp_path / "registry.json", bundled=False
    )
    result = train_encoder(
        tmp_path / "run",
        dataset=tmp_path / "data",
        device="cpu",
        hidden_size=32,
        layers=1,
        heads=4,
        vocab_size=512,
        pretrain_steps=1,
        max_steps=1,
        grad_accum=1,
    )
    assert result["rows"] == 64 and result["test_rows"] == 8
    assert result["experimental"]
    report = json.loads((tmp_path / "run/benchmark.json").read_text())
    assert report["experimental"] and report["calibration_coverage"] == 1
    assert report["unrepresented_categories"] == []
    checkpoint = json.loads((tmp_path / "run/encoder/manifest.json").read_text())
    assert checkpoint["random_initialization"] and checkpoint["allow_unreviewed"]
    assert checkpoint["stage"] == "decision"
    with pytest.raises(FileExistsError):
        train_encoder(tmp_path / "run", dataset=tmp_path / "data", device="cpu")


def test_cli_export_and_bad_encoder_parameters(tmp_path):
    from starlings.cli import main

    assert main(["data", "cui-demo", "--output", str(tmp_path / "data")]) == 0
    assert (
        main(
            [
                "train-encoder",
                "--output",
                str(tmp_path / "bad"),
                "--hidden-size",
                "31",
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert not (tmp_path / "bad").exists()
