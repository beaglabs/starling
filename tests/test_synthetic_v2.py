from collections import Counter, defaultdict

from starlings.datasets import content_text, read_rows
from starlings.registry import load_registry
from starlings.synthetic_v2 import GENERATOR_VERSION, PROFILES, export_benchmark, generate_benchmark


def test_adversarial_benchmark_is_balanced_reproducible_and_all_category(tmp_path):
    rows, manifest, _ = generate_benchmark(seed=2026)
    again, same, _ = generate_benchmark(seed=2026)
    changed, changed_manifest, _ = generate_benchmark(seed=7)

    categories = {category.id for category in load_registry().categories}
    assert manifest == same
    assert [row.model_dump() for row in rows] == [row.model_dump() for row in again]
    assert changed_manifest["sha256"] != manifest["sha256"]
    assert [row.id for row in changed] != [row.id for row in rows]

    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["intended_use"] == "external_holdout_evaluation_only"
    assert manifest["rows"] == 756
    assert manifest["categories"] == 126
    assert manifest["examples_per_category"] == 6
    assert set(manifest["profiles"]) == set(PROFILES)
    assert manifest["labels"] == {
        "applicable": 252,
        "not_applicable": 252,
        "insufficient_evidence": 252,
    }
    assert {row.category for row in rows} == categories
    assert all(count == 6 for count in Counter(row.category for row in rows).values())

    labels_by_category = defaultdict(Counter)
    for row in rows:
        labels_by_category[row.category][row.label] += 1
        assert row.group_id.startswith("adv2-")
        assert row.state["provenance"]["fictional"] is True
        assert row.review.status == "pending" and row.review.reviewer is None
        assert "label" not in row.state and "rationale" not in row.state
        assert "Fictional " not in content_text(row.content)
    assert all(
        counts == Counter({"applicable": 2, "not_applicable": 2, "insufficient_evidence": 2})
        for counts in labels_by_category.values()
    )

    exported = export_benchmark(tmp_path / "v2")
    assert exported["sha256"] == manifest["sha256"]
    saved = read_rows(tmp_path / "v2/benchmark.jsonl")
    assert len(saved) == 756
    assert (tmp_path / "v2/manifest.json").exists()
    assert (tmp_path / "v2/registry.json").exists()


def test_adversarial_profiles_force_content_state_disagreement():
    rows, _, _ = generate_benchmark()
    by_profile = {}
    for row in rows:
        profile = row.review.rationale.split("/", 1)[1].split(":", 1)[0]
        by_profile.setdefault(profile, row)

    assert set(by_profile) == set(PROFILES)

    controlled = by_profile["controlled_false_release_claim"]
    assert "approved for public posting" in content_text(controlled.content)
    assert controlled.label == "applicable"
    assert controlled.state["facts"]["handling"]["controls_in_force"] is True

    stale = by_profile["released_stale_banner"]
    assert "stale banner" in content_text(stale.content)
    assert "DO NOT DISTRIBUTE" in content_text(stale.content)
    assert stale.label == "not_applicable"
    assert stale.state["facts"]["handling"]["controls_in_force"] is False
    assert stale.state["facts"]["handling"]["decontrol_record"]

    conflict = by_profile["conflicting_release_chain"]
    assert conflict.label == "insufficient_evidence"
    assert conflict.state["facts"]["handling"]["controls_in_force"] is None
    assert "conflict" in conflict.state["facts"]["release_authorization"]["status"]

    unknown = by_profile["unknown_origin_strong_marking"]
    assert "CONTROLLED" in content_text(unknown.content)
    assert unknown.label == "insufficient_evidence"
    assert unknown.state["facts"]["handling"]["controls_in_force"] is None


def test_cli_exports_adversarial_benchmark(tmp_path):
    from starlings.cli import main

    output = tmp_path / "cui-v2"
    assert main(["data", "cui-demo-v2", "--output", str(output)]) == 0
    assert len(read_rows(output / "benchmark.jsonl")) == 756
