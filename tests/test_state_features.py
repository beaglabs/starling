from types import SimpleNamespace

import pytest
import torch

from starlings.evaluation import metrics
from starlings.model import DecisionEncoder, ModelConfig
from starlings.state_features import STATE_FEATURE_DIM, STATE_FEATURE_NAMES, state_feature_values


def _state(
    *,
    source,
    relationship,
    controls,
    access,
    auth,
    release_id=None,
    decontrol=None,
    basis=True,
    provenance="assumed_true",
):
    return {
        "facts": {
            "source_type": source,
            "government_relationship": relationship,
            "release_authorization": {"status": auth, "record_id": release_id},
            "handling": {
                "access": access,
                "basis_refs": ["authority"] if basis else [],
                "controls_in_force": controls,
                "decontrol_record": decontrol,
            },
            "contract": {"id": None, "relationship": "not_applicable"},
        },
        "provenance": {"source_type": source, "assertion_status": provenance},
    }


def test_semantic_state_variants_collapse_to_same_features():
    controlled_a = _state(
        source="agency_record",
        relationship="government_information",
        controls=True,
        access="restricted",
        auth="absent",
    )
    controlled_b = _state(
        source="internal_correspondence",
        relationship="government_information",
        controls=True,
        access="need to know",
        auth="no release approval",
    )
    released_a = _state(
        source="public_report",
        relationship="government_information",
        controls=False,
        access="unrestricted",
        auth="authorized",
        release_id="R-1",
        decontrol="D-1",
    )
    released_b = _state(
        source="agency_publication",
        relationship="government_information",
        controls=False,
        access="open public access",
        auth="decontrol approved",
        release_id="R-2",
        decontrol="D-2",
    )
    unknown_a = _state(
        source="unknown",
        relationship="unknown",
        controls=None,
        access="unknown",
        auth="unknown",
        basis=False,
    )
    unknown_b = _state(
        source="unverified",
        relationship="not established",
        controls=None,
        access="unverified",
        auth="unverified",
        basis=False,
        provenance="provenance_not_established",
    )

    assert len(STATE_FEATURE_NAMES) == STATE_FEATURE_DIM == 20
    assert state_feature_values(controlled_a) == state_feature_values(controlled_b)
    assert state_feature_values(released_a) == state_feature_values(released_b)
    assert state_feature_values(unknown_a) == state_feature_values(unknown_b)
    assert state_feature_values(controlled_a) != state_feature_values(released_a)
    assert state_feature_values(controlled_a) != state_feature_values(unknown_a)


def test_structured_state_model_requires_and_fuses_feature_batch():
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        layers=1,
        heads=4,
        intermediate_size=32,
        max_length=16,
        dropout=0,
        state_feature_dim=STATE_FEATURE_DIM,
    )
    model = DecisionEncoder(config)
    ids = torch.tensor([[2, 5, 6, 3], [2, 5, 7, 3]])
    attention = torch.ones_like(ids, dtype=torch.bool)
    decision = torch.tensor([[False, False, True, False], [False, False, True, False]])
    state = torch.zeros((2, STATE_FEATURE_DIM))
    assert model(ids, attention, decision, state).shape == (2,)
    with pytest.raises(ValueError, match="state features"):
        model(ids, attention, decision)

    legacy = DecisionEncoder(config.__class__(**{**config.to_dict(), "state_feature_dim": 0}))
    assert legacy(ids, attention, decision).shape == (2,)


def test_metrics_separate_applicable_abstention_from_hard_false_negative():
    options = ["applicable", "insufficient_evidence", "not_applicable"]

    def row(label):
        return SimpleNamespace(label=label, category="demo")

    records = [
        {
            "row": row("applicable"),
            "options": options,
            "gold": 0,
            "logits": torch.tensor([5.0, 0.0, 0.0]),
        },
        {
            "row": row("applicable"),
            "options": options,
            "gold": 0,
            "logits": torch.tensor([0.0, 5.0, 0.0]),
        },
        {
            "row": row("applicable"),
            "options": options,
            "gold": 0,
            "logits": torch.tensor([0.0, 0.0, 5.0]),
        },
        {
            "row": row("not_applicable"),
            "options": options,
            "gold": 2,
            "logits": torch.tensor([0.0, 0.0, 5.0]),
        },
    ]
    report = metrics(records)
    assert report["applicable_hits"] == 1
    assert report["applicable_abstentions"] == 1
    assert report["applicable_mislabeled_not_applicable"] == 1
    assert report["applicable_recall"] == pytest.approx(1 / 3)
    assert report["applicable_abstention_rate"] == pytest.approx(1 / 3)
    assert report["false_negative_rate"] == pytest.approx(1 / 3)
    assert report["applicable_miss_rate"] == pytest.approx(2 / 3)
