"""Deterministic normalization of caller-supplied control/release facts.

The text encoder still receives the complete JSON state for auditability and context. These
features provide a stable semantic path for high-value host facts so the decision model does not
have to relearn that every wording variant of release/control metadata means the same thing.
"""

from __future__ import annotations

from typing import Any

import torch

STATE_FEATURE_VERSION = 1
STATE_FEATURE_NAMES = (
    "source_known",
    "source_public",
    "government_information",
    "controls_in_force_true",
    "controls_in_force_false",
    "controls_in_force_unknown",
    "decontrol_record_present",
    "release_record_present",
    "release_authorization_approved",
    "release_authorization_not_approved",
    "release_authorization_conflicting",
    "release_authorization_unknown",
    "access_restricted",
    "access_unrestricted",
    "access_conflicting",
    "access_unknown",
    "provenance_conflicting",
    "control_basis_present",
    "contract_present",
    "contract_unknown",
)
STATE_FEATURE_DIM = len(STATE_FEATURE_NAMES)

_UNKNOWN = {"", "unknown", "none", "null", "not_established", "unverified", "unknown_from_available_records"}
_CONFLICT_TERMS = ("conflict", "disput", "contest", "disagree", "contradict")
_PUBLIC_SOURCE_TERMS = ("public", "publication", "archive", "reading_room")
_RESTRICTED_ACCESS_TERMS = ("restrict", "need_to_know", "controlled", "internal")
_UNRESTRICTED_ACCESS_TERMS = ("unrestrict", "open_public", "public_access")
_APPROVED_TERMS = ("authoriz", "approved", "decontrol", "cleared", "unrestricted")
_NOT_APPROVED_TERMS = ("absent", "denied", "not_approved", "no_approved", "no_release", "missing")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _unknown(value: Any) -> bool:
    return _text(value) in _UNKNOWN


def _contains(value: Any, terms: tuple[str, ...]) -> bool:
    text = _text(value)
    return any(term in text for term in terms)


def _release_authorization(status: Any) -> str:
    text = _text(status)
    if not text or text in _UNKNOWN:
        return "unknown"
    if any(term in text for term in _CONFLICT_TERMS):
        return "conflicting"
    # Check negation before positive stems: "no_approved_release" contains "approved".
    if any(term in text for term in _NOT_APPROVED_TERMS) or text.startswith("no_"):
        return "not_approved"
    if any(term in text for term in _APPROVED_TERMS):
        return "approved"
    return "unknown"


def _access_state(value: Any) -> str:
    text = _text(value)
    if not text or text in _UNKNOWN:
        return "unknown"
    if any(term in text for term in _CONFLICT_TERMS):
        return "conflicting"
    if any(term in text for term in _UNRESTRICTED_ACCESS_TERMS):
        return "unrestricted"
    if any(term in text for term in _RESTRICTED_ACCESS_TERMS):
        return "restricted"
    return "unknown"


def state_feature_values(state: dict[str, Any] | None) -> list[float]:
    """Project the documented state contract into stable, non-learned semantic indicators."""
    state = state if isinstance(state, dict) else {}
    facts = state.get("facts") if isinstance(state.get("facts"), dict) else {}
    provenance = state.get("provenance") if isinstance(state.get("provenance"), dict) else {}
    handling = facts.get("handling") if isinstance(facts.get("handling"), dict) else {}
    release = (
        facts.get("release_authorization")
        if isinstance(facts.get("release_authorization"), dict)
        else {}
    )
    contract = facts.get("contract") if isinstance(facts.get("contract"), dict) else {}

    source_type = facts.get("source_type", provenance.get("source_type"))
    relationship = facts.get("government_relationship")
    controls = handling.get("controls_in_force")
    auth = _release_authorization(release.get("status"))
    access = _access_state(handling.get("access"))
    provenance_status = provenance.get("assertion_status")
    basis_refs = handling.get("basis_refs")
    contract_relationship = contract.get("relationship")

    source_known = not _unknown(source_type)
    source_public = source_known and _contains(source_type, _PUBLIC_SOURCE_TERMS)
    government_information = _text(relationship) == "government_information"
    controls_true = controls is True
    controls_false = controls is False
    controls_unknown = controls not in (True, False)
    decontrol_present = bool(handling.get("decontrol_record"))
    release_record_present = bool(release.get("record_id"))
    provenance_conflicting = _contains(provenance_status, _CONFLICT_TERMS)
    basis_present = isinstance(basis_refs, (list, tuple)) and bool(basis_refs)
    contract_present = bool(contract.get("id"))
    contract_unknown = _unknown(contract_relationship)

    values = [
        source_known,
        source_public,
        government_information,
        controls_true,
        controls_false,
        controls_unknown,
        decontrol_present,
        release_record_present,
        auth == "approved",
        auth == "not_approved",
        auth == "conflicting",
        auth == "unknown",
        access == "restricted",
        access == "unrestricted",
        access == "conflicting",
        access == "unknown",
        provenance_conflicting,
        basis_present,
        contract_present,
        contract_unknown,
    ]
    if len(values) != STATE_FEATURE_DIM:
        raise RuntimeError("Structured-state feature contract mismatch")
    return [float(value) for value in values]


def state_feature_tensor(state: dict[str, Any] | None, device=None) -> torch.Tensor:
    return torch.tensor(state_feature_values(state), dtype=torch.float32, device=device)
