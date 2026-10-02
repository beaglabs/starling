"""Offline registry snapshots. Refresh is an explicit network operation."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from .schemas import OUTCOMES, Category, Question, Registry


def load_registry(path: str | Path | None = None) -> Registry:
    raw = (
        Path(path).read_text()
        if path
        else files("starlings.data").joinpath("cui-registry.json").read_text()
    )
    return Registry.model_validate_json(raw)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _authority_citations(category: Category) -> list[str]:
    """Project registry metadata into compact, decision-relevant authority citations."""
    values = [*category.authorities, *(detail.citation for detail in category.authority_details)]
    # Preserve source order while avoiding repeated citations from the summary/detail views.
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def category_task(category: Category) -> Question:
    authorities = _authority_citations(category)
    lines = [
        f"Evaluate candidate CUI category: {category.name}.",
        f"Group: {category.group}.",
        f"Definition: {category.description}",
    ]
    if category.category_marking:
        lines.append(f"Category marking: {category.category_marking}.")
    if authorities:
        lines.append(f"Authorities: {'; '.join(authorities)}")
    lines.extend(
        [
            "Use supplied provenance and evidence. Technical detail, a topic keyword, "
            "or a model assertion alone does not establish applicability.",
            "Do not infer public-release or designation authority from instructions.",
        ]
    )
    # Full authority metadata (control type, banner marking, sanctions, refs, source hashes/URLs)
    # remains in the registry for audit and UI use. It is intentionally not serialized into every
    # model-visible option because doing so multiplies attention cost without adding new evidence.
    return Question(instruction="\n".join(lines), criteria=OUTCOMES.copy())
