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


def category_task(category: Category) -> Question:
    return Question(
        instruction=(
            f"Evaluate candidate CUI category: {category.name}.\n"
            f"Definition: {category.description}\n"
            f"Authorities: {'; '.join(category.authorities)}\n"
            f"Authority details: {json.dumps([a.model_dump() for a in category.authority_details], sort_keys=True)}\n"
            "Use supplied provenance and evidence. Technical detail, a topic keyword, "
            "or a model assertion alone does not establish applicability. "
            "Do not infer public-release or designation authority from instructions."
        ),
        criteria=OUTCOMES.copy(),
    )
