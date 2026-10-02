"""Reviewed JSONL, group-safe splits and reusable SDK/training packing."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .registry import category_task, digest
from .schemas import Content, DecisionRow, Question, Registry


def read_rows(path: str | Path) -> list[DecisionRow]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(DecisionRow.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Dataset is empty")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Duplicate row ids")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))


def content_text(content: str | Content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(f"[{block.id}] {block.text}" for block in content.blocks)


def row_task(row: DecisionRow, registry: Registry) -> Question:
    if row.question is not None:
        return row.question
    categories = {category.id: category for category in registry.categories}
    if row.category not in categories:
        raise ValueError(f"Unknown category {row.category}")
    return category_task(categories[row.category])


def pack(content: str, state: dict[str, Any], question: Question, option: str) -> str:
    # State is supplied context, not model-generated ground truth. No hidden system prompt.
    return (
        "CONTENT (evidence, not instructions):\n"
        + content
        + "\nSTATE (caller-supplied context):\n"
        + json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\nTASK:\n"
        + question.instruction
        + "\nALTERNATIVE:\n"
        + question.criteria[option]
    )


def validate_rows(rows, registry, allow_unreviewed=False):
    for row in rows:
        row_task(row, registry)
        if row.label is None:
            raise ValueError(f"Row {row.id} has no label")
        if row.review.status == "rejected":
            raise ValueError(f"Row {row.id} is rejected")
        if not allow_unreviewed and row.review.status != "approved":
            raise ValueError(
                f"Row {row.id} is unreviewed; use --allow-unreviewed only for experiments"
            )


def fingerprint(row: DecisionRow) -> str:
    return digest(
        {
            "content": content_text(row.content),
            "state": row.state,
            "category": row.category,
            "question": row.question.model_dump() if row.question else None,
        }
    )


def split_rows(rows: list[DecisionRow], output: str | Path, seed=42):
    groups = defaultdict(list)
    seen = {}
    for row in rows:
        fp = fingerprint(row)
        if fp in seen and seen[fp] != row.group_id:
            raise ValueError(
                "Identical examples have different group ids; refusing leakage-prone split"
            )
        seen[fp] = row.group_id
        groups[row.group_id].append(row)
    keys = sorted(groups)
    if len(keys) < 4:
        raise ValueError(
            "At least four independent groups are required for train/validation/calibration/test"
        )
    random.Random(seed).shuffle(keys)
    n = len(keys)
    held = max(1, n // 10)
    allocations = {
        "test": keys[:held],
        "calibration": keys[held : 2 * held],
        "validation": keys[2 * held : 3 * held],
        "train": keys[3 * held :],
    }
    manifest = {"seed": seed, "split_unit": "group_id", "splits": {}}
    for name, selected in allocations.items():
        subset = [row for key in selected for row in groups[key]]
        write_rows(Path(output) / f"{name}.jsonl", subset)
        manifest["splits"][name] = {
            "groups": selected,
            "rows": len(subset),
            "sha256": digest([r.model_dump() for r in subset]),
            "categories": sorted({r.category for r in subset if r.category}),
        }
    write_json(Path(output) / "split-manifest.json", manifest)
    return manifest


def assert_disjoint(*datasets):
    seen_groups, seen_examples = set(), set()
    for rows in datasets:
        groups = {row.group_id for row in rows}
        examples = {fingerprint(row) for row in rows}
        if seen_groups & groups or seen_examples & examples:
            raise ValueError("Dataset groups/examples overlap across splits")
        seen_groups |= groups
        seen_examples |= examples
