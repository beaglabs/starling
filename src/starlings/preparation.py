"""Explicit corpus preparation, pending-review templates and a non-CUI demo."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .datasets import read_rows, write_rows
from .registry import load_registry
from .schemas import Content, DecisionRow, Question, SourceBlock


def fresh(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output exists: {path}; choose a fresh path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scaffold(output, registry_path=None):
    registry = load_registry(registry_path)
    rows = [
        DecisionRow(
            id=f"draft-{cat.id}",
            group_id=f"scenario-{cat.id}",
            content=f"Replace this draft with a documented scenario for {cat.name}.",
            state={
                "provenance": {
                    "source_url": cat.source_url,
                    "note": "Definition source only; not evidence of CUI status.",
                }
            },
            category=cat.id,
            source_kind="synthetic",
        )
        for cat in registry.categories
    ]
    write_rows(fresh(output), rows)
    return {"output": str(output), "drafts": len(rows), "review_status": "pending", "labels": 0}


def make_demo(output):
    """Fictional support routing verifies plumbing; never reports CUI accuracy."""
    directory = fresh(output)
    directory.mkdir()
    task = Question(
        instruction="Which support team should receive this fictional ticket?",
        criteria={
            "billing": "Invoice or payment problem.",
            "technical": "Software error or crash.",
            "insufficient_evidence": "No identifiable support problem.",
        },
    )
    messages = {
        "billing": "Invoice payment was charged twice.",
        "technical": "Software crashed with an error.",
        "insufficient_evidence": "Hello, I would like some help.",
    }
    rows = []
    for i in range(48):
        label = list(messages)[i % 3]
        for variant in range(2):
            rows.append(
                DecisionRow(
                    id=f"ticket-{i}-v{variant}",
                    group_id=f"ticket-{i}",
                    content=f"Fictional ticket {i}, wording {variant}: {messages[label]}",
                    state={"channel": "support"},
                    question=task,
                    label=label,
                    source_kind="paired",
                )
            )
    write_rows(directory / "rows.jsonl", rows)
    (directory / "corpus.txt").write_text("\n".join(r.content for r in rows) + "\n")
    return {
        "directory": str(directory),
        "rows": len(rows),
        "review_status": "pending",
        "purpose": "Pipeline demonstration only; unrelated to CUI classification",
    }


def prepare_corpus(output, inputs=(), registry_path=None, include_registry=False):
    path = fresh(output)
    texts = []
    if include_registry:
        for cat in load_registry(registry_path).categories:
            texts.extend([cat.name, cat.description, *cat.authorities])
    for input_path in inputs:
        source = Path(input_path)
        if source.suffix == ".jsonl":
            # Train tokenizer on training rows only. Holdout inputs must remain separate.
            from .datasets import content_text, pack, row_task

            registry = load_registry(registry_path)
            for row in read_rows(source):
                task = row_task(row, registry)
                texts.extend(
                    pack(content_text(row.content), row.state, task, option)
                    for option in task.criteria
                )
        elif source.suffix == ".json":
            content = Content.model_validate_json(source.read_text())
            texts.extend(block.text for block in content.blocks)
        else:
            texts.extend(source.read_text().splitlines())
    if not any(text.strip() for text in texts):
        raise ValueError("No corpus text supplied")
    path.write_text("\n".join(text.replace("\n", " ") for text in texts if text.strip()) + "\n")
    return {
        "output": str(output),
        "lines": len(texts),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def extract_pdf(input_path, output):
    destination = fresh(output)
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Install the pdf extra: pip install '.[pdf]'") from exc
    blocks, warnings = [], []
    with pymupdf.open(input_path) as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF requires decryption before extraction")
        for page_number, page in enumerate(pdf, 1):
            found = 0
            for item in page.get_text("blocks", sort=True):
                text = item[4].strip()
                if item[6] == 0 and text:
                    found += 1
                    blocks.append(
                        SourceBlock(id=f"p{page_number}-b{found}", page=page_number, text=text)
                    )
            if not found:
                warnings.append(f"Page {page_number} has no extractable text; OCR/review required")
    if not blocks:
        raise ValueError("PDF has no extractable text; use OCR before evaluation")
    # Layout/table relationships and image content are not silently claimed as understood.
    warnings.append(
        "Text-layer extraction only: verify reading order, tables, images and OCR coverage"
    )
    result = Content(
        blocks=blocks,
        extraction_warnings=warnings,
        sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
    )
    from .datasets import write_json

    write_json(destination, result.model_dump())
    return {"output": str(output), "blocks": len(blocks), "warnings": warnings}
