import json

import pytest
from pydantic import ValidationError

from starlings.datasets import assert_disjoint, read_rows, split_rows
from starlings.model import ModelConfig
from starlings.preparation import extract_pdf, make_demo, scaffold
from starlings.registry import category_task, load_registry
from starlings.schemas import Content, DecisionRow, Question, Review


def test_all_categories_are_pinned_and_scaffold_is_unreviewed(tmp_path):
    registry = load_registry()
    assert len(registry.categories) == 126
    assert all(
        c.source_url.startswith("https://www.archives.gov/") and len(c.source_sha256) == 64
        for c in registry.categories
    )
    assert all(category_task(c).criteria for c in registry.categories)
    cti = next(c for c in registry.categories if c.id == "controlled_technical_information")
    assert "48 CFR 252.204-7012" in cti.authorities
    assert cti.authority_details[0].control_type == "Specified"
    scaffold(tmp_path / "drafts.jsonl")
    rows = read_rows(tmp_path / "drafts.jsonl")
    assert {r.category for r in rows} == {c.id for c in registry.categories}
    assert all(r.label is None and r.review.status == "pending" for r in rows)


def test_group_split_and_duplicate_content_detection(tmp_path):
    make_demo(tmp_path / "demo")
    rows = read_rows(tmp_path / "demo/rows.jsonl")
    split_rows(rows, tmp_path / "splits")
    splits = [
        read_rows(tmp_path / f"splits/{name}.jsonl")
        for name in ["train", "validation", "calibration", "test"]
    ]
    assert_disjoint(*splits)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(splits[0], splits[0])
    rows.append(rows[0].model_copy(update={"id": "copied", "group_id": "different"}))
    with pytest.raises(ValueError, match="Identical"):
        split_rows(rows, tmp_path / "bad")


def test_schema_rejects_ambiguous_tasks_unknown_fields_and_nonfinite_targets():
    with pytest.raises(ValidationError):
        Content(blocks=[{"id": "x", "text": "a"}, {"id": "x", "text": "b"}])
    with pytest.raises(ValidationError):
        Review(status="approved")
    with pytest.raises(ValidationError):
        Question(instruction="task", criteria={"one": "only option"})
    with pytest.raises(ValidationError):
        DecisionRow(
            id="x",
            group_id="x",
            content="text",
            category="x",
            question=Question(instruction="task"),
            source_kind="public",
        )
    with pytest.raises(ValidationError):
        DecisionRow(
            id="x",
            group_id="x",
            content="text",
            category="x",
            target={"applicable": float("nan"), "not_applicable": 0, "insufficient_evidence": 1},
            source_kind="public",
        )
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=31, heads=4)


def test_pdf_stable_blocks_and_empty_page_warning(tmp_path):
    import pymupdf

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Public fictional specification.")
        pdf.new_page()
        pdf.save(tmp_path / "document.pdf")
    extract_pdf(tmp_path / "document.pdf", tmp_path / "content.json")
    content = Content.model_validate_json((tmp_path / "content.json").read_text())
    assert content.blocks[0].id == "p1-b1" and content.blocks[0].page == 1
    assert len(content.sha256) == 64
    assert any("Page 2" in warning for warning in content.extraction_warnings)
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(tmp_path / "empty.pdf")
    with pytest.raises(ValueError, match="OCR"):
        extract_pdf(tmp_path / "empty.pdf", tmp_path / "empty.json")


def test_cli_predict_and_errors(pipeline, tmp_path):
    from starlings.cli import main

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"content": "Example specification", "categories": ["test_category"]})
    )
    assert (
        main(
            [
                "--threads",
                "1",
                "predict",
                "--checkpoint",
                str(pipeline / "trained"),
                "--request",
                str(request),
                "--output",
                str(tmp_path / "result.json"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["requires_review"]
    assert main(["--threads", "0", "doctor"]) == 2
    request.write_text('{"content":"text","unsupported":true}')
    assert (
        main(
            [
                "predict",
                "--checkpoint",
                str(pipeline / "trained"),
                "--request",
                str(request),
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )
        == 2
    )
