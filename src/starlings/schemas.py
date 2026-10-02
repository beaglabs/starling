"""Wire contracts. Caller assertions are evidence inputs, never authentication."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OUTCOMES = {
    "applicable": "The supplied evidence establishes that the category criteria apply.",
    "not_applicable": "The supplied evidence establishes that the category criteria do not apply.",
    "insufficient_evidence": "Required evidence or context is missing, conflicting or inconclusive.",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceBlock(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=1_000_000)
    page: int | None = Field(default=None, ge=1)


class Content(StrictModel):
    blocks: list[SourceBlock] = Field(min_length=1, max_length=10_000)
    extraction_warnings: list[str] = Field(default_factory=list)
    sha256: str | None = None

    @model_validator(mode="after")
    def unique_ids(self) -> Content:
        ids = [block.id for block in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("Source block ids must be unique")
        return self


class Authority(StrictModel):
    citation: str
    control_type: str = ""
    banner_marking: str = ""
    sanctions: str = ""
    refs: list[str] = Field(default_factory=list)


class Category(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=128)
    name: str
    group: str
    description: str = Field(min_length=1)
    category_marking: str = ""
    authority_details: list[Authority] = Field(default_factory=list)
    authority_refs: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    source_url: str
    source_sha256: str


class Registry(StrictModel):
    version: str
    retrieved_at: str
    source_url: str
    categories: list[Category] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Registry:
        ids = [category.id for category in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("Registry category ids must be unique")
        return self


class Question(StrictModel):
    instruction: str = Field(min_length=1, max_length=64000)
    criteria: dict[str, str] = Field(
        default_factory=lambda: {
            "yes": "The evidence supports yes.",
            "no": "The evidence supports no.",
            "insufficient_evidence": "The evidence does not establish an answer.",
        }
    )

    @model_validator(mode="after")
    def valid_criteria(self) -> Question:
        if not 2 <= len(self.criteria) <= 32:
            raise ValueError("Questions require 2..32 described alternatives")
        if any(not key or not value for key, value in self.criteria.items()):
            raise ValueError("Option ids and descriptions must not be empty")
        return self


class Candidate(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    evidence_refs: list[str] = Field(min_length=1, max_length=64)
    category_hypotheses: list[str] = Field(default_factory=list)
    quote: str | None = None


class Review(StrictModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer: str | None = None
    rationale: str | None = None
    authority_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def approved_has_record(self) -> Review:
        if self.status == "approved" and (not self.reviewer or not self.rationale):
            raise ValueError("Approved labels require a reviewer and rationale")
        return self


class DecisionRow(StrictModel):
    id: str
    group_id: str = Field(min_length=1)
    content: str | Content
    state: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    question: Question | None = None
    label: str | None = None
    target: dict[str, float] | None = None
    review: Review = Field(default_factory=Review)
    source_kind: Literal["official_scenario", "public", "synthetic", "paired", "operational"]

    @model_validator(mode="after")
    def one_task(self) -> DecisionRow:
        if (self.category is None) == (self.question is None):
            raise ValueError("Exactly one of category or question is required")
        criteria = OUTCOMES if self.category else self.question.criteria
        if self.label is not None and self.label not in criteria:
            raise ValueError("Label is not a defined alternative")
        if self.target is not None:
            if set(self.target) != set(criteria):
                raise ValueError("Soft targets must cover every alternative exactly")
            if any(not 0 <= p <= 1 for p in self.target.values()):
                raise ValueError("Target probabilities must be finite and in [0,1]")
            if abs(sum(self.target.values()) - 1) > 1e-6:
                raise ValueError("Target probabilities must sum to one")
        return self


class Finding(StrictModel):
    task_id: str
    block_refs: list[str]
    outcome: str
    probabilities: dict[str, float]
    confidence: float
    calibrated: bool
    requires_review: bool
    reasons: list[str] = Field(default_factory=list)
    scope: Literal["block", "candidate", "document", "question"]


class EvaluationResult(StrictModel):
    schema_version: str = "1"
    model_id: str
    registry_version: str
    evaluated_categories: list[str]
    unevaluated_categories: list[str]
    findings: list[Finding]
    question_findings: list[Finding]
    candidate_findings: dict[str, list[Finding]]
    coverage: dict[str, Any]
    requires_review: bool
    warnings: list[str]
    provenance: dict[str, Any]

    def filter_categories(self, categories: list[str]) -> list[Finding]:
        """Presentation filter; does not rerun inference or change coverage."""
        unknown = set(categories) - set(self.evaluated_categories)
        if unknown:
            raise ValueError(f"Categories were not evaluated: {sorted(unknown)}")
        return [finding for finding in self.findings if finding.task_id in categories]
