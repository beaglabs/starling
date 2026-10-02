# Python SDK and JSON contract

```python
from starlings import Classifier

model = Classifier(
    "runs/decision",
    device="auto",
    threshold=0.95,
    calibration=None,
    max_evaluations=4096,
    deterministic=False,
)
result = model.evaluate(content="Source text", state={}, categories="all")
```

`classify(content, *, state=None, categories="all")` is a convenience wrapper. `evaluate` also accepts `questions` and `candidates`. There is no remote service dependency or generative step. Both initialization and pretraining checkpoints are rejected; inference requires a decision checkpoint.

## Request

The same request works with `starlings predict --request request.json ...`. See [complete example](../examples/request.json).

| Input | Meaning |
| --- | --- |
| `content` | Nonempty string or source `Content` object |
| `state` | JSON object encoded as caller context; untrusted assertions are not authenticated |
| `categories` | `"all"` (default) or a list of exact registry IDs; `[]` permits question-only use |
| `questions` | Map of question IDs to instructions and described alternatives |
| `candidates` | Optional LLM hypotheses, each with an ID, source-block references, category hypotheses, and optional exact quote |

Source blocks have unique IDs, nonempty text, and optional 1-based page numbers. A content object may include `extraction_warnings` and a source file `sha256`. `questions` define their own 2–32 alternatives; categories always use the three CUI outcomes. Question wording and option descriptions affect task identity and calibration.

Candidate references must resolve; duplicate references, unknown categories, duplicate candidate IDs, and invented quotes fail before inference. Candidate hypotheses select additional joint-block checks **within** the caller-selected category scope. They never suppress the independent source scan. A quote must be an exact substring of the referenced source blocks joined with newlines; paraphrases belong outside the quote field.

## Evaluation scopes

1. Every source block is evaluated for each selected category. Long blocks use overlapping token windows.
2. Every selected category receives a separate whole-document pass to expose aggregation/context needs.
3. Candidate references are evaluated jointly against their category hypotheses or the selected categories.
4. Custom questions receive the whole document and state.

Every evaluation scores all described alternatives in one batch. `max_evaluations` limits these task evaluations, not the number of individual alternatives. Exceeding the budget raises an explicit error, with no partial result. Use a fresh narrower request or explicitly raise the budget.

Long whole-document or candidate inputs are not truncated. They produce `insufficient_evidence`, zero confidence, a context-overflow reason, and review status. Block scans continue, but this does not solve the mosaic effect for arbitrarily long documents. An external aggregation/review stage is needed when whole-document context does not fit. If state and policy text alone exceed the context, block evaluations also remain unresolved.

## Response

| Field | Meaning |
| --- | --- |
| `schema_version` | JSON contract version |
| `model_id`, `registry_version` | Artifact identities |
| `evaluated_categories`, `unevaluated_categories` | Explicit scope coverage |
| `findings` | Category block/window and document findings |
| `question_findings` | Whole-document custom-question findings |
| `candidate_findings` | Findings keyed by candidate ID |
| `coverage` | Source count, window count, evaluation count, context completeness |
| `requires_review`, `warnings` | Overall review recommendation and extraction/scope/context warnings |
| `provenance` | Input/registry/calibration hashes, runtime settings, threshold, interpretation |

A finding includes `task_id`, `block_refs`, `scope`, `outcome`, `probabilities`, `confidence`, `calibrated`, `requires_review`, and `reasons`. Evidence references identify the evaluated source blocks; they are not a learned causal explanation or extracted legal rationale. Multiple windows from one block currently share its block reference.

On overflow, probabilities are uniform placeholders while `confidence=0` and `outcome="insufficient_evidence"` explicitly signal unresolved evaluation. For custom questions that do not define an insufficient-evidence alternative, the overflow outcome is still this reserved status rather than a fabricated choice.

## Review behavior

Findings require review if calibration is missing/mismatched for the task, confidence is below threshold, the outcome is insufficient evidence, context is unresolved, or the checkpoint/calibration used unreviewed labels. The overall result additionally considers extraction warnings, narrowed category scope, and absent provenance metadata.

Providing a `provenance` object suppresses the missing-field warning; it does **not** authenticate its contents. The host must validate provenance and apply any marking or release workflow. Even `requires_review=False` is a model workflow recommendation, not formal CUI designation or authorization.

`result.filter_categories(["controlled_technical_information"])` selects already-computed category findings and rejects categories not evaluated. It does not rerun the model or rewrite coverage.

## Calibration

A calibration artifact must match checkpoint weights, registry hash, and recorded environment. A task must match its exact definition signature; adding a new informational query can make the score uncalibrated. State/content changes do not alter the task signature, so check deployment distribution and provenance quality separately.

The SDK supports deterministic kernels through `deterministic=True`, which changes process-wide PyTorch settings. Require the same settings when fitting calibration. CPU repeatability is tested; equality across hardware/backends is not claimed.
