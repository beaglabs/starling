# Datasets and review

Decision datasets are UTF-8 JSONL: one strict `DecisionRow` per line. Unknown schema fields are rejected. See [example row](../examples/decision-row.json).

| Field | Requirement |
| --- | --- |
| `id` | Unique row identifier within the dataset |
| `group_id` | Shared identifier for an entire document/scenario family, including variants |
| `content` | String or `{"blocks": [{"id": "p1-b1", "page": 1, "text": "..."}]}` |
| `state` | Finite JSON object containing caller-established context |
| `category` | Registry slug; mutually exclusive with `question` |
| `question` | Instruction and 2–32 described alternatives; mutually exclusive with `category` |
| `label` | Exact alternative ID; required for training, calibration and evaluation |
| `target` | Optional probability distribution covering every alternative and summing to one |
| `review` | Pending, approved, or rejected; approval requires reviewer and rationale |
| `source_kind` | `official_scenario`, `public`, `synthetic`, `paired`, or `operational` |

Category labels are `applicable`, `not_applicable`, or `insufficient_evidence`. A soft target changes the training objective; a hard `label` remains necessary for review records and held-out metrics.

## Four source types

**Official materials:** use definitions and authorities as criteria. When using an official training scenario, record its exact source and conditions. A category definition is not itself a labeled document.

**Public documents:** use permitted text for vocabulary and language pretraining. Review document-specific applicability rather than labeling every public technical paragraph by keyword. Public availability and release provenance are distinct facts to document.

**Synthetic scenarios:** write clearly fictional cases with specified provenance and missing-context variants. Keep `review.status="pending"` until a reviewer supplies a rationale. Generated labels are hypotheses, not ground truth.

**Paired variations:** change one factor at a time: source provenance, authorized public release, markings, authority applicability, a missing contract fact, or the addition/removal of contextual passages. Keep the same `group_id` across all versions and all category/question rows derived from the scenario.

## Approval records

An approved example has a record such as:

```json
{
  "status": "approved",
  "reviewer": "reviewer-record-id",
  "rationale": "Documented reasoning tied to the scenario and applicable criteria.",
  "authority_refs": ["https://www.archives.gov/cui/registry/category-list"]
}
```

These fields are auditable records supplied by your workflow. The CLI does not verify a reviewer's identity, legal qualifications, or authority. That belongs to the host review process. Rejected examples always fail training validation, including experiments.

`--allow-unreviewed` permits **labeled pending examples** for experiments; it does not fill missing labels or bypass rejected records. The experimental flag follows the checkpoint and calibration lineage, and SDK findings require review.

## All-category coverage

`data scaffold` creates one unlabeled pending row for every category in the selected registry. It is an inventory, not enough training data. Collect positives, negatives, and missing-evidence cases for each category, plus cross-category confusions and document aggregation cases. Review the counts and class distribution within every split.

The splitter shuffles whole groups with a fixed seed: roughly 70% training and 10% each validation/calibration/test, with at least one held-out group per split. It requires four independent groups and does not stratify categories. Sparse category families may be absent from a split; the manifest and test report surface coverage, but you must remedy gaps.

Do not train the tokenizer or language model on held-out scenario families. Exact-duplicate checks cannot detect paraphrases, shared source documents with different IDs, synthetic siblings assigned different groups, or hidden semantic duplication.

## Provenance and instructions

Keep source text, host-verified metadata, and LLM candidates separate. JSON state accepts arbitrary keys, so establish a stable host schema for fields such as source identity, release determination, contract facts, marking history, and authority references. Neither `"public_release": true` nor `"host_verified": true` is authenticated by Starlings.

Task prompts and JSON structure do not eliminate instruction injection or factual uncertainty. Include adversarial document instructions and conflicting provenance in reviewer-checked evaluation. Train the model to recognize missing context rather than forcing a binary answer.

Registry refresh fetches category metadata only. Public document acquisition, licensing, scenario review, OCR, and operational data access are intentionally supplied by the dataset author; there is no built-in harvesting of real CUI.
