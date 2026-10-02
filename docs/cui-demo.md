# The 4,000-example CUI demo

This is a concrete, packaged dataset, not an unlabeled scaffold. It is stored as `src/starlings/data/cui-demo-4000.jsonl.gz`, with its generation/split manifest alongside it. The standard installation includes both files. Export to readable JSONL using:

```bash
starlings data cui-demo --output datasets/cui-4000
```

Exports include `examples.jsonl` (all 4,000 rows), `train.jsonl` (2,488), `validation.jsonl` (504), `calibration.jsonl` (504), `test.jsonl` (504), `registry.json`, and `manifest.json`. Every split includes all 126 categories and all three outcomes for each category. Each category contributes 31–32 rows.

## Generation

Generator version `cui-synthetic-v1`, seed 42. Eight scenario families per category each provide four variations; 32 trailing training variants are omitted to reach exactly 4,000. Five families/category go to training and one each to validation, calibration and test. The seeded shuffle randomizes family assignment, source types, fictional agencies, dates, project names, identifiers, wording and row order.

Families are case files, email threads, form attachments, technical reports, meeting packets, database exports, contract deliveries, and inspection notes. Category-group-specific descriptions provide fictional contextual material. These are template-generated records, not 4,000 independently authored case studies.

All siblings stay together by `group_id`. The export verifies manifest digests. The tokenizer and language-pretraining corpus in `train-encoder` use only training examples plus the fixed public registry. Templates and category definitions are shared across splits; this benchmark does not test unseen template generalization.

## Label assumptions

| Variation | Synthetic label | Assumption |
| --- | --- | --- |
| Controlled record | `applicable` | Content falls within the named category and the synthetic host establishes relevant controls |
| Released record | `not_applicable` | Authorized decontrol explicitly removed all applicable restrictions; content is identical to the controlled sibling |
| Missing origin | `insufficient_evidence` | Origin, government relationship, release and control facts explicitly remain unknown |
| Routine topic mention | `not_applicable` | Routine logistics text mentions a category but contains no category-covered records |
| Embedded instructions | `applicable` | Instructions inside source content do not override the host facts |
| Conflicting assertions | `insufficient_evidence` | Release and control assertions are disputed |
| Joint context | `applicable` | A second block supplies the relevant category context; no real legal mosaic-effect claim is made |

The first three variations occur in every family. The fourth variation cycles through the remaining cases. Labels are generator assumptions, not reviewer determinations. Review records remain `pending`, with no invented reviewer identity. The rationale is kept in the review record, outside model-visible content/state.

## JSON facts and sources

All examples include a `state.provenance` object with a synthetic source ID, `https://example.invalid/fixtures/...` URI, source type, fictional host identity, assertion status, collection date and `fictional=true`. The original definitions and authority references are attributable to the pinned NARA registry.

`state.facts` includes source type and identity, government relationship, public release status, release-authorization status/record, handling access/basis/control status/decontrol record, and contract identity/relationship. Non-applicable contracts have `id=null`; unresolved facts use `unknown` or `null` explicitly. Unknown source origin does not prevent the fixture from retaining an internal capture ID.

These are **caller assertions treated as premises**, not facts independently authenticated by the encoder. Source and state are fully populated as a schema; this does not mean every fact is known. No expected label, generator profile or review rationale is passed to the model as state.

## Train locally

```bash
starlings train-encoder --output runs/cui-demo --device mps
```

The command creates random width-128, two-layer encoder weights, trains BPE, performs 50 masked-language pretraining steps, trains one decision epoch, calibrates on 504 examples, then evaluates the other 504 benchmark examples. Encoder weights are updated, not frozen. Flags include `--hidden-size`, `--layers`, `--heads`, `--vocab-size`, `--epochs`, `--pretrain-steps`, `--max-steps`, `--grad-accum`, `--lr`, and `--seed`.

Context length is chosen from the longest packed training/validation input, rounded up with a margin. Category prompts use a compact model-visible projection containing the category name, group, definition, marking and authority citations. Full authority metadata such as control type, banner marking, sanctions, references, source URLs and hashes remains in the registry for auditability and UI use instead of being serialized into every decision option. This keeps attention focused on decision-relevant policy text and avoids multiplying registry boilerplate across the three outcome sequences.

The bundled dataset is seed 42 regardless of the model initialization seed. Regenerate fixtures explicitly with `data cui-demo --seed 7 --output ...`, then pass that directory using `train-encoder --dataset ...`. Generation count is bounded by this eight-family design; `--count` defaults to exactly 4,000.

`run.json` gives the checkpoint path and benchmark summary. `benchmark.json` includes per-category applicable recall, false negatives, false positives, abstention rate, calibration coverage and missing-category coverage. Every artifact remains experimental because labels are unreviewed. Model predictions can be wrong even when generator metadata is consistent.

## What the benchmark means

The benchmark measures whether the small model reproduces these synthetic assumptions on distinct scenario-family instances. Category titles and factual JSON fields can be strong shortcuts. It does not prove independent legal reasoning, real-document applicability, long-document mosaic detection, OCR robustness, or production accuracy. It is a concrete starting point for training and demonstrating fact-sensitive behavior without waiting for a reviewer.
