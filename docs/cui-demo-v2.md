# CUI demo v2 adversarial benchmark

`cui-demo-v2` is a benchmark-only synthetic holdout designed to be harder than the original 4,000-example CUI demo. It is not a replacement training corpus and it does not contain train, validation, or calibration splits.

Generate it with:

```bash
starlings data cui-demo-v2 --output datasets/cui-demo-v2
```

The export contains:

- `benchmark.jsonl` — 756 adversarial examples
- `manifest.json` — generator identity, balance, limitations, and digest
- `registry.json` — the exact category registry used to build the benchmark

The benchmark contains six examples for each of the 126 CUI categories. Each category has exactly two `applicable`, two `not_applicable`, and two `insufficient_evidence` examples, for 252 examples per outcome overall.

## Adversarial profiles

The six profiles are intentionally different from the v1 training/test templates:

1. `controlled_bland` — controlled host facts with ordinary administrative prose, no explicit category-name shortcut, and a cross-category distractor.
2. `controlled_false_release_claim` — the document itself claims it was approved for public posting and contains an instruction to ignore host metadata, while caller-established controls remain in force.
3. `released_stale_banner` — stale `CUI` / `DO NOT DISTRIBUTE` text remains in an archived copy after the caller-established release/decontrol record says restrictions are no longer in force.
4. `public_topic_only` — a public agenda mentions the category at a high level but contains no category-covered case material.
5. `conflicting_release_chain` — release records explicitly disagree and the host cannot establish whether controls remain in force.
6. `unknown_origin_strong_marking` — the document strongly marks itself controlled, but provenance, government relationship, release status, and control status are not established.

State keys remain compatible with the v1 caller-facts contract, but many values use new paraphrases such as `cleared_for_public_distribution`, `no_release_authorization_located`, `records_disagree_on_release`, and `unknown_from_available_records`. This is deliberate: the benchmark tests whether the encoder learned the semantics of the state facts rather than memorizing a handful of exact enum strings.

## Evaluate the trained demo model

For a checkpoint produced by the three-epoch demo run:

```bash
starlings evaluate \
  --checkpoint runs/cui-demo-v4/encoder \
  --data datasets/cui-demo-v2/benchmark.jsonl \
  --output runs/cui-demo-v4/benchmark-v2.json \
  --calibration runs/cui-demo-v4/calibration.json \
  --device mps \
  --allow-unreviewed
```

Then inspect the high-level metrics:

```bash
jq '{
  accuracy,
  nll,
  brier,
  ece,
  false_negative_rate,
  false_positive_rate,
  score_threshold_coverage,
  score_threshold_accuracy,
  confusion_matrix
}' runs/cui-demo-v4/benchmark-v2.json
```

And the weakest categories:

```bash
jq '
  .per_category
  | to_entries
  | sort_by(.value.accuracy)
  | .[:20]
  | map({
      category: .key,
      accuracy: .value.accuracy,
      applicable_recall: .value.applicable_recall,
      abstention_rate: .value.abstention_rate,
      samples: .value.samples
    })
' runs/cui-demo-v4/benchmark-v2.json
```

## Interpretation

A strong result here is more meaningful than 100% on the original synthetic holdout because this set uses a distinct generator, distinct document wording, cross-category distractors, misleading source-document instructions, stale markings, and paraphrased state values. It still does **not** establish real-world CUI accuracy: the examples are synthetic, labels are unreviewed generator assumptions, and the same registry/category task contract is shared with training.

Do not train on `benchmark.jsonl` if you want to preserve its value as an external synthetic holdout. If the benchmark becomes part of training or tuning, generate a fresh seed or create another independent benchmark before reporting performance.
