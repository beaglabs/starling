# Training and evaluation

Follow the executable command sequence in the [README](../README.md#train-your-model). The CLI and Python functions use the same packing and checkpoint format.

## Lifecycle

1. Define independent document/scenario families with `group_id` and create reviewed decision rows.
2. Split into training, model-selection validation, calibration, and test before preparing your corpus.
3. Build a training-only text corpus. Public definitions are metadata, not positive CUI examples.
4. Train BPE locally. Initialize an encoder with random weights and a pinned registry.
5. Pretrain on permitted text with masked language modeling; then train the decision head and encoder.
6. Fit a temperature using the calibration split. It changes confidence, not the highest-scoring option.
7. Evaluate once against independent test families. Inspect errors and coverage per category before using scores operationally.

No pretrained assets are downloaded. The shipped model is a plain bidirectional PyTorch Transformer with learned positional embeddings and a shared alternative scorer. Pretraining ties its vocabulary projection to token embeddings and computes logits only at masked positions.

## Trainer behavior

| Setting | Behavior |
| --- | --- |
| Precision | FP32 |
| Optimizer | AdamW, fixed learning rate, gradient norm clipping at 1.0 |
| Language objective | 15% masked tokens, 80/10/10 mask/random/unchanged corruption |
| Decision objective | Cross entropy over alternatives; soft target distributions are supported |
| Memory | One row's alternatives form a microbatch; `--grad-accum` accumulates rows |
| Validation | Final validation loss; no automatic best-checkpoint selection or early stopping |
| Continuation | Pass a prior decision checkpoint to `train`; optimizer state starts fresh |
| Encoder freezing | `--freeze-encoder` updates only the decision scorer |
| Ordering | Alternative IDs are sorted; each is scored independently using its description |
| Long training examples | Reject before compute; no silent truncation |
| Dataset size | Reference preparation/pretraining loads text into memory; not a streaming distributed trainer |

The small default runs are software examples. Training meaningful semantic behavior from random weights requires adequate language data, reviewed decision coverage, and independent evaluation. Starting small on M2 is feasible as an experiment; training time and accuracy remain empirical questions.

## Devices and repeatability

`auto` tries CUDA, MPS, then CPU. Explicit backends fail if unavailable. No automatic CPU fallback masks an MPS error. PyTorch owns backend support; kernel availability depends on the installed version and hardware.

`--seed` initializes Python, NumPy, and PyTorch seeds. `starlings --deterministic ...` requires deterministic algorithms and disables PyTorch's MHA fast path. These settings are process-wide in Python, as are CPU thread settings. Use a dedicated inference process when other workloads need different settings.

Calibration records Python/PyTorch versions, platform, device, CPU thread count, deterministic mode, MHA fast-path mode, and CUDA matrix TF32 setting. SDK and test evaluation enforce matching recorded settings. This checks known settings; it is not an exhaustive hardware fingerprint or a promise of reproducibility across devices.

## Checkpoints

Each immutable output directory contains:

| File | Contents |
| --- | --- |
| `model.safetensors` | FP32 model weights; no pickle model deserialization |
| `tokenizer.json` | Locally trained BPE |
| `config.json` | Encoder dimensions and context length |
| `registry.json` | Exact category definitions and authorities used by the checkpoint |
| `manifest.json` | File hashes, model/registry identities, stage, training lineage and metrics |

Writes use a temporary sibling directory and rename on success. Loads verify hashes, registry identity, tokenizer compatibility, and weight shapes. Integrity checks detect accidental changes; they do not authenticate an untrusted manifest. Accept checkpoint and calibration artifacts from trusted sources.

There is no full optimizer-state resume, distributed trainer, AMP, scheduler, ONNX export, or inference quantization in this initial implementation. These are future extensions, not hidden capabilities.

## Calibration and test reports

Calibration searches one positive scalar temperature on a deterministic logarithmic grid using NLL. A global temperature is fitted across the supplied tasks; its per-task sample size and suitability still need assessment. A task is matched by a hash of its exact instruction and option descriptions. Unseen or edited tasks retain raw scores and require review.

Training rejects overlap between current train/validation groups and exact examples. Continued training also checks prior training versus new validation and prior validation versus new training. Calibration/test reject training and model-selection overlap. Test evaluation additionally rejects calibration overlap. Related but textually different documents require correct group IDs from the dataset author.

Reports expose overall accuracy, NLL, multiclass Brier score, ten-bin ECE, score-threshold coverage/accuracy, confusion, and per-category counts. `false_negative_rate` specifically counts gold `applicable` predicted `not_applicable`; abstentions are not included in that count. Inspect the confusion matrix and applicable recall alongside it. Missing categories are explicitly listed. No metric grants designation or release authority.

The SDK also performs independent block and whole-document passes. A calibration corpus of short scenarios does not establish calibration for long documents, OCR corruption, windowed text, or new provenance patterns. Evaluate those deployment conditions separately.
