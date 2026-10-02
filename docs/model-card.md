# Starlings model card

## What is released

This repository releases a **trainable architecture, data contracts, CLI, and Python SDK**. It ships a registry snapshot, not decision-trained weights or a validated operational CUI detector. Test fixtures and the smoke workflow use fictional support routing. No CUI performance benchmark is claimed.

## Architecture

A randomly initialized bidirectional Transformer uses learned token and positional embeddings, pre-norm encoder layers, GELU feed-forward blocks, and a final layer norm. A shared two-layer scalar scorer reads the first token representation for each independently packed alternative. Softmax compares these scores. The language-pretraining head shares token embedding weights.

Defaults: vocabulary target 16,000, hidden width 512, eight layers, eight attention heads, feed-forward width 2,048, context 1,024, dropout 0.1. Actual parameter count is recorded at initialization. The README's local starter uses width 256 and four layers. These dimensions are experimental choices, not evidence of task sufficiency.

Inputs explicitly separate source content, caller state, task instruction, and alternative description. Packing is identical during decision training and SDK inference. Alternative IDs are sorted, and each alternative uses the same scoring network.

## Intended use

Local decision support, category triage, scenario research, and evidence-linked review proposals. The SDK accepts optional candidates from a larger model while retaining an independent scan of supplied source blocks. Official source criteria and host metadata help train context-sensitive decisions, but the encoder cannot authenticate provenance or dynamically resolve legal applicability by itself.

## Limits

- Real all-category performance depends on representative, independently reviewed labels; metadata coverage is not accuracy coverage.
- Random-weight language training may require more data and compute than an M2 experiment can provide economically.
- A small encoder can learn contextual distinctions, but nuanced authority/provenance reasoning and rare categories are unproven here.
- Whole-document context is finite. Windowing preserves local scans but cannot guarantee long-range mosaic detection.
- A source block reference is an evaluated evidence location, not an explanation of which words caused a decision.
- Calibration does not guarantee correctness, detect all out-of-distribution cases, or make new task definitions trustworthy.
- Text extraction does not understand all PDF tables, images, OCR errors, or reading order.
- Caller state, reviewer records, checkpoint manifests and calibration files must be trusted/validated by the host.
- Deterministic inference improves repeatability under pinned conditions; it does not improve semantic correctness.
- Changes in registry metadata require a new explicit checkpoint workflow and renewed evaluation.

## Operational evaluation needed

Collect category-specific applicable, not-applicable, and insufficient-evidence examples. Hold out source families and paired variations. Measure applicable recall, false negatives, false positives, abstention/review burden, calibration, OCR effects, long-document aggregation, and sensitivity to provenance changes. Include contradictory metadata, content instructions, invented candidates, unseen questions, and distribution shifts.

Assess actual deployment hardware and latency. The reference implementation is not optimized to batch hundreds of categories simultaneously and has no quantized inference/export backend. MPS support is implemented; implementation verification occurred on CPU, not an M2.

The model outputs proposals. The host remains responsible for authoritative designation, marking, handling, and release decisions.
