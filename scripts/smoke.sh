#!/usr/bin/env bash
# A small CPU workflow proving the pipeline, not CUI detection quality.
set -euo pipefail
STARLINGS_SMOKE_DIR="${1:-runs/smoke}"
if [[ -e "$STARLINGS_SMOKE_DIR" ]]; then
  echo "Output already exists: $STARLINGS_SMOKE_DIR" >&2
  exit 2
fi
mkdir -p "$STARLINGS_SMOKE_DIR"
cmd=(starlings --threads 1 --deterministic)
"${cmd[@]}" data demo --output "$STARLINGS_SMOKE_DIR/demo"
"${cmd[@]}" data split --input "$STARLINGS_SMOKE_DIR/demo/rows.jsonl" --output "$STARLINGS_SMOKE_DIR/splits"
"${cmd[@]}" data corpus --input "$STARLINGS_SMOKE_DIR/splits/train.jsonl" --output "$STARLINGS_SMOKE_DIR/corpus.txt"
"${cmd[@]}" tokenizer train --corpus "$STARLINGS_SMOKE_DIR/corpus.txt" --vocab-size 512 --output "$STARLINGS_SMOKE_DIR/tokenizer.json"
"${cmd[@]}" init --tokenizer "$STARLINGS_SMOKE_DIR/tokenizer.json" --output "$STARLINGS_SMOKE_DIR/init" \
  --hidden-size 32 --layers 1 --heads 4 --intermediate-size 64 --max-length 512 --dropout 0
"${cmd[@]}" pretrain --checkpoint "$STARLINGS_SMOKE_DIR/init" --corpus "$STARLINGS_SMOKE_DIR/corpus.txt" \
  --output "$STARLINGS_SMOKE_DIR/pretrained" --device cpu --steps 2
"${cmd[@]}" train --checkpoint "$STARLINGS_SMOKE_DIR/pretrained" --train "$STARLINGS_SMOKE_DIR/splits/train.jsonl" \
  --validation "$STARLINGS_SMOKE_DIR/splits/validation.jsonl" --output "$STARLINGS_SMOKE_DIR/decision" \
  --device cpu --grad-accum 2 --max-steps 3 --allow-unreviewed
"${cmd[@]}" calibrate --checkpoint "$STARLINGS_SMOKE_DIR/decision" --data "$STARLINGS_SMOKE_DIR/splits/calibration.jsonl" \
  --output "$STARLINGS_SMOKE_DIR/calibration.json" --device cpu --allow-unreviewed
"${cmd[@]}" evaluate --checkpoint "$STARLINGS_SMOKE_DIR/decision" --data "$STARLINGS_SMOKE_DIR/splits/test.jsonl" \
  --calibration "$STARLINGS_SMOKE_DIR/calibration.json" --output "$STARLINGS_SMOKE_DIR/report.json" --device cpu --allow-unreviewed
"${cmd[@]}" predict --checkpoint "$STARLINGS_SMOKE_DIR/decision" --request examples/request.json \
  --calibration "$STARLINGS_SMOKE_DIR/calibration.json" --output "$STARLINGS_SMOKE_DIR/result.json" --device cpu
