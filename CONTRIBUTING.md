# Contributing

Install Python 3.11+ and the development extras, then run `ruff check .`, `ruff format --check .`, `pytest -q`, and `python -m build` before opening a pull request. CPU-only Linux users should install CPU PyTorch first, as described in the README. Run `bash scripts/smoke.sh` to exercise the public CLI end to end.

Keep training data, extracted documents, weights, and experiment outputs outside source control. The ignored `datasets/` and `runs/` directories are intended for local work. Tests use public metadata and fictional fixtures.

Changes to input packing, category instructions, tokenization, or model behavior can invalidate trained artifacts or calibration. Update the model card and schemas as needed; do not silently reuse old confidence claims. Tests should exercise observable behavior and failure modes.

Registry updates must be explicit, complete, and attributed. Refresh into a fresh local path, inspect the diff and source citations, then replace the bundled snapshot deliberately. Network access must stay out of training and inference.

Dataset contributions need a documented source, usage rights, reviewer identity, label rationale, and paired-scenario group membership. Synthetic drafts must remain pending until a reviewer checks them. Never describe the routing smoke test as evidence of CUI classification quality.
