"""Validate an LLM's hypotheses against locally extracted source evidence.

Usage: python examples/llm-handoff.py --checkpoint runs/decision \
    --content runs/content.json --candidates runs/candidates.json --output runs/result.json

The host supplies provenance. The LLM supplies only candidate hypotheses/references.
No inference providers are called by this example or by the SDK.
"""

import argparse
import json
from pathlib import Path

from starlings import Classifier, Content

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--content", required=True)
parser.add_argument("--candidates", required=True)
parser.add_argument("--state", help="Host-established metadata JSON")
parser.add_argument("--calibration")
parser.add_argument("--output", required=True)
args = parser.parse_args()

content = Content.model_validate_json(Path(args.content).read_text())
candidates = json.loads(Path(args.candidates).read_text())
state = json.loads(Path(args.state).read_text()) if args.state else {}
classifier = Classifier(args.checkpoint, calibration=args.calibration)
result = classifier.evaluate(content=content, state=state, categories="all", candidates=candidates)
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(result.model_dump_json(indent=2) + "\n")
print(f"Saved {output}; requires_review={result.requires_review}")
