"""One CLI for data, random initialization, training, evaluation and local inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .datasets import read_rows, split_rows, validate_rows, write_json
from .registry import load_registry


def parser():
    root = argparse.ArgumentParser(
        prog="starlings", description="Train and run local decision models from scratch."
    )
    root.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic PyTorch kernels; unsupported operations fail",
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--threads", type=int, help="Explicit PyTorch CPU thread budget")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect available CPU/MPS/CUDA devices")
    registry = commands.add_parser(
        "registry", help="Inspect or explicitly refresh the category snapshot"
    ).add_subparsers(dest="action", required=True)
    listing = registry.add_parser("list")
    listing.add_argument("--registry")
    refresh = registry.add_parser("refresh")
    refresh.add_argument("--output", required=True)
    refresh.add_argument("--workers", type=int, default=4)
    data = commands.add_parser(
        "data", help="Prepare reviewed JSONL and leak-resistant group splits"
    ).add_subparsers(dest="action", required=True)
    for name in ["scaffold", "validate", "split", "demo", "corpus", "cui-demo"]:
        sub = data.add_parser(name)
        if name != "demo":
            sub.add_argument("--registry")
        if name in ["validate", "split"]:
            sub.add_argument("--input", required=True)
        if name != "validate":
            sub.add_argument("--output", required=True)
        if name == "cui-demo":
            sub.add_argument("--count", type=int, default=4000)
            sub.add_argument("--seed", type=int, default=42)
        if name == "split":
            sub.add_argument("--seed", type=int, default=42)
        if name == "validate":
            sub.add_argument("--allow-unreviewed", action="store_true")
        if name == "corpus":
            sub.add_argument(
                "--input",
                action="append",
                default=[],
                help="Repeat for text, extracted JSON, or training JSONL",
            )
            sub.add_argument("--include-registry", action="store_true")
    tokenizer = (
        commands.add_parser("tokenizer", help="Train byte-level BPE locally")
        .add_subparsers(dest="action", required=True)
        .add_parser("train")
    )
    tokenizer.add_argument("--corpus", required=True)
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=16000)
    init = commands.add_parser(
        "init", help="Initialize random encoder weights; no pretrained downloads"
    )
    init.add_argument("--tokenizer", required=True)
    init.add_argument("--registry")
    init.add_argument("--output", required=True)
    init.add_argument("--seed", type=int, default=42)
    for flag, default in [
        ("hidden-size", 512),
        ("layers", 8),
        ("heads", 8),
        ("intermediate-size", 2048),
        ("max-length", 1024),
    ]:
        init.add_argument(f"--{flag}", type=int, default=default)
    init.add_argument("--dropout", type=float, default=0.1)
    for name in ["pretrain", "train", "calibrate", "evaluate", "predict"]:
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        if name in ["train", "calibrate", "evaluate"]:
            sub.add_argument(
                "--allow-unreviewed",
                action="store_true",
                help="Experiments only; marks checkpoint/results accordingly",
            )
        if name in ["pretrain", "train"]:
            sub.add_argument("--lr", type=float, default=3e-4)
            sub.add_argument("--seed", type=int, default=42)
        if name == "pretrain":
            sub.add_argument("--corpus", required=True)
            sub.add_argument("--steps", type=int, default=100)
            sub.add_argument("--batch-size", type=int, default=2)
        if name == "train":
            sub.add_argument("--train", required=True)
            sub.add_argument("--validation", required=True)
            sub.add_argument("--epochs", type=int, default=2)
            sub.add_argument("--grad-accum", type=int, default=8)
            sub.add_argument("--max-steps", type=int)
            sub.add_argument("--freeze-encoder", action="store_true")
        if name in ["calibrate", "evaluate"]:
            sub.add_argument("--data", required=True)
        if name in ["evaluate", "predict"]:
            sub.add_argument("--calibration")
            sub.add_argument("--threshold", type=float, default=0.95)
        if name == "predict":
            sub.add_argument(
                "--request",
                required=True,
                help="JSON with content, state, categories, questions and optional candidates",
            )
            sub.add_argument("--deterministic", action="store_true", default=argparse.SUPPRESS)
            sub.add_argument("--max-evaluations", type=int, default=4096)
    encoder = commands.add_parser(
        "train-encoder",
        help="Train and benchmark a random encoder on the 4000 synthetic CUI examples",
    )
    encoder.add_argument("--output", required=True)
    encoder.add_argument(
        "--dataset",
        help="Directory containing four split JSONL files and registry.json; defaults to bundled CUI demo",
    )
    encoder.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    encoder.add_argument("--seed", type=int, default=42)
    for flag, default in [
        ("vocab-size", 8000),
        ("hidden-size", 128),
        ("layers", 2),
        ("heads", 4),
        ("epochs", 1),
        ("pretrain-steps", 50),
        ("grad-accum", 4),
    ]:
        encoder.add_argument(f"--{flag}", type=int, default=default)
    encoder.add_argument("--max-steps", type=int)
    encoder.add_argument("--lr", type=float, default=3e-4)
    pdf = (
        commands.add_parser("pdf", help="Extract stable source blocks from a PDF text layer")
        .add_subparsers(dest="action", required=True)
        .add_parser("extract")
    )
    pdf.add_argument("--input", required=True)
    pdf.add_argument("--output", required=True)
    return root


def run(args):
    from .preparation import fresh

    if args.command == "doctor":
        import torch

        from .model import select_device
        from .training import environment

        return {
            **environment(select_device()),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(),
            "note": "Device detection is not a memory or training-performance benchmark.",
        }
    if args.command == "registry":
        if args.action == "refresh":
            from .registry_refresh import refresh

            fresh(args.output)
            return refresh(args.output, args.workers)
        registry = load_registry(args.registry)
        return {
            "version": registry.version,
            "categories": [c.model_dump() for c in registry.categories],
        }
    if args.command == "train-encoder":
        from .encoder_demo import train_encoder

        return train_encoder(
            args.output,
            dataset=args.dataset,
            device=args.device,
            seed=args.seed,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            layers=args.layers,
            heads=args.heads,
            epochs=args.epochs,
            pretrain_steps=args.pretrain_steps,
            max_steps=args.max_steps,
            grad_accum=args.grad_accum,
            lr=args.lr,
        )
    if args.command == "data":
        from .preparation import make_demo, prepare_corpus, scaffold

        if args.action == "cui-demo":
            from .synthetic import export_dataset

            manifest = export_dataset(
                args.output, count=args.count, seed=args.seed, registry_path=args.registry
            )
            return {
                "output": args.output,
                "rows": manifest["rows"],
                "categories": len(manifest["splits"]["train"]["categories"]),
                "splits": {name: value["rows"] for name, value in manifest["splits"].items()},
                "manifest": str(Path(args.output) / "manifest.json"),
                "review_status": "pending",
            }
        if args.action == "demo":
            return make_demo(args.output)
        if args.action == "scaffold":
            return scaffold(args.output, args.registry)
        if args.action == "corpus":
            return prepare_corpus(args.output, args.input, args.registry, args.include_registry)
        rows = read_rows(args.input)
        if args.action == "validate":
            validate_rows(rows, load_registry(args.registry), args.allow_unreviewed)
            return {
                "rows": len(rows),
                "groups": len({r.group_id for r in rows}),
                "categories": sorted({r.category for r in rows if r.category}),
                "approved": sum(r.review.status == "approved" for r in rows),
            }
        fresh(args.output)
        return split_rows(rows, args.output, args.seed)
    if args.command == "tokenizer":
        from .tokenizer import train_tokenizer

        texts = Path(args.corpus).read_text().splitlines()
        if not any(text.strip() for text in texts):
            raise ValueError("Tokenizer corpus is empty")
        tokenizer = train_tokenizer(texts, fresh(args.output), args.vocab_size)
        return {"output": args.output, "vocab_size": tokenizer.get_vocab_size()}
    if args.command == "init":
        from .training import initialize

        config = {
            key: getattr(args, key)
            for key in [
                "hidden_size",
                "layers",
                "heads",
                "intermediate_size",
                "max_length",
                "dropout",
            ]
        }
        return initialize(args.tokenizer, args.output, config, args.registry, args.seed)
    if args.command == "pretrain":
        from .training import pretrain

        return pretrain(
            args.checkpoint,
            args.corpus,
            args.output,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
    if args.command == "train":
        from .training import train

        return train(
            args.checkpoint,
            args.train,
            args.validation,
            args.output,
            epochs=args.epochs,
            lr=args.lr,
            grad_accum=args.grad_accum,
            device=args.device,
            seed=args.seed,
            max_steps=args.max_steps,
            allow_unreviewed=args.allow_unreviewed,
            freeze_encoder=args.freeze_encoder,
        )
    if args.command == "calibrate":
        from .evaluation import calibrate

        return calibrate(
            args.checkpoint,
            args.data,
            args.output,
            device=args.device,
            allow_unreviewed=args.allow_unreviewed,
        )
    if args.command == "evaluate":
        from .evaluation import evaluate

        return evaluate(
            args.checkpoint,
            args.data,
            args.output,
            calibration=args.calibration,
            device=args.device,
            threshold=args.threshold,
            allow_unreviewed=args.allow_unreviewed,
        )
    if args.command == "predict":
        from .sdk import Classifier

        fresh(args.output)
        classifier = Classifier(
            args.checkpoint,
            device=args.device,
            threshold=args.threshold,
            calibration=args.calibration,
            max_evaluations=args.max_evaluations,
            deterministic=args.deterministic,
        )
        request = json.loads(Path(args.request).read_text())
        allowed = {"content", "state", "categories", "questions", "candidates"}
        if not isinstance(request, dict) or set(request) - allowed or "content" not in request:
            raise ValueError("Request requires content and only documented fields")
        result = classifier.evaluate(**request)
        write_json(args.output, result.model_dump())
        return {
            "output": args.output,
            "requires_review": result.requires_review,
            "coverage": result.coverage,
        }
    if args.command == "pdf":
        from .preparation import extract_pdf

        return extract_pdf(args.input, args.output)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.threads is not None:
            if args.threads <= 0:
                raise ValueError("Thread budget must be positive")
            import torch

            torch.set_num_threads(args.threads)
        if args.deterministic:
            import torch

            torch.use_deterministic_algorithms(True)
            torch.backends.mha.set_fastpath_enabled(False)
        result = run(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"starlings: {exc}", file=sys.stderr)
        return 2
