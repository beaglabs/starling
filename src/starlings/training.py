"""From-scratch initialization, masked pretraining, supervised decision training."""

from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .datasets import (
    assert_disjoint,
    content_text,
    fingerprint,
    pack,
    read_rows,
    row_task,
    validate_rows,
)
from .model import DecisionEncoder, ModelConfig, seed_everything, select_device
from .registry import digest, load_registry
from .tokenizer import encode


def environment(device):
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def initialize(tokenizer_path, output, config, registry_path=None, seed=42):
    if Path(output).exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    seed_everything(seed)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = DecisionEncoder(
        ModelConfig(
            **{
                **config,
                "vocab_size": tokenizer.get_vocab_size(),
                "pad_id": tokenizer.token_to_id("[PAD]"),
            }
        )
    )
    save_checkpoint(
        output,
        model,
        tokenizer,
        load_registry(registry_path),
        {
            "stage": "initialized",
            "seed": seed,
            "random_initialization": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
    )
    return {"checkpoint": str(output), "parameters": sum(p.numel() for p in model.parameters())}


def tensor_batch(sequences, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, device=device)
        mask[i, : len(seq)] = True
    return ids, mask


def decision_logits(model, tokenizer, content, state, question, device):
    options = sorted(question.criteria)
    sequences = [
        encode(tokenizer, pack(content, state, question, option), model.config.max_length)
        for option in options
    ]
    ids, mask = tensor_batch(sequences, model.config.pad_id, device)
    return options, model(ids, mask)


def row_loss(model, tokenizer, row, registry, device):
    question = row_task(row, registry)
    options, logits = decision_logits(
        model, tokenizer, content_text(row.content), row.state, question, device
    )
    if row.target is not None:
        targets = torch.tensor([row.target[option] for option in options], device=device)
        return -(targets * F.log_softmax(logits, dim=-1)).sum()
    return F.cross_entropy(logits[None, :], torch.tensor([options.index(row.label)], device=device))


def check_training_args(epochs, lr, grad_accum, max_steps):
    if epochs <= 0 or not math.isfinite(lr) or lr <= 0 or grad_accum <= 0:
        raise ValueError("epochs, learning rate and gradient accumulation must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")


def train(
    checkpoint,
    train_path,
    validation_path,
    output,
    *,
    epochs=2,
    lr=3e-4,
    grad_accum=8,
    device="auto",
    seed=42,
    max_steps=None,
    allow_unreviewed=False,
    freeze_encoder=False,
):
    check_training_args(epochs, lr, grad_accum, max_steps)
    if Path(output).exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    seed_everything(seed)
    selected = select_device(device)
    model, tokenizer, registry, previous = load_checkpoint(checkpoint, selected)
    train_rows, val_rows = read_rows(train_path), read_rows(validation_path)
    validate_rows(train_rows, registry, allow_unreviewed)
    validate_rows(val_rows, registry, allow_unreviewed)
    assert_disjoint(train_rows, val_rows)
    if any(
        r.group_id in previous.get("training_groups", [])
        or fingerprint(r) in previous.get("training_examples", [])
        for r in val_rows
    ):
        raise ValueError("Validation overlaps prior checkpoint training data")
    if any(
        r.group_id in previous.get("validation_groups", [])
        or fingerprint(r) in previous.get("validation_examples", [])
        for r in train_rows
    ):
        raise ValueError("Training overlaps prior model-selection validation")
    # Validate every packed sequence before spending compute; never truncate training labels.
    for row in train_rows + val_rows:
        question = row_task(row, registry)
        for option in question.criteria:
            encode(
                tokenizer,
                pack(content_text(row.content), row.state, question, option),
                model.config.max_length,
            )
    if freeze_encoder:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("scorer."))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses, steps, started = [], 0, time.perf_counter()
    for epoch in range(epochs):
        model.train()
        shuffled = list(train_rows)
        random.Random(seed + epoch).shuffle(shuffled)
        for start in range(0, len(shuffled), grad_accum):
            group = shuffled[start : start + grad_accum]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for row in group:
                loss = row_loss(model, tokenizer, row, registry, selected)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / len(group)).backward()
                total += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            losses.append(total / len(group))
            if steps == 1 or steps % 25 == 0:
                print(
                    json.dumps({"epoch": epoch + 1, "step": steps, "loss": losses[-1]}), flush=True
                )
            if max_steps and steps >= max_steps:
                break
        if max_steps and steps >= max_steps:
            break
    model.eval()
    with torch.inference_mode():
        validation_loss = sum(
            row_loss(model, tokenizer, row, registry, selected).item() for row in val_rows
        ) / len(val_rows)
    inherited_groups = previous.get("training_groups", [])
    inherited_examples = previous.get("training_examples", [])
    save_checkpoint(
        output,
        model,
        tokenizer,
        registry,
        {
            **previous,
            "stage": "decision",
            "parent_model_id": previous["model_id"],
            "training_groups": sorted(set(inherited_groups) | {r.group_id for r in train_rows}),
            "training_examples": sorted(
                set(inherited_examples) | {fingerprint(r) for r in train_rows}
            ),
            "validation_groups": sorted(
                set(previous.get("validation_groups", [])) | {r.group_id for r in val_rows}
            ),
            "validation_examples": sorted(
                set(previous.get("validation_examples", [])) | {fingerprint(r) for r in val_rows}
            ),
            "allow_unreviewed": allow_unreviewed or previous.get("allow_unreviewed", False),
            "environment": environment(selected),
            "seed": seed,
            "steps": steps,
            "loss": losses[-1],
            "validation_loss": validation_loss,
            "duration_seconds": time.perf_counter() - started,
            "train_digest": digest([r.model_dump() for r in train_rows]),
        },
    )
    return {
        "checkpoint": str(output),
        "steps": steps,
        "loss": losses[-1],
        "validation_loss": validation_loss,
        "device": str(selected),
    }


def pretrain(
    checkpoint, corpus, output, *, steps=100, lr=3e-4, batch_size=2, device="auto", seed=42
):
    check_training_args(1, lr, batch_size, steps)
    if Path(output).exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    seed_everything(seed)
    selected = select_device(device)
    model, tokenizer, registry, previous = load_checkpoint(checkpoint, selected)
    if previous["stage"] == "decision":
        raise ValueError(
            "Pretrain before decision training; this would invalidate the decision head"
        )
    texts = [line for line in Path(corpus).read_text().splitlines() if line.strip()]
    if not texts:
        raise ValueError("Pretraining corpus is empty")
    # Streaming-scale corpus tooling can extend this bounded in-memory reference trainer.
    sequences = []
    for text in texts:
        token_ids = tokenizer.encode(text).ids
        for start in range(0, len(token_ids), model.config.max_length - 2):
            chunk = token_ids[start : start + model.config.max_length - 2]
            sequences.append(
                [tokenizer.token_to_id("[CLS]"), *chunk, tokenizer.token_to_id("[SEP]")]
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    model.train()
    tokens, loss_value, started = 0, 0.0, time.perf_counter()
    for step in range(steps):
        batch = [rng.choice(sequences) for _ in range(batch_size)]
        ids, mask = tensor_batch(batch, model.config.pad_id, selected)
        labels = ids.clone()
        eligible = mask.clone()
        for special in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]:
            eligible &= ids != tokenizer.token_to_id(special)
        positions = (torch.rand(ids.shape, device=selected) < 0.15) & eligible
        if not positions.any():
            first = eligible.nonzero()
            if len(first) == 0:
                raise ValueError("Corpus contains no trainable tokens")
            positions[first[0, 0], first[0, 1]] = True
        noise = torch.rand(ids.shape, device=selected)
        ids[positions & (noise < 0.8)] = tokenizer.token_to_id("[MASK]")
        replacements = torch.randint(5, model.config.vocab_size, ids.shape, device=selected)
        ids[positions & (noise >= 0.8) & (noise < 0.9)] = replacements[
            positions & (noise >= 0.8) & (noise < 0.9)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.masked_logits(ids, mask, positions), labels[positions])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite pretraining loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = loss.detach().item()
        tokens += mask.sum().item()
        if step == 0 or (step + 1) % 25 == 0:
            print(json.dumps({"step": step + 1, "mlm_loss": loss_value}), flush=True)
    duration = time.perf_counter() - started
    save_checkpoint(
        output,
        model,
        tokenizer,
        registry,
        {
            **previous,
            "stage": "pretrained",
            "parent_model_id": previous["model_id"],
            "pretrain_steps": steps,
            "pretrain_tokens": tokens,
            "corpus_sha256": digest(texts),
            "environment": environment(selected),
            "seed": seed,
            "mlm_loss": loss_value,
            "duration_seconds": duration,
        },
    )
    return {
        "checkpoint": str(output),
        "loss": loss_value,
        "tokens": tokens,
        "training_tokens_per_second": tokens / duration,
        "device": str(selected),
    }
