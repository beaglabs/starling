"""Atomic safetensors checkpoints with model/tokenizer/registry identities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from .datasets import write_json
from .model import DecisionEncoder, ModelConfig
from .registry import digest
from .schemas import Registry


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, model, tokenizer, registry, metadata):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".starlings-", dir=destination.parent))
    try:
        save_file(
            {
                key: value.detach().cpu().contiguous().clone()
                for key, value in model.state_dict().items()
            },
            str(temporary / "model.safetensors"),
        )
        tokenizer.save(str(temporary / "tokenizer.json"))
        write_json(temporary / "config.json", model.config.to_dict())
        write_json(temporary / "registry.json", registry.model_dump())
        manifest = {
            **metadata,
            "format_version": 1,
            "registry_hash": digest(registry.model_dump()),
            "files": {
                name: file_hash(temporary / name)
                for name in ["model.safetensors", "tokenizer.json", "config.json", "registry.json"]
            },
        }
        manifest["model_id"] = manifest["files"]["model.safetensors"]
        write_json(temporary / "manifest.json", manifest)
        # Output checkpoints are immutable; the caller must select a fresh directory.
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    expected_files = {"model.safetensors", "tokenizer.json", "config.json", "registry.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("Invalid checkpoint file manifest")
    for name in expected_files:
        if file_hash(path / name) != manifest["files"][name]:
            raise ValueError(f"Checkpoint integrity check failed: {name}")
    registry = Registry.model_validate_json((path / "registry.json").read_text())
    if digest(registry.model_dump()) != manifest["registry_hash"]:
        raise ValueError("Registry hash mismatch")
    if manifest.get("model_id") != manifest["files"]["model.safetensors"]:
        raise ValueError("Model identity mismatch")
    config = ModelConfig(**json.loads((path / "config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    if (
        tokenizer.get_vocab_size() != config.vocab_size
        or tokenizer.token_to_id("[PAD]") != config.pad_id
    ):
        raise ValueError("Tokenizer/model mismatch")
    if any(
        tokenizer.token_to_id(token) is None
        for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    ):
        raise ValueError("Tokenizer missing required special tokens")
    model = DecisionEncoder(config)
    model.load_state_dict(load_file(str(path / "model.safetensors"), device="cpu"), strict=True)
    model.to(device).eval()
    return model, tokenizer, registry, manifest
