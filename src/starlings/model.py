"""Random-initialized bidirectional encoder with MLM and independent option scoring."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_000
    hidden_size: int = 512
    layers: int = 8
    heads: int = 8
    intermediate_size: int = 2048
    max_length: int = 1024
    dropout: float = 0.1
    pad_id: int = 0
    # Legacy checkpoints omit this field and therefore retain CLS pooling. Fresh
    # initializations set option_mean explicitly so option identity cannot be diluted
    # across a long evidence sequence.
    decision_pooling: str = "cls"
    # Legacy checkpoints also omit structured state features and preserve the original scorer
    # shape. Fresh checkpoints set this to the deterministic state feature contract size.
    state_feature_dim: int = 0

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.hidden_size,
                self.layers,
                self.heads,
                self.intermediate_size,
                self.max_length,
            )
            <= 0
        ):
            raise ValueError("Model dimensions must be positive")
        if self.max_length < 8 or self.hidden_size < 2 or self.vocab_size < 6:
            raise ValueError("Context, hidden size or vocabulary is too small")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if not 0 <= self.dropout < 1 or not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("Invalid dropout/pad id")
        if self.decision_pooling not in {"cls", "option_mean"}:
            raise ValueError("decision_pooling must be 'cls' or 'option_mean'")
        if self.state_feature_dim < 0:
            raise ValueError("state_feature_dim must be nonnegative")

    def to_dict(self):
        return asdict(self)


class DecisionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_id
        )
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.heads,
            config.intermediate_size,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.hidden_size)
        if config.state_feature_dim:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.state_feature_dim, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size),
            )
            scorer_input = config.hidden_size * 2
        else:
            self.state_encoder = None
            scorer_input = config.hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.apply(self._initialize)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def representations(self, input_ids, attention_mask):
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds model context")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return self.norm(x)

    def forward(self, input_ids, attention_mask, decision_mask=None, state_features=None):
        hidden = self.representations(input_ids, attention_mask)
        if self.config.decision_pooling == "cls":
            pooled = hidden[:, 0]
        else:
            if decision_mask is None or decision_mask.shape != attention_mask.shape:
                raise ValueError("option_mean pooling requires a decision mask matching attention")
            selected = decision_mask.bool() & attention_mask.bool()
            counts = selected.sum(dim=1, keepdim=True)
            if (counts == 0).any():
                raise ValueError("Decision mask contains an empty option span")
            pooled = (hidden * selected.unsqueeze(-1)).sum(dim=1) / counts

        if self.state_encoder is not None:
            expected = (input_ids.shape[0], self.config.state_feature_dim)
            if state_features is None or tuple(state_features.shape) != expected:
                raise ValueError(
                    f"Structured-state model requires state features with shape {expected}"
                )
            state_hidden = self.state_encoder(state_features.to(dtype=pooled.dtype))
            pooled = torch.cat([pooled, state_hidden], dim=-1)
        return self.scorer(pooled).squeeze(-1)

    def masked_logits(self, input_ids, attention_mask, positions):
        hidden = self.representations(input_ids, attention_mask)[positions]
        return nn.functional.linear(hidden, self.token_embedding.weight, self.mlm_bias)


def select_device(requested="auto") -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; choose --device cpu explicitly")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
