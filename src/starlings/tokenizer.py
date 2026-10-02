"""Locally trained byte-level BPE. No model or tokenizer downloads."""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_tokenizer(texts: Iterable[str], path: str | Path, vocab_size: int = 16_000) -> Tokenizer:
    if vocab_size < 261:
        raise ValueError("Vocabulary must fit byte alphabet and special tokens (>=261)")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def encode(tokenizer: Tokenizer, text: str, max_length: int) -> list[int]:
    ids = tokenizer.encode(text).ids
    if len(ids) + 2 > max_length:
        raise ValueError(
            f"Input exceeds context: {len(ids) + 2} tokens > {max_length}; no silent truncation"
        )
    return [tokenizer.token_to_id("[CLS]")] + ids + [tokenizer.token_to_id("[SEP]")]
