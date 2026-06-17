"""Data plumbing for the character-level GPT: tokenization, the train/val
split, and reproducible mini-batch sampling.

A language model never sees text. It sees integers. Everything in this module
exists to convert a raw Unicode corpus into the exact tensor shapes the model
consumes — a context block `x` of shape `(B, T)` and a target block `y` of the
same shape, where `y` is `x` shifted one position to the left. The single most
important invariant in autoregressive training lives here: target token `y[t]`
is the character that *actually followed* context `x[:t+1]` in the corpus, so a
prediction at position `t` is graded against ground truth the model was not
allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


# Resolve the bundled corpus relative to this file so imports work regardless of
# the caller's working directory (notebooks run from notebooks/, tests from /tmp).
_DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "input.txt"


class CharTokenizer:
    """Character-level tokenizer: a bijection between the characters present in a
    corpus and the integers ``0 .. vocab_size-1``.

    Mathematical operation
    -----------------------
    ``encode``  maps a string ``s`` to ``[stoi[c] for c in s]``.
    ``decode``  maps an integer list ``l`` to ``''.join(itos[i] for i in l)``.
    The two are exact inverses on any string drawn from the fitted alphabet.

    Why character-level
    -------------------
    The vocabulary is tiny — 65 symbols for Tiny Shakespeare — which makes the
    embedding table and the final ``lm_head`` projection cheap (both scale with
    ``vocab_size``). The cost is paid on the sequence-length axis: a character
    model needs a longer context window to span the same number of *words* a
    subword model would, and it must spend capacity learning orthography
    (spelling, capitalization) that a BPE tokenizer gets for free. This is the
    deliberate trade the teaching model makes — a transparent vocabulary at the
    price of a harder modeling task per token.

    Why fit the alphabet from the corpus rather than fixing it
    ----------------------------------------------------------
    ``sorted(set(text))`` guarantees a stable, deterministic ordering of symbols
    every time the tokenizer is built from the same corpus, so a checkpoint's
    embedding rows stay aligned with the same characters across machines. A
    hard-coded ASCII table would waste rows on symbols the corpus never uses and
    would silently break on any character it omits.
    """

    def __init__(self, text: str):
        # Sorting makes the integer assignment deterministic and reproducible.
        chars = sorted(list(set(text)))
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        """String -> list of token ids. Raises ``KeyError`` on out-of-alphabet
        characters, which is the desired loud failure rather than a silent
        substitution that would corrupt the data stream."""
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> string. The inverse of :meth:`encode`."""
        return "".join(self.itos[i] for i in ids)


def load_text(path: str | Path | None = None) -> str:
    """Read the corpus file as UTF-8 and return it as one string.

    The default path resolves to the repository's ``data/input.txt`` (Tiny
    Shakespeare, ~1.1M characters). UTF-8 is specified explicitly so that the
    fitted alphabet is byte-for-byte identical on every platform — relying on the
    OS default encoding is the classic source of a checkpoint that loads but
    produces garbage because its vocabulary silently shifted.
    """
    path = Path(path) if path is not None else _DEFAULT_CORPUS
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class Dataset:
    """A tokenized corpus split into contiguous train/val tensors plus the
    tokenizer that produced them.

    The split is **chronological, not shuffled**. For images you would shuffle
    before splitting; for a language corpus you must not. Shuffling characters
    would shatter every word and sentence, destroying exactly the sequential
    structure the model exists to learn. Holding out the *last* 10% as a
    contiguous block gives a validation set of text the model never trained on,
    which is what makes val loss a real generalization signal rather than a
    memorization echo.
    """

    train_data: torch.Tensor   # (n_train,) int64 token ids
    val_data: torch.Tensor     # (n_val,)   int64 token ids
    tokenizer: CharTokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def make_dataset(text: str | None = None, train_frac: float = 0.9) -> Dataset:
    """Tokenize ``text`` and split it into train/val tensors.

    Implementation notes
    --------------------
    * ``dtype=torch.long`` (int64) is required, not merely convenient.
      ``nn.Embedding`` and all PyTorch advanced-indexing kernels expect 64-bit
      integer indices; an int8/int32 id tensor raises at the embedding lookup.
    * ``train_frac=0.9`` matches the source. With ~1.1M characters the model sees
      ~1.0M training tokens and ~0.1M held out — enough validation text that the
      loss estimate is stable across eval batches.
    """
    if text is None:
        text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return Dataset(train_data=data[:n], val_data=data[n:], tokenizer=tok)


def get_batch(
    dataset: Dataset,
    split: str,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one mini-batch of ``(context, target)`` pairs.

    Mathematical operation
    -----------------------
    Draw ``batch_size`` random start indices ``i``; for each, take
    ``x = data[i : i+block_size]`` and ``y = data[i+1 : i+block_size+1]``. The
    ``+1`` shift is the whole game: a single length-``T`` window is not one
    training example but ``T`` of them, because position ``t`` predicts ``y[t]``
    using only ``x[:t+1]``. One forward pass therefore produces ``B*T``
    supervised next-token predictions — the source of the transformer's
    training efficiency.

    Shape walk
    ----------
        ix              : (B,)            random start offsets
        x = stack(...)  : (B, T)          contexts
        y = stack(...)  : (B, T)          targets, x shifted left by one

    Why ``len(data) - block_size`` as the sampling ceiling
    ------------------------------------------------------
    A start index ``i`` needs ``i + block_size + 1`` valid positions to form a
    full ``(x, y)`` pair. Capping ``randint`` at ``len(data) - block_size``
    guarantees the target slice never runs off the end of the tensor.

    Why an explicit ``generator``
    -----------------------------
    The source draws batches from the *global* RNG, which the evaluation loop
    also consumes — so the training trajectory silently depends on how often you
    evaluate. Passing a dedicated :class:`torch.Generator` decouples data
    sampling from every other random op, making a run reproducible regardless of
    eval cadence, dropout, or sampling temperature elsewhere in the program.
    """
    data = dataset.train_data if split == "train" else dataset.val_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + block_size] for i in ix])            # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])    # (B, T)
    return x.to(device), y.to(device)
