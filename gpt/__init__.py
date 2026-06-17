"""``gpt`` — a modular, instrumented re-implementation of Karpathy's
"Let's build GPT" capstone, factored for an interpretability analysis.

Public surface
--------------
Data:     ``CharTokenizer``, ``Dataset``, ``make_dataset``, ``load_text``, ``get_batch``
Layers:   ``Head``, ``MultiHeadAttention``, ``FeedForward``, ``Block``, ``ActivationCache``
Runtime:  ``train``, ``estimate_loss``, ``generate``, ``set_seed``, ``TrainConfig``

The model classes live in the top-level ``models`` package (``models.gpt.GPT``,
``models.bigram.BigramLanguageModel``) so that "the reusable machinery" and "the
architectures built from it" stay cleanly separated.
"""

from .data import (
    CharTokenizer,
    Dataset,
    get_batch,
    load_text,
    make_dataset,
)
from .layers import (
    ActivationCache,
    Block,
    FeedForward,
    Head,
    MultiHeadAttention,
)
from .train import (
    TrainConfig,
    estimate_loss,
    generate,
    set_seed,
    train,
)

__all__ = [
    "CharTokenizer",
    "Dataset",
    "make_dataset",
    "load_text",
    "get_batch",
    "ActivationCache",
    "Head",
    "MultiHeadAttention",
    "FeedForward",
    "Block",
    "train",
    "estimate_loss",
    "generate",
    "set_seed",
    "TrainConfig",
]
