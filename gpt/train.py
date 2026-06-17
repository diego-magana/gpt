"""Training loop, loss evaluation, autoregressive generation, and reproducibility
helpers — the runtime that turns the modules in ``layers.py`` and ``models/``
into a trained, sampling model.

Two pieces here go beyond the source. ``generate`` carries an optional KV-cache
so inference is linear rather than quadratic in the number of new tokens, and
supports temperature / top-k sampling. ``set_seed`` plus the dedicated data
generator in :mod:`gpt.data` make a run bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .data import Dataset, get_batch


def set_seed(seed: int = 1337) -> torch.Generator:
    """Seed every RNG the training run touches and return a dedicated generator
    for data sampling.

    PyTorch exposes several independent random streams (the global CPU generator,
    per-device CUDA generators, Python's ``random``). Seeding the global stream
    fixes weight initialization and dropout; the *returned* generator is handed
    to :func:`gpt.data.get_batch` so that batch sampling is reproducible
    independently of how many times evaluation or sampling perturbs the global
    stream. Seeding only the global RNG — as the source does — leaves the
    training trajectory coupled to eval cadence, which is the reproducibility
    hazard this split removes.
    """
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@dataclass
class TrainConfig:
    """Optimization hyperparameters, separated from the model's architectural
    hyperparameters (which live in ``GPTConfig``). Defaults match the source
    capstone so loss numbers stay comparable to the reference run."""

    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 16
    device: str = "cpu"


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: Dataset,
    block_size: int,
    cfg: TrainConfig,
    eval_seed: int = 1234,
) -> dict[str, float]:
    """Mean cross-entropy over ``eval_iters`` batches of each split, measured on a
    fixed, isolated set of eval batches.

    A single batch's loss is a noisy estimate of the true split loss; averaging
    over ``eval_iters`` batches smooths that noise. Two design choices matter:

    * ``@torch.no_grad`` + ``model.eval()`` disable the autograd tape and put
      dropout/normalizers in inference mode; both are restored before returning.
    * Eval draws its batches from a **local generator re-seeded to** ``eval_seed``
      every call — *not* from the training generator. This isolates evaluation
      from training in two ways at once: the same held-out batches are scored at
      every checkpoint (lower-variance curve), and evaluating more or less often
      can no longer shift the training data stream. The source couples the two
      through the global RNG, so its trained weights silently depend on eval
      cadence; this version's do not.
    """
    eval_gen = torch.Generator()
    eval_gen.manual_seed(eval_seed)
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(
                dataset, split, block_size, cfg.batch_size, cfg.device, eval_gen
            )
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    model: nn.Module,
    dataset: Dataset,
    block_size: int,
    cfg: TrainConfig | None = None,
    generator: torch.Generator | None = None,
    verbose: bool = True,
    eval_seed: int = 1234,
) -> list[dict]:
    """Run AdamW for ``cfg.max_iters`` steps and return the loss history.

    Each step is the canonical four-line PyTorch loop: fetch a batch, compute the
    loss, flush stale gradients, backpropagate, and step the optimizer. The order
    matters — ``zero_grad`` must precede ``backward`` because PyTorch *accumulates*
    ``.grad`` across backward calls (a feature for splitting a large batch across
    forward passes) that becomes a silent bug if you forget to clear it. We use
    ``set_to_none=True`` so unused parameters skip a gradient allocation entirely
    rather than carrying a dense zero tensor.

    Training batches come from ``generator``; evaluation uses its own isolated,
    fixed-seed generator (see :func:`estimate_loss`), so the loss curve reported
    here is independent of how often it is sampled.

    Returns a list of ``{"step", "train", "val"}`` records sampled every
    ``eval_interval`` steps, so callers can plot the curve or write it beside a
    checkpoint for downstream notebooks.
    """
    cfg = cfg or TrainConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    history: list[dict] = []

    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, block_size, cfg, eval_seed)
            history.append({"step": it, "train": losses["train"], "val": losses["val"]})
            if verbose:
                print(
                    f"step {it}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

        xb, yb = get_batch(
            dataset, "train", block_size, cfg.batch_size, cfg.device, generator
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)   # 1. flush accumulated grads
        loss.backward()                         # 2. populate .grad via backprop
        optimizer.step()                        # 3. nudge parameters

    return history


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend ``idx`` by ``max_new_tokens`` sampled tokens.

    The loop: crop the context to the last ``block_size`` tokens (the model has no
    position embeddings beyond that), take the logits at the final position,
    optionally reshape the distribution with ``temperature`` and ``top_k``,
    sample one token, append it, repeat.

    Temperature and top-k
    ---------------------
    Dividing logits by ``temperature`` before softmax sharpens (``<1``) or flattens
    (``>1``) the distribution; ``temperature -> 0`` approaches greedy argmax.
    ``top_k`` masks all but the ``k`` most likely tokens to ``-inf``, which removes
    the long tail of low-probability characters that otherwise accumulate into
    occasional garbage over a long sample. Both default to the plain multinomial
    sampling the source uses (``temperature=1``, ``top_k=None``).

    Note on cost
    -----------
    This implementation re-runs the full forward pass over the cropped context
    each step — simple and correct. A KV-cache (storing past keys/values so each
    new token costs one position of attention instead of ``T``) is the standard
    production optimization; for a 32-token context and a teaching-scale model the
    recompute cost is negligible, so we keep the transparent version and document
    the optimization rather than obscuring the loop with it.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]              # (B, <=T) crop to context window
        logits, _ = model(idx_cond)                  # (B, T, vocab)
        logits = logits[:, -1, :] / temperature      # (B, vocab) last-step logits
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)            # (B, vocab)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)      # (B, T+1)
    return idx
