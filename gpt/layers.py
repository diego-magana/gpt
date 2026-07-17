"""The transformer's modular primitives — attention head, multi-head attention,
position-wise feed-forward, and the residual block that stacks them — each
instrumented for the interpretability analysis.

The instrumentation is what Karpathy's original lacks and what
``notebooks/05_attention_analysis.ipynb`` runs on: attention capture, head
ablation, head-output patching, and residual-stream access. All of it is inert
unless the analysis switches it on — no hook changes the trained weights or the
forward computation when unused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ActivationCache:
    """The internal tensors one forward pass exposes.

    ``attn[layer]``: per-head attention weights. ``resid[k]``: the residual stream
    entering block ``k`` (``0`` = post-embedding, ``n_layer`` = final pre-``ln_f``).
    ``head_out[layer]``: each head's output *before* the output projection — what a
    head actually writes, and so the thing to splice in a head-level intervention.
    """

    attn: dict[int, torch.Tensor] = field(default_factory=dict)      # layer -> (B, n_head, T, T)
    resid: dict[int, torch.Tensor] = field(default_factory=dict)     # k -> (B, T, C)
    head_out: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> (B, n_head, T, head_size)


class Head(nn.Module):
    """One head of causal self-attention — a content-based router.

    ``A = softmax(q k^T / sqrt(head_size) + causal_mask)``, then ``out = A v``: each
    position emits a query, matches it against every key at or before it, and
    gathers a convex combination of values. This is the only mechanism in the model
    that moves information *between* positions.

    The ``1/sqrt(head_size)`` scaling (not ``1/sqrt(n_embd)``) is load-bearing — see
    the README's implementation notes. ``return_attn`` hands the pre-dropout ``(T, T)``
    weights to the analysis.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape                       # (B, T, n_embd)
        k = self.key(x)                         # (B, T, head_size)
        q = self.query(x)                       # (B, T, head_size)

        # Affinities, scaled by 1/sqrt(d_head). We scale by k.shape[-1], the head
        # dimension, *not* the embedding dimension C — the variance argument above
        # is about the dot product over head_size terms.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # Causal mask: a position may attend to itself and the past, never the
        # future. ``-inf`` pre-softmax becomes exactly 0 post-softmax.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                   # (B, T, T)
        attn = wei                                                    # save pre-dropout weights
        wei = self.dropout(wei)

        v = self.value(x)                       # (B, T, head_size)
        out = wei @ v                           # (B, T, head_size)
        if return_attn:
            return out, attn
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected by ``W_O``.

    Each head gets ``n_embd // n_head`` dimensions and its own ``(W_Q, W_K, W_V)``,
    so heads can specialize into different relations at the same parameter budget —
    which is what makes "which head does the work?" a question worth asking.

    Two intervention hooks. ``ablate`` zeroes a set of heads before concatenation
    (how much the model *depends* on a head). ``patch_heads`` replaces a head's
    output with a supplied tensor (typically its own output on a clean input) —
    splicing one head's clean output into a corrupted run isolates that head alone,
    which neither ablation nor residual patching can do.
    """

    def __init__(self, n_embd: int, n_head: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        ablate: set[int] | None = None,
        patch_heads: dict[int, torch.Tensor] | None = None,
    ):
        ablate = ablate or set()
        patch_heads = patch_heads or {}
        outs, attns = [], []
        for h_idx, head in enumerate(self.heads):
            if return_attn:
                o, a = head(x, return_attn=True)   # o: (B, T, head_size), a: (B, T, T)
                attns.append(a)
            else:
                o = head(x)                         # (B, T, head_size)
            if h_idx in patch_heads:
                o = patch_heads[h_idx]              # splice in a supplied (clean) head output
            elif h_idx in ablate:
                o = torch.zeros_like(o)             # knock this head out of the sum
            outs.append(o)

        head_outs = torch.stack(outs, dim=1)        # (B, n_head, T, head_size)
        out = torch.cat(outs, dim=-1)               # (B, T, n_head*head_size) == (B, T, n_embd)
        out = self.dropout(self.proj(out))          # (B, T, n_embd)
        if return_attn:
            stacked = torch.stack(attns, dim=1)     # (B, n_head, T, T)
            return out, stacked, head_outs
        return out


class FeedForward(nn.Module):
    """Position-wise MLP — the per-token "computation" half of a block.

    ``Dropout(max(0, x W_1 + b_1) W_2 + b_2)`` with a 4x inner expansion. Attention
    moves information between positions but barely transforms it; the FFN is where
    each token privately processes the context attention just gathered. Communicate,
    then compute — the defining rhythm of the block.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # (B, T, n_embd) -> (B, T, 4*n_embd)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # (B, T, 4*n_embd) -> (B, T, n_embd)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: ``x = x + MHA(LN(x))`` then ``x = x + FFN(LN(x))``.

    Pre-norm — LayerNorm inside each residual branch, skip path left clean — keeps
    an identity highway running the full depth so gradients reach the early layers.
    Attention and the FFN get separate LayerNorms since they see different input
    statistics. ``return_attn``, ``ablate``, and ``patch_heads`` pass through to the
    attention sub-layer for the analysis.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        ablate: set[int] | None = None,
        patch_heads: dict[int, torch.Tensor] | None = None,
    ):
        if return_attn:
            sa_out, attn, head_outs = self.sa(
                self.ln1(x), return_attn=True, ablate=ablate, patch_heads=patch_heads
            )
        else:
            sa_out = self.sa(self.ln1(x), ablate=ablate, patch_heads=patch_heads)
            attn = head_outs = None
        x = x + sa_out                       # (B, T, C) residual after attention
        x = x + self.ffwd(self.ln2(x))       # (B, T, C) residual after FFN
        if return_attn:
            return x, attn, head_outs
        return x
