"""The transformer's modular primitives — a single attention head, multi-head
attention, the position-wise feed-forward network, and the residual block that
stacks them — each instrumented for the interpretability analysis.

The instrumentation is the part Karpathy's original does not have, and it is
what the analysis in ``notebooks/05_attention_analysis.ipynb`` runs on. Three
hooks are threaded through these modules:

* **attention capture** — a head can return its ``(T, T)`` attention matrix so
  the analysis can characterize what each head attends to;
* **head ablation** — multi-head attention can zero a chosen subset of heads to
  measure each head's causal contribution to the loss;
* **residual-stream access** — the block returns its post-attention and
  post-block activations so the analysis can patch and read the residual stream.

None of these change the trained weights or the forward computation when left
inactive; they are pure observation/intervention surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ActivationCache:
    """A write-once container for the internal tensors a forward pass exposes.

    Mechanistic-interpretability work treats a transformer as a set of named
    activations connected by a residual stream; the cache is the minimal version
    of that idea. ``attn[layer]`` holds the per-head attention weights of block
    ``layer``; ``resid[k]`` holds the residual stream *entering* block ``k``
    (with ``resid[0]`` the post-embedding stream and ``resid[n_layer]`` the final
    pre-``ln_f`` stream). The analysis reads these instead of re-deriving them.
    """

    attn: dict[int, torch.Tensor] = field(default_factory=dict)    # layer -> (B, n_head, T, T)
    resid: dict[int, torch.Tensor] = field(default_factory=dict)   # k -> (B, T, C)


class Head(nn.Module):
    """One head of causal self-attention — a single content-based router.

    Mathematical operation
    -----------------------
        k = x W_K,  q = x W_Q,  v = x W_V                       projections
        A = softmax( (q kᵀ) / sqrt(d_head)  +  causal_mask )    affinities
        out = A v                                               weighted gather

    Intuition (the database metaphor)
    ---------------------------------
    Every position emits a **query** ("what am I looking for?"), a **key** ("what
    do I offer?"), and a **value** ("what will I hand over if selected?"). The
    dot product ``q · k`` scores how well position *i*'s query matches position
    *j*'s key; softmax turns those scores into a convex combination; the output
    at *i* is that combination of the *values*. Attention is therefore a
    differentiable, content-addressable lookup — the one mechanism in the model
    where information moves *between* token positions.

    Why divide by ``sqrt(d_head)``
    ------------------------------
    With unit-variance ``q`` and ``k`` and head dimension ``d``, the raw dot
    product ``q · k`` has variance ``d``. Feeding variance-``d`` logits into
    softmax makes it peaky — for ``d=16`` the largest logit dominates and the
    distribution collapses toward one-hot, which zeros the gradient to every
    non-selected position and stalls learning at initialization. Scaling by
    ``1/sqrt(d)`` restores unit variance so softmax starts soft and stays
    trainable. This is the "scaled" in scaled dot-product attention, and omitting
    it is a silent training-killer rather than a loud crash.

    Why a registered buffer for ``tril``
    ------------------------------------
    The lower-triangular causal mask is constant, not learned, but it must follow
    the module across ``.to(device)`` and in/out of ``state_dict``. ``register_
    buffer`` gives exactly that — device-tracked, persisted, but excluded from
    ``parameters()`` so the optimizer never touches it.
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
    """Several attention heads in parallel, concatenated and projected.

    Mathematical operation
    -----------------------
        out = Dropout( Concat(head_1, ..., head_h) W_O )

    Why multiple heads instead of one wide head
    -------------------------------------------
    Each head gets ``head_size = n_embd // n_head`` dimensions and learns its own
    ``(W_Q, W_K, W_V)``, so heads can specialize into different relations — one
    might track the previous character, another the start-of-line position,
    another a longer-range dependency. A single head of the full width can only
    represent *one* attention pattern per position; splitting the width buys
    several patterns at the same parameter budget. The output projection ``W_O``
    then mixes the per-head results back into the model dimension so the residual
    stream stays width-``n_embd``.

    The head-ablation hook
    ----------------------
    ``ablate`` is a set of head indices whose outputs are zeroed before
    concatenation. Zeroing a head's contribution and re-measuring the loss is a
    genuine intervention — it reports how much the model's predictions *depend*
    on that head, which correlational attention maps cannot tell you.
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
    ):
        ablate = ablate or set()
        outs, attns = [], []
        for h_idx, head in enumerate(self.heads):
            if return_attn:
                o, a = head(x, return_attn=True)   # o: (B, T, head_size), a: (B, T, T)
                attns.append(a)
            else:
                o = head(x)                         # (B, T, head_size)
            if h_idx in ablate:
                o = torch.zeros_like(o)             # knock this head out of the sum
            outs.append(o)

        out = torch.cat(outs, dim=-1)               # (B, T, n_head*head_size) == (B, T, n_embd)
        out = self.dropout(self.proj(out))          # (B, T, n_embd)
        if return_attn:
            stacked = torch.stack(attns, dim=1)     # (B, n_head, T, T)
            return out, stacked
        return out


class FeedForward(nn.Module):
    """Position-wise MLP: the per-token "computation" half of a block.

    Mathematical operation
    -----------------------
        FFN(x) = Dropout( max(0, x W_1 + b_1) W_2 + b_2 ),
        with W_1: (n_embd, 4*n_embd) and W_2: (4*n_embd, n_embd).

    Why it follows attention
    ------------------------
    Attention *moves* information between positions but applies no nonlinearity to
    the gathered content beyond the value projection — on its own it is close to a
    weighted average. The feed-forward network is where each token, now holding
    context aggregated by attention, does private nonlinear processing on it.
    The two-step rhythm "communicate (attention) then compute (FFN)" is the
    defining motif of the transformer block.

    Why the 4x expansion
    --------------------
    Projecting up to ``4*n_embd`` before the ReLU gives the nonlinearity a
    higher-dimensional space to carve, then projecting back down keeps the
    residual-stream width fixed. The 4x ratio is the convention inherited from
    the original transformer; it is the cheapest knob that reliably adds
    capacity without changing the stream geometry.
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
    """One transformer block: pre-norm attention and FFN, each wrapped in a
    residual connection.

    Mathematical operation
    -----------------------
        x = x + MHA(LayerNorm(x))      communication, with skip
        x = x + FFN(LayerNorm(x))      computation, with skip

    Why pre-norm (LayerNorm *inside* the residual branch)
    -----------------------------------------------------
    Normalizing the *input* to each sub-layer, while leaving the skip path
    ``x +`` un-normalized, keeps a clean identity highway running the full depth
    of the network. Gradients flow back along that highway undistorted, which is
    what lets a 4-layer stack train at all. The source uses this pre-norm
    arrangement (LayerNorm before the sub-layer); the original 2017 transformer
    put LayerNorm *after* the residual add, which is markedly harder to optimize
    without learning-rate warmup.

    Why two separate LayerNorms
    ---------------------------
    Attention and the FFN see different input statistics, so each gets its own
    normalizer with its own learnable gain/bias rather than sharing one.

    Cache / ablation pass-through
    ----------------------------
    When ``return_attn`` is set the block returns the per-head attention from its
    attention sub-layer; ``ablate`` forwards a set of head indices to zero. These
    are inert during ordinary training and only activate from the analysis code.
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
    ):
        if return_attn:
            sa_out, attn = self.sa(self.ln1(x), return_attn=True, ablate=ablate)
        else:
            sa_out = self.sa(self.ln1(x), ablate=ablate)
            attn = None
        x = x + sa_out                       # (B, T, C) residual after attention
        x = x + self.ffwd(self.ln2(x))       # (B, T, C) residual after FFN
        if return_attn:
            return x, attn
        return x
