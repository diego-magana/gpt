"""The full GPT — token+position embeddings, a stack of pre-norm transformer
blocks, and a language-model head — assembled from the primitives in
``gpt.layers`` and instrumented for the interpretability analysis.

The architecture is locked to Karpathy's capstone (``n_embd=64``, ``n_head=4``,
``n_layer=4``, ``block_size=32``) so the trained loss stays comparable to the
reference run. What the source lacks and this adds is a forward pass that can
(1) cache attention weights and the residual stream, (2) ablate chosen heads,
and (3) patch residual-stream activations — the three surfaces the analysis
notebook drives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.layers import ActivationCache, Block


@dataclass
class GPTConfig:
    """Architectural hyperparameters. Kept as a dataclass rather than module
    globals (the source's approach) so a checkpoint can carry the exact config
    that produced it and rebuild the matching model on load — the single most
    common cause of "checkpoint loads but outputs garbage" is an architecture
    that silently drifted from the one that was trained."""

    vocab_size: int
    block_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0


class GPT(nn.Module):
    """A decoder-only transformer language model at teaching scale (~0.21M params).

    Lifecycle of a token (the forward pass)
    ---------------------------------------
    1. **Embed.** ``token_embedding_table[idx]`` maps each id to a learned
       ``n_embd`` vector ("what this character means"); ``position_embedding_
       table[0..T-1]`` adds a learned vector per slot ("where it sits"). Their
       sum is the initial residual stream — attention is permutation-invariant, so
       without the position term the model could not tell ``"ab"`` from ``"ba"``.
    2. **Process.** ``n_layer`` blocks each read the residual stream, route
       information across positions (attention), compute on it (FFN), and write
       their result back via residual addition. The stream is the model's working
       memory; every block reads and writes the same ``(B, T, n_embd)`` tensor.
    3. **Read out.** A final LayerNorm, then ``lm_head`` projects each position's
       vector to ``vocab_size`` logits — the next-character distribution.

    Why ``ModuleList`` rather than ``Sequential`` for the blocks
    -----------------------------------------------------------
    The source wraps the blocks in ``nn.Sequential``. We use a ``ModuleList`` and
    loop explicitly so the forward pass can inject per-layer behavior — capture
    the residual stream entering each block, route ablation flags to the right
    layer, and overwrite activations for patching. ``Sequential`` hides the loop
    and forbids exactly the per-layer access the analysis needs.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        C, V, T = config.n_embd, config.vocab_size, config.block_size

        self.token_embedding_table = nn.Embedding(V, C)
        self.position_embedding_table = nn.Embedding(T, C)
        self.blocks = nn.ModuleList(
            [Block(C, config.n_head, T, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V)

    # -- introspection -------------------------------------------------------

    def num_params(self) -> int:
        """Total parameter count. Reported in millions in the training scripts so
        the ~0.21M figure can be checked against the source at a glance."""
        return sum(p.numel() for p in self.parameters())

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_cache: bool = False,
        ablate_heads: set[tuple[int, int]] | None = None,
        patch: dict[tuple[int, int], torch.Tensor] | None = None,
        patch_heads: dict[tuple[int, int], torch.Tensor] | None = None,
    ):
        """Run the model, optionally capturing/intervening on internals.

        Parameters
        ----------
        targets
            If given, also return the mean cross-entropy loss.
        return_cache
            If True, return an :class:`ActivationCache` of per-layer attention
            ``(B, n_head, T, T)``, the residual stream entering each block, and
            each layer's per-head outputs ``(B, n_head, T, head_size)``.
        ablate_heads
            Set of ``(layer, head)`` pairs whose attention outputs are zeroed —
            the coarse head-importance intervention.
        patch
            Dict mapping ``(k, pos) -> vector`` that overwrites the residual
            stream at residual index ``k`` (``0`` = post-embedding, ``i+1`` =
            output of block ``i``) and position ``pos``. Localizes information to
            a (layer, position) but mixes every head's contribution at that site.
        patch_heads
            Dict mapping ``(layer, head) -> tensor`` of shape
            ``(B, T, head_size)`` that *replaces* a single head's output with the
            supplied one. Unlike ``patch``, this isolates one head: splice its
            clean output into a corrupted run and the recovered prediction is
            attributable to that head alone.

        Shape walk
        ----------
            idx            : (B, T)
            tok_emb        : (B, T, C)
            pos_emb        : (T, C)         broadcast-added over batch
            x  (resid)     : (B, T, C)      stable width through every block
            logits         : (B, T, vocab)
        """
        B, T = idx.shape
        device = idx.device
        ablate_heads = ablate_heads or set()
        patch = patch or {}
        patch_heads = patch_heads or {}
        cache = ActivationCache() if return_cache else None

        def maybe_patch(x: torch.Tensor, k: int) -> torch.Tensor:
            # Overwrite residual-stream entries scheduled for index k. Done out of
            # place so a patched run cannot corrupt the cached clean activations.
            for (kk, pos), vec in patch.items():
                if kk == k:
                    x = x.clone()
                    x[:, pos, :] = vec
            return x

        tok_emb = self.token_embedding_table(idx)                       # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb                                           # (B, T, C)

        x = maybe_patch(x, 0)
        if cache is not None:
            cache.resid[0] = x.detach()

        for i, block in enumerate(self.blocks):
            # Per-layer interventions for *this* block.
            layer_ablate = {h for (l, h) in ablate_heads if l == i}
            layer_patch_heads = {h: v for (l, h), v in patch_heads.items() if l == i}
            if cache is not None or layer_patch_heads:
                x, attn, head_outs = block(
                    x, return_attn=True, ablate=layer_ablate, patch_heads=layer_patch_heads
                )
                if cache is not None:
                    cache.attn[i] = attn.detach()                       # (B, n_head, T, T)
                    cache.head_out[i] = head_outs.detach()              # (B, n_head, T, head_size)
            else:
                x = block(x, ablate=layer_ablate)
            x = maybe_patch(x, i + 1)
            if cache is not None:
                cache.resid[i + 1] = x.detach()

        x = self.ln_f(x)                                               # (B, T, C)
        logits = self.lm_head(x)                                       # (B, T, vocab)

        loss = None
        if targets is not None:
            Bv, Tv, Cv = logits.shape
            loss = F.cross_entropy(logits.view(Bv * Tv, Cv), targets.view(Bv * Tv))

        if return_cache:
            return logits, loss, cache
        return logits, loss

    # -- logit lens ----------------------------------------------------------

    @torch.no_grad()
    def logit_lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Read a mid-stack residual-stream tensor as if it were final: apply the
        final LayerNorm and ``lm_head`` to it.

        This is the "logit lens" — it asks what the model *would* predict if it
        stopped at a given depth. Applied to ``cache.resid[k]`` for increasing
        ``k`` it traces how the next-token prediction sharpens layer by layer,
        because every block writes into the same stream that ``lm_head`` reads.
        """
        return self.lm_head(self.ln_f(resid))                          # (..., vocab)

    # -- checkpointing -------------------------------------------------------

    def save(self, path: str, history: list | None = None, extra: dict | None = None) -> None:
        """Persist weights, the exact ``GPTConfig``, and optional loss history.

        Storing the config beside the weights is what makes :meth:`from_pretrained`
        able to rebuild the right architecture without the caller restating
        hyperparameters — the checkpoint is self-describing.
        """
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": self.config.__dict__,
                "history": history,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "GPT":
        """Rebuild a model from a checkpoint saved by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
