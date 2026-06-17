"""The bigram language model — the baseline the whole progression improves on.

This is the simplest model that can be trained with the exact same loop, loss,
and sampling interface as the full GPT, which is the point: it isolates "what
does attention buy us?" by giving a memoryless control to compare against.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next character from the current character alone.

    Mathematical operation
    -----------------------
    A single ``(vocab_size, vocab_size)`` embedding table *is* the model: row
    ``i`` holds the unnormalized log-probabilities (logits) over the next
    character given that the current character is ``i``. There is no hidden
    state, no context beyond one token — the lookup returns logits directly.

        logits = E[idx]            E: (vocab, vocab),  logits: (B, T, vocab)

    Why this is the right baseline
    ------------------------------
    The bigram has no mechanism to look past the immediately-preceding character,
    so its loss is a hard floor set by the corpus's first-order character
    statistics (~2.45 nats on Tiny Shakespeare). Every architectural ingredient
    added later — attention, depth, the feed-forward network — has to earn its
    place by beating this number. Sharing the ``(forward, generate)`` interface
    with the full GPT means the same training loop and sampler drive both, so the
    comparison is clean.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)        # (B, T, vocab)

        if targets is None:
            return logits, None

        # F.cross_entropy wants (N, C) logits against (N,) targets, so collapse
        # the batch and time axes into one example axis. This is the recurring
        # PyTorch reshape every language model in the series performs.
        B, T, C = logits.shape
        logits = logits.view(B * T, C)                  # (B*T, vocab)
        targets = targets.view(B * T)                   # (B*T,)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Sample ``max_new_tokens`` continuations. No context cropping is needed
        because only the last token affects the next prediction."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                        # (B, T, vocab)
            logits = logits[:, -1, :]                    # (B, vocab) last step
            probs = F.softmax(logits, dim=-1)            # (B, vocab)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)      # (B, T+1)
        return idx
