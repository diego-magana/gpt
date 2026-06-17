"""Interpretability analysis primitives, factored out of the notebook so the
notebook stays readable and the heavy numerical work is unit-testable and
re-runnable. Every function here is correlational or causal *measurement* on the
trained model — none of it changes the weights.

Four analyses:
  1. attention_summary   — per-head attention-pattern statistics (correlational)
  2. head_ablation       — Δ val-loss from zeroing each head (causal, coarse)
  3. activation_patching — residual-stream interchange interventions (causal)
  4. residual_profile    — residual-stream norm growth + logit-lens sharpening
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from gpt import make_dataset, estimate_loss, TrainConfig
from gpt.data import get_batch, Dataset


# --------------------------------------------------------------------------
# Shared eval data
# --------------------------------------------------------------------------

def eval_batches(dataset: Dataset, block_size: int, batch_size: int,
                 n_batches: int, seed: int = 4242):
    """A fixed list of held-out (x, y) batches, reproducible across runs."""
    g = torch.Generator(); g.manual_seed(seed)
    return [get_batch(dataset, "val", block_size, batch_size, "cpu", g)
            for _ in range(n_batches)]


# --------------------------------------------------------------------------
# 1. Attention-pattern characterization (correlational)
# --------------------------------------------------------------------------

def attention_summary(model, batches):
    """Average attention maps and per-head pattern statistics.

    Returns
    -------
    mean_maps : (n_layer, n_head, T, T) batch-averaged attention matrices
    stats     : dict[(layer, head)] -> {prev, first, self, dist, entropy}

    Statistics are computed per (sequence, query position) and averaged, with the
    first query position excluded because it is forced to attend to itself.
    """
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    T = model.config.block_size
    acc = torch.zeros(n_layer, n_head, T, T)
    count = 0
    # per-head metric accumulators
    prev = torch.zeros(n_layer, n_head)
    first = torch.zeros(n_layer, n_head)
    selfn = torch.zeros(n_layer, n_head)
    dist = torch.zeros(n_layer, n_head)
    ent = torch.zeros(n_layer, n_head)
    qpos = 0

    with torch.no_grad():
        for xb, _ in batches:
            _, _, cache = model(xb, return_cache=True)
            B = xb.shape[0]
            for l in range(n_layer):
                A = cache.attn[l]                       # (B, n_head, T, T)
                acc[l] += A.sum(dim=0)
                for h in range(n_head):
                    a = A[:, h]                         # (B, T, T)
                    for t in range(1, T):               # skip forced t=0
                        row = a[:, t, : t + 1]          # (B, t+1) valid keys
                        prev[l, h] += row[:, t - 1].sum().item()
                        first[l, h] += row[:, 0].sum().item()
                        selfn[l, h] += row[:, t].sum().item()
                        offsets = torch.arange(t, -1, -1).float()  # distance t-s
                        dist[l, h] += (row * offsets).sum().item()
                        ent[l, h] += (-(row.clamp_min(1e-12).log() * row).sum()).item()
            count += B
            qpos += B * (T - 1)

    mean_maps = acc / count
    stats = {}
    for l in range(n_layer):
        for h in range(n_head):
            stats[(l, h)] = {
                "prev": prev[l, h].item() / qpos,
                "first": first[l, h].item() / qpos,
                "self": selfn[l, h].item() / qpos,
                "dist": dist[l, h].item() / qpos,
                "entropy": ent[l, h].item() / qpos,
            }
    return mean_maps, stats


def classify_head(s: dict) -> str:
    """Label a head from its statistics with explicit, inspectable thresholds.
    Labels are descriptive shorthand for the dominant tendency, not claims about
    a single causal role."""
    if s["prev"] > 0.35:
        return "previous-token"
    if s["first"] > 0.35:
        return "first-token / sink"
    if s["self"] > 0.45:
        return "self / current-token"
    if s["dist"] < 2.0:
        return "local (short-range)"
    return "diffuse / long-range"


# --------------------------------------------------------------------------
# 2. Head ablation (causal, coarse)
# --------------------------------------------------------------------------

def head_ablation(model, dataset, batches):
    """Δ mean val-loss when each (layer, head) is zeroed, vs the intact model."""
    def mean_loss(ablate):
        tot = 0.0
        with torch.no_grad():
            for xb, yb in batches:
                _, loss = model(xb, yb, ablate_heads=ablate)
                tot += loss.item()
        return tot / len(batches)

    base = mean_loss(set())
    n_layer, n_head = model.config.n_layer, model.config.n_head
    delta = np.zeros((n_layer, n_head))
    for l in range(n_layer):
        for h in range(n_head):
            delta[l, h] = mean_loss({(l, h)}) - base
    return base, delta


# --------------------------------------------------------------------------
# 3. Activation patching (causal interchange intervention)
# --------------------------------------------------------------------------

def activation_patching(model, dataset, n_examples=192, corrupt_pos=None, seed=7):
    """Residual-stream interchange interventions.

    For each example we take a clean context, corrupt one token a few positions
    from the end, and ask how much patching the *clean* residual-stream vector at
    each (residual-index, position) back into the corrupted run restores the
    model's clean prediction at the final position.

    metric = log-prob the run assigns to the clean run's top-1 next token.
    recovery = (patched - corrupt) / (clean - corrupt), averaged over examples.

    Corrupting a token shortly before the end (default ``T-2``) means its
    influence must travel forward to the final position through attention, so the
    recovery map can reveal *where* — which layer and position — that information
    is carried.
    """
    T = model.config.block_size
    if corrupt_pos is None:
        corrupt_pos = T - 2
    V = model.config.vocab_size
    n_layer = model.config.n_layer

    ds = dataset
    g = torch.Generator(); g.manual_seed(seed)
    data = ds.val_data

    rec = np.zeros((n_layer + 1, T))
    used = 0
    with torch.no_grad():
        for _ in range(n_examples):
            i = torch.randint(len(data) - T - 1, (1,), generator=g).item()
            x = data[i : i + T].clone().unsqueeze(0)            # (1, T)

            # clean run + cache
            logits_c, _, cache = model(x, return_cache=True)
            lp_clean = F.log_softmax(logits_c[0, -1], dim=-1)
            tok = int(lp_clean.argmax())
            m_clean = lp_clean[tok].item()

            # corrupt one token (different id) and run
            xc = x.clone()
            new = torch.randint(V, (1,), generator=g).item()
            while new == int(x[0, corrupt_pos]):
                new = torch.randint(V, (1,), generator=g).item()
            xc[0, corrupt_pos] = new
            logits_k, _ = model(xc)
            m_corrupt = F.log_softmax(logits_k[0, -1], dim=-1)[tok].item()

            denom = m_clean - m_corrupt
            if denom < 0.20:        # corruption barely mattered; skip noisy ratio
                continue
            used += 1

            for k in range(n_layer + 1):
                for p in range(corrupt_pos, T):   # only positions the corruption can reach
                    vec = cache.resid[k][0, p]    # clean activation (C,)
                    logits_p, _ = model(xc, patch={(k, p): vec})
                    m_p = F.log_softmax(logits_p[0, -1], dim=-1)[tok].item()
                    rec[k, p] += (m_p - m_corrupt) / denom

    rec /= max(used, 1)
    return rec, used, corrupt_pos


# --------------------------------------------------------------------------
# 4. Residual-stream norm growth + logit lens
# --------------------------------------------------------------------------

def residual_profile(model, batches):
    """Mean residual-stream L2 norm and logit-lens CE/accuracy at each depth.

    The logit lens applies the *final* LayerNorm + unembed to each intermediate
    residual stream, reading out what the model would predict if it stopped at
    that depth — a measure of how sharply the next-token prediction has formed.
    """
    n_layer = model.config.n_layer
    norms = np.zeros(n_layer + 1)
    ce = np.zeros(n_layer + 1)
    acc = np.zeros(n_layer + 1)
    ntok = 0
    with torch.no_grad():
        for xb, yb in batches:
            _, _, cache = model(xb, return_cache=True)
            B, T = xb.shape
            for k in range(n_layer + 1):
                r = cache.resid[k]                         # (B, T, C)
                norms[k] += r.norm(dim=-1).sum().item()
                logits = model.logit_lens(r)               # (B, T, V)
                lt = logits.view(B * T, -1)
                tgt = yb.view(B * T)
                ce[k] += F.cross_entropy(lt, tgt, reduction="sum").item()
                acc[k] += (lt.argmax(-1) == tgt).sum().item()
            ntok += B * T
    return norms / ntok, ce / ntok, acc / ntok
