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
    a single causal role. The "sharp" tier separates heads that put most of their
    weight on the previous token (the two that matter here) from heads that merely
    lean that way."""
    if s["prev"] > 0.70:
        return "previous-token (sharp)"
    if s["prev"] > 0.35:
        return "previous-token (weak)"
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
    """Δ mean val-loss when each (layer, head) is zeroed, vs the intact model.

    Returns ``base, delta, delta_sem``. The error bar is the standard error of
    the paired per-batch difference (ablated minus intact on the *same* batch),
    which cancels the batch-to-batch variation in the base loss and so is a
    tighter estimate of the ablation effect than differencing two independent
    means."""
    def per_batch_losses(ablate):
        out = []
        with torch.no_grad():
            for xb, yb in batches:
                _, loss = model(xb, yb, ablate_heads=ablate)
                out.append(loss.item())
        return np.array(out)

    base_losses = per_batch_losses(set())
    base = float(base_losses.mean())
    n_layer, n_head = model.config.n_layer, model.config.n_head
    delta = np.zeros((n_layer, n_head))
    delta_sem = np.zeros((n_layer, n_head))
    for l in range(n_layer):
        for h in range(n_head):
            d = per_batch_losses({(l, h)}) - base_losses          # paired, per batch
            delta[l, h] = d.mean()
            delta_sem[l, h] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return base, delta, delta_sem


def redundancy_test(model, batches, a=(0, 3), b=(1, 1)):
    """Pairwise ablation to test whether head ``a`` is functionally redundant
    with head ``b``.

    If ``a`` is a cheap-to-ablate head only because ``b`` carries a stronger copy
    of the same signal downstream, then ablating ``a`` should cost much more once
    ``b`` is *also* gone. We compare the marginal cost of removing ``a`` alone to
    the marginal cost of removing ``a`` when ``b`` is already ablated. A large gap
    supports redundancy; a small one refutes it."""
    def mean_loss(ablate):
        tot = 0.0
        with torch.no_grad():
            for xb, yb in batches:
                _, loss = model(xb, yb, ablate_heads=set(ablate))
                tot += loss.item()
        return tot / len(batches)

    base = mean_loss([])
    la = mean_loss([a])
    lb = mean_loss([b])
    lab = mean_loss([a, b])
    return {
        "a": a, "b": b, "base": base,
        "ablate_a": la, "ablate_b": lb, "ablate_both": lab,
        "marg_a_alone": la - base,        # cost of removing a with everything else intact
        "marg_a_given_b": lab - lb,       # cost of removing a once b is already gone
    }


# --------------------------------------------------------------------------
# 3. Activation patching (causal interchange intervention)
# --------------------------------------------------------------------------

def _logit_diff(logits, clean_top, corrupt_top):
    """Contrastive metric: how much the final-position logits favor the clean
    answer over the token the corruption pushed. Less sensitive to overall
    distribution shifts than the log-prob of a single token."""
    return (logits[0, -1, clean_top] - logits[0, -1, corrupt_top]).item()


def activation_patching(model, dataset, n_examples=256, corrupt_pos=None, seed=7):
    """Residual-stream interchange interventions, localized to (layer, position).

    For each example: take a clean context, corrupt one token a few positions
    from the end, and patch the *clean* residual-stream vector at each
    (residual-index, position) back into the corrupted run, measuring how much of
    the clean prediction returns.

    Metric is a contrastive logit difference (clean top token vs. the token the
    corruption promotes). Only examples where the corruption actually *flips* the
    top-1 prediction are used — a principled, reportable selection rather than an
    arbitrary threshold. ``recovery = (patched - corrupt) / (clean - corrupt)``.

    Returns ``rec, rec_sem, used, corrupt_pos`` where the maps are
    ``(n_layer+1, T)``. Note this patches the residual *sum* at a position, so it
    localizes information to a (layer, position) but does not isolate a head — see
    :func:`head_patching` for that.
    """
    T = model.config.block_size
    if corrupt_pos is None:
        corrupt_pos = T - 2
    V = model.config.vocab_size
    n_layer = model.config.n_layer
    g = torch.Generator(); g.manual_seed(seed)
    data = dataset.val_data

    samples = {(k, p): [] for k in range(n_layer + 1) for p in range(corrupt_pos, T)}
    used = 0
    with torch.no_grad():
        for _ in range(n_examples):
            i = torch.randint(len(data) - T - 1, (1,), generator=g).item()
            x = data[i : i + T].clone().unsqueeze(0)
            logits_c, _, cache = model(x, return_cache=True)
            clean_top = int(logits_c[0, -1].argmax())

            xc = x.clone()
            new = torch.randint(V, (1,), generator=g).item()
            while new == int(x[0, corrupt_pos]):
                new = torch.randint(V, (1,), generator=g).item()
            xc[0, corrupt_pos] = new
            logits_k, _ = model(xc)
            corrupt_top = int(logits_k[0, -1].argmax())
            if corrupt_top == clean_top:        # corruption left the prediction unchanged
                continue
            m_clean = _logit_diff(logits_c, clean_top, corrupt_top)
            m_corrupt = _logit_diff(logits_k, clean_top, corrupt_top)
            denom = m_clean - m_corrupt
            if denom < 1e-3:
                continue
            used += 1
            for k in range(n_layer + 1):
                for p in range(corrupt_pos, T):
                    logits_p, _ = model(xc, patch={(k, p): cache.resid[k][0, p]})
                    samples[(k, p)].append(
                        (_logit_diff(logits_p, clean_top, corrupt_top) - m_corrupt) / denom
                    )

    rec = np.zeros((n_layer + 1, T))
    rec_sem = np.zeros((n_layer + 1, T))
    for (k, p), vals in samples.items():
        arr = np.array(vals) if vals else np.array([0.0])
        rec[k, p] = arr.mean()
        rec_sem[k, p] = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return rec, rec_sem, used, corrupt_pos


def head_patching(model, dataset, n_examples=256, corrupt_pos=None, seed=7):
    """Head-level interchange intervention — the test residual patching cannot do.

    Same clean/corrupt setup as :func:`activation_patching`, but instead of
    patching the residual stream at a position, we splice a single head's *clean*
    output into the corrupted run (all positions) and measure recovery. Because
    only that one head's contribution is replaced, the recovered prediction is
    attributable to that head alone — this is what isolates L1 H1 rather than
    inferring it from the coincidence of "block 1 matters" and "L1 H1 ablates
    hardest."

    Returns ``rec, rec_sem, used, corrupt_pos`` with maps shaped
    ``(n_layer, n_head)``.
    """
    T = model.config.block_size
    if corrupt_pos is None:
        corrupt_pos = T - 2
    V = model.config.vocab_size
    n_layer, n_head = model.config.n_layer, model.config.n_head
    g = torch.Generator(); g.manual_seed(seed)
    data = dataset.val_data

    samples = [[[] for _ in range(n_head)] for _ in range(n_layer)]
    used = 0
    with torch.no_grad():
        for _ in range(n_examples):
            i = torch.randint(len(data) - T - 1, (1,), generator=g).item()
            x = data[i : i + T].clone().unsqueeze(0)
            logits_c, _, cache = model(x, return_cache=True)
            clean_top = int(logits_c[0, -1].argmax())

            xc = x.clone()
            new = torch.randint(V, (1,), generator=g).item()
            while new == int(x[0, corrupt_pos]):
                new = torch.randint(V, (1,), generator=g).item()
            xc[0, corrupt_pos] = new
            logits_k, _ = model(xc)
            corrupt_top = int(logits_k[0, -1].argmax())
            if corrupt_top == clean_top:
                continue
            m_clean = _logit_diff(logits_c, clean_top, corrupt_top)
            m_corrupt = _logit_diff(logits_k, clean_top, corrupt_top)
            denom = m_clean - m_corrupt
            if denom < 1e-3:
                continue
            used += 1
            for l in range(n_layer):
                for h in range(n_head):
                    logits_p, _ = model(xc, patch_heads={(l, h): cache.head_out[l][:, h]})
                    samples[l][h].append(
                        (_logit_diff(logits_p, clean_top, corrupt_top) - m_corrupt) / denom
                    )

    rec = np.zeros((n_layer, n_head))
    rec_sem = np.zeros((n_layer, n_head))
    for l in range(n_layer):
        for h in range(n_head):
            arr = np.array(samples[l][h]) if samples[l][h] else np.array([0.0])
            rec[l, h] = arr.mean()
            rec_sem[l, h] = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return rec, rec_sem, used, corrupt_pos


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


# --------------------------------------------------------------------------
# 6. Weight-space circuits (QK / OV) — no data, no forward pass
# --------------------------------------------------------------------------

def _head_weights(model, layer, head):
    """(W_Q, W_K, W_V, W_O_slice) for one head. W_O_slice is the block of the
    output projection that reads this head's slot in the concatenated output."""
    h = model.blocks[layer].sa.heads[head]
    hs = h.query.weight.shape[0]
    W_O = model.blocks[layer].sa.proj.weight[:, head * hs:(head + 1) * hs]   # (C, hs)
    return h.query.weight, h.key.weight, h.value.weight, W_O


def qk_positional_circuit(model, layer, head):
    """Attention logits this head would produce from *position alone*.

    Runs the QK circuit on the position embeddings with the token embeddings left
    out: ``(P W_Q)(P W_K)^T / sqrt(head_size)``. Every behavioral measure so far
    (attention maps, ablation, patching) reads the network's output on data; this
    reads the weights directly, so a previous-token pattern here is a property of
    the learned parameters rather than of the corpus. Returns ``(T, T)``.
    """
    P = model.position_embedding_table.weight              # (T, C)
    W_Q, W_K, _, _ = _head_weights(model, layer, head)
    hs = W_Q.shape[0]
    with torch.no_grad():
        return ((P @ W_Q.T) @ (P @ W_K.T).T) / (hs ** 0.5)  # (T, T)


def qk_offset_profile(model, layer, head):
    """Where the positional QK circuit points, per query position.

    For each query ``i`` takes the argmax key over the causal window ``j <= i`` and
    records the offset ``i - j``. A previous-token head lands on offset 1 almost
    everywhere. Returns ``dict(offsets, frac_offset_one, mean_offset)``.
    """
    S = qk_positional_circuit(model, layer, head)
    T = S.shape[0]
    offsets = np.array([i - int(S[i, :i + 1].argmax()) for i in range(1, T)])
    return {
        "offsets": offsets,
        "frac_offset_one": float((offsets == 1).mean()),
        "mean_offset": float(offsets.mean()),
    }


def ov_circuit(model, layer, head):
    """What the head writes into the residual stream for each attended token.

    ``E W_V W_O`` gives the residual-stream write per vocabulary token ``(V, C)``;
    composing with the unembedding gives its direct effect on logits ``(V, V)``.
    Returns ``(write, to_logits)``. A head that copies the attended token to the
    output would show a dominant diagonal in ``to_logits``.
    """
    E = model.token_embedding_table.weight                 # (V, C)
    U = model.lm_head.weight                               # (V, C)
    _, _, W_V, W_O = _head_weights(model, layer, head)
    with torch.no_grad():
        write = (E @ W_V.T) @ W_O.T                        # (V, C)
        return write, write @ U.T                          # (V, C), (V, V)


def ov_copy_profile(model, layer, head):
    """Is the OV circuit a copy, and does it preserve token identity at all?

    Three diagnostics. ``top1_logit``: how often attending to token *i* most raises
    logit *i* (a direct-path copier scores ~1). ``top1_embed``: how often the write
    aligns most with token *i*'s own embedding (a copier writing in the embedding
    basis scores ~1). ``sep_ratio``: mean nearest-neighbour distance over mean
    pairwise distance among the ``V`` written vectors — near 0 means the writes
    collapse and token identity is destroyed; well above 0 means the tokens stay
    distinct (information preserved, whatever basis it lives in).
    """
    E = model.token_embedding_table.weight
    write, to_logits = ov_circuit(model, layer, head)
    with torch.no_grad():
        V = E.shape[0]
        top1_logit = float(np.mean([int(to_logits[i].argmax()) == i for i in range(V)]))
        S = (F.normalize(write, dim=1) @ F.normalize(E, dim=1).T)
        top1_embed = float(np.mean([int(S[i].argmax()) == i for i in range(V)]))
        W = write.numpy()
        D = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        nn_mean = float(D.min(1).mean())
        D[np.isinf(D)] = np.nan
        pair_mean = float(np.nanmean(D))
        return {
            "top1_logit": top1_logit,
            "top1_embed": top1_embed,
            "rank": int(np.linalg.matrix_rank(W, tol=1e-4)),
            "sep_ratio": nn_mean / pair_mean,
        }
