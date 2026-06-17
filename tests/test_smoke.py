"""Smoke tests — fast correctness checks, not a benchmark.

Each test asserts a property that would break silently under a plausible bug:
the embedding really is a one-hot matmul, the causal mask really zeros the
future, attention rows really are distributions, the activation-patching
machinery really is a no-op when fed clean activations, training really reduces
loss, and a checkpoint really round-trips. Data is loaded through an absolute
path derived from this file so ``pytest`` passes from any working directory.
"""

from __future__ import annotations

from pathlib import Path

import torch

from gpt import (
    CharTokenizer,
    Dataset,
    TrainConfig,
    estimate_loss,
    generate,
    get_batch,
    make_dataset,
    set_seed,
    train,
)
from models import GPT, BigramLanguageModel, GPTConfig

# Absolute path to the bundled corpus — robust to the caller's CWD (spec 11.1).
CORPUS = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def _small_dataset() -> Dataset:
    # Use a slice of the real corpus so the alphabet is realistic but construction
    # stays instant.
    text = CORPUS.read_text(encoding="utf-8")[:20_000]
    return make_dataset(text)


# --- (a) imports + tokenizer round-trip ------------------------------------

def test_tokenizer_roundtrip():
    tok = CharTokenizer("hello transformer world")
    s = "world"
    assert tok.decode(tok.encode(s)) == s
    # Encoding is a bijection on the fitted alphabet.
    assert sorted(tok.stoi.values()) == list(range(tok.vocab_size))


# --- (b) construction at small scale ---------------------------------------

def test_gpt_constructs_and_runs():
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    xb, yb = get_batch(ds, "train", cfg.block_size, 8, "cpu", g)
    logits, loss = model(xb, yb)
    assert logits.shape == (8, cfg.block_size, ds.vocab_size)
    assert loss.item() > 0
    # Untrained loss should sit near the uniform-prediction value, ln(vocab).
    import math
    assert abs(loss.item() - math.log(ds.vocab_size)) < 1.0


# --- (c) mathematical correctness properties -------------------------------

def test_embedding_is_onehot_matmul():
    """Embedding(idx) must equal one_hot(idx) @ W — the definition of a lookup
    table as a linear map. A transpose or indexing bug breaks this exactly."""
    torch.manual_seed(0)
    emb = torch.nn.Embedding(7, 5)
    idx = torch.tensor([0, 3, 6, 1])
    direct = emb(idx)
    onehot = torch.nn.functional.one_hot(idx, num_classes=7).float() @ emb.weight
    assert torch.allclose(direct, onehot, atol=1e-6)


def test_attention_is_causal_and_normalized():
    """Future positions receive exactly zero attention, and every query's
    attention row is a probability distribution (sums to 1)."""
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    xb, _ = get_batch(ds, "train", cfg.block_size, 4, "cpu", g)
    _, _, cache = model(xb, return_cache=True)
    A = cache.attn[0]                                  # (B, n_head, T, T)
    T = A.shape[-1]
    future = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert A[..., future].abs().max().item() < 1e-6   # no leakage to the future
    rows = A.sum(dim=-1)                               # (B, n_head, T)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_clean_activation_patch_is_noop():
    """Splicing a run's own cached residual activation back into it must not
    change the output — the correctness invariant the patching analysis relies
    on. If this fails, every patching number is meaningless."""
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    xb, _ = get_batch(ds, "train", cfg.block_size, 4, "cpu", g)
    base_logits, _, cache = model(xb, return_cache=True)
    clean_vec = cache.resid[1][:, 3, :]               # (B, C) at block-1 input, pos 3
    patched_logits, _ = model(xb, patch={(1, 3): clean_vec})
    assert torch.allclose(base_logits, patched_logits, atol=1e-6)


def test_clean_head_patch_is_noop():
    """The head-level analogue: splicing a head's own clean output back in must
    be an exact no-op. This is what makes the head-patching recovery numbers
    trustworthy as a per-head causal measure."""
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    xb, yb = get_batch(ds, "train", cfg.block_size, 4, "cpu", g)
    base_logits, _, cache = model(xb, return_cache=True)
    clean_head = cache.head_out[1][:, 1]              # (B, T, head_size) for L1 H1
    patched_logits, _ = model(xb, patch_heads={(1, 1): clean_head})
    assert torch.allclose(base_logits, patched_logits, atol=1e-6)
    # Zeroing a head via patch_heads must equal ablating it.
    _, loss_zero = model(xb, yb, patch_heads={(1, 1): torch.zeros_like(clean_head)})
    _, loss_abl = model(xb, yb, ablate_heads={(1, 1)})
    assert abs(loss_zero.item() - loss_abl.item()) < 1e-6


def test_head_ablation_changes_loss():
    """Zeroing a head must perturb the loss; a no-op would mean the ablation hook
    is silently disconnected from the forward pass."""
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    xb, yb = get_batch(ds, "train", cfg.block_size, 8, "cpu", g)
    _, base = model(xb, yb)
    _, ablated = model(xb, yb, ablate_heads={(0, 0), (1, 2)})
    assert abs(ablated.item() - base.item()) > 1e-6


# --- (d) training reduces loss ---------------------------------------------

def test_training_reduces_loss():
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    tcfg = TrainConfig(max_iters=120, eval_interval=40, eval_iters=20, batch_size=16)
    history = train(model, ds, cfg.block_size, tcfg, generator=g, verbose=False)
    assert history[-1]["train"] < history[0]["train"]


def test_bigram_trains_and_generates():
    ds = _small_dataset()
    model = BigramLanguageModel(ds.vocab_size)
    g = set_seed(0)
    tcfg = TrainConfig(max_iters=120, eval_interval=40, eval_iters=20, batch_size=16)
    history = train(model, ds, block_size=8, cfg=tcfg, generator=g, verbose=False)
    assert history[-1]["train"] < history[0]["train"]
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)


# --- checkpoint round-trip --------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    ds = _small_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    xb, _ = get_batch(ds, "train", cfg.block_size, 4, "cpu", g)
    before, _ = model(xb)
    path = tmp_path / "ckpt.pth"
    model.save(str(path), history=[{"step": 0, "train": 1.0, "val": 1.0}])
    restored = GPT.from_pretrained(str(path))
    after, _ = restored(xb)
    assert torch.allclose(before, after, atol=1e-6)
    assert restored.config.n_embd == cfg.n_embd
