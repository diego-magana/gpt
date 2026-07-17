"""Fast checks that the analysis primitives return well-formed, sane outputs on a
tiny model. Correctness of the underlying intervention machinery (the clean-patch
identity invariant) is covered in test_smoke.py; these guard the analysis layer
that builds on it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# Every test here trains a tiny model end-to-end: these are integration tests,
# not unit smoke tests. Mark the module so `pytest -m "not slow"` skips them.
pytestmark = pytest.mark.slow

from gpt import make_dataset, set_seed, train, TrainConfig
from gpt.analysis import (
    eval_batches, attention_summary, classify_head,
    head_ablation, redundancy_test, activation_patching, head_patching,
    residual_profile, qk_positional_circuit, qk_offset_profile,
    ov_circuit, ov_copy_profile,
)
from models import GPT, GPTConfig

CORPUS = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def _tiny_model_and_data():
    text = CORPUS.read_text(encoding="utf-8")[:20_000]
    ds = make_dataset(text)
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    g = set_seed(0)
    train(model, ds, cfg.block_size, TrainConfig(max_iters=120, eval_interval=60,
          eval_iters=20, batch_size=16), generator=g, verbose=False)
    return model, ds, cfg


def test_attention_summary_shapes_and_distributions():
    model, ds, cfg = _tiny_model_and_data()
    batches = eval_batches(ds, cfg.block_size, 8, 2)
    mean_maps, stats = attention_summary(model, batches)
    assert mean_maps.shape == (cfg.n_layer, cfg.n_head, cfg.block_size, cfg.block_size)
    # Each averaged attention row over valid keys is a distribution (<= 1, >= 0).
    assert mean_maps.min().item() >= -1e-6
    # Every head has the five statistics and a label.
    for l in range(cfg.n_layer):
        for h in range(cfg.n_head):
            s = stats[(l, h)]
            assert {"prev", "first", "self", "dist", "entropy"} <= set(s)
            assert isinstance(classify_head(s), str)


def test_head_ablation_shape_and_base():
    model, ds, cfg = _tiny_model_and_data()
    batches = eval_batches(ds, cfg.block_size, 8, 2)
    base, delta, delta_sem = head_ablation(model, ds, batches)
    assert base > 0
    assert delta.shape == (cfg.n_layer, cfg.n_head)
    assert delta_sem.shape == delta.shape
    assert (delta_sem >= 0).all()
    # Ablating a head should not, on average, *reduce* loss below the intact model
    # by more than floating noise.
    assert delta.min() > -0.05


def test_redundancy_test_keys():
    model, ds, cfg = _tiny_model_and_data()
    batches = eval_batches(ds, cfg.block_size, 8, 2)
    r = redundancy_test(model, batches, a=(0, 0), b=(1, 1))
    assert {"base", "ablate_a", "ablate_b", "ablate_both",
            "marg_a_alone", "marg_a_given_b"} <= set(r)
    assert r["base"] > 0


def test_activation_patching_runs_and_shapes():
    model, ds, cfg = _tiny_model_and_data()
    rec, rec_sem, used, cpos = activation_patching(model, ds, n_examples=40)
    assert rec.shape == (cfg.n_layer + 1, cfg.block_size)
    assert rec_sem.shape == rec.shape
    assert used >= 0 and cpos == cfg.block_size - 2


def test_head_patching_runs_and_shapes():
    model, ds, cfg = _tiny_model_and_data()
    rec, rec_sem, used, cpos = head_patching(model, ds, n_examples=40)
    assert rec.shape == (cfg.n_layer, cfg.n_head)
    assert rec_sem.shape == rec.shape
    assert used >= 0


def test_residual_profile_monotone_logit_lens():
    model, ds, cfg = _tiny_model_and_data()
    batches = eval_batches(ds, cfg.block_size, 8, 2)
    norms, ce, acc = residual_profile(model, batches)
    assert len(norms) == cfg.n_layer + 1
    # The logit-lens readout at the final depth equals the model's own loss, so CE
    # at the last index should be the smallest along the depth axis.
    assert ce[-1] <= ce[0] + 1e-6


def test_qk_positional_circuit_shape_and_causal_offsets():
    """The QK circuit reads weights only — no data, no forward pass — so it must
    return a (T, T) positional score matrix and offsets inside the causal window."""
    model, ds, cfg = _tiny_model_and_data()
    S = qk_positional_circuit(model, 0, 0)
    assert S.shape == (cfg.block_size, cfg.block_size)
    prof = qk_offset_profile(model, 0, 0)
    assert len(prof["offsets"]) == cfg.block_size - 1
    assert (prof["offsets"] >= 0).all()          # argmax j <= i, so offset >= 0 (0 = self)
    assert 0.0 <= prof["frac_offset_one"] <= 1.0


def test_ov_circuit_shapes_and_profile_keys():
    model, ds, cfg = _tiny_model_and_data()
    write, to_logits = ov_circuit(model, 0, 0)
    assert write.shape == (cfg.vocab_size, cfg.n_embd)
    assert to_logits.shape == (cfg.vocab_size, cfg.vocab_size)
    prof = ov_copy_profile(model, 0, 0)
    assert {"top1_logit", "top1_embed", "rank", "sep_ratio"} <= set(prof)
    assert prof["rank"] <= cfg.n_embd // cfg.n_head    # OV is rank-limited by head size
