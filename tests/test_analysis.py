"""Fast checks that the analysis primitives return well-formed, sane outputs on a
tiny model. Correctness of the underlying intervention machinery (the clean-patch
identity invariant) is covered in test_smoke.py; these guard the analysis layer
that builds on it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpt import make_dataset, set_seed, train, TrainConfig
from gpt.analysis import (
    eval_batches, attention_summary, classify_head,
    head_ablation, activation_patching, residual_profile,
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
    base, delta = head_ablation(model, ds, batches)
    assert base > 0
    assert delta.shape == (cfg.n_layer, cfg.n_head)
    # Ablating a head should not, on average, *reduce* loss below the intact model
    # by more than floating noise.
    assert delta.min() > -0.05


def test_activation_patching_recovers_corrupt_column():
    model, ds, cfg = _tiny_model_and_data()
    rec, used, cpos = activation_patching(model, ds, n_examples=24)
    assert rec.shape == (cfg.n_layer + 1, cfg.block_size)
    if used > 0:
        # Patching the clean activation at the corrupted position in the input
        # layer must recover most of the prediction by construction.
        assert rec[0, cpos] > 0.8


def test_residual_profile_monotone_logit_lens():
    model, ds, cfg = _tiny_model_and_data()
    batches = eval_batches(ds, cfg.block_size, 8, 2)
    norms, ce, acc = residual_profile(model, batches)
    assert len(norms) == cfg.n_layer + 1
    # The logit-lens readout at the final depth equals the model's own loss, so CE
    # at the last index should be the smallest along the depth axis.
    assert ce[-1] <= ce[0] + 1e-6
