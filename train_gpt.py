"""Reproduce the committed checkpoint assets/gpt.pth.

    python train_gpt.py

Trains the full GPT to convergence (30000 steps) with seed 1337 and writes
assets/gpt.pth plus assets/gpt_history.json. Karpathy's teaching budget is 5000
steps (val ~1.80); training continues to ~30k (val ~1.65) because the analysis in
notebooks/05_attention_analysis.ipynb benefits from a converged model, and the
step count is reported rather than matched to the shorter reference. Because
evaluation uses an isolated fixed-seed generator (see gpt.train.estimate_loss),
the trained weights are independent of eval cadence and reproducible across
machines. On a CPU this takes ~15 minutes; on a GPU, well under a minute.
"""
import json
from pathlib import Path

import torch

from gpt import make_dataset, set_seed, train, generate, TrainConfig
from models import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent


def main():
    g = set_seed(1337)
    ds = make_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size)          # locked source architecture
    model = GPT(cfg)
    print(f"{model.num_params()/1e6:.6f}M parameters")

    tcfg = TrainConfig(max_iters=30000, eval_interval=1000, eval_iters=200,
                       learning_rate=1e-3, batch_size=16, device="cpu")
    history = train(model, ds, cfg.block_size, tcfg, generator=g)

    out = ROOT / "assets" / "gpt.pth"
    out.parent.mkdir(exist_ok=True)
    model.save(str(out), history=history, extra={"seed": 1337})
    (ROOT / "assets" / "gpt_history.json").write_text(json.dumps(history))
    print(f"final: {history[-1]}")
    print(f"saved {out}")

    sample = generate(model, torch.zeros((1, 1), dtype=torch.long),
                      max_new_tokens=400, block_size=cfg.block_size)
    print("\n----- sample -----")
    print(ds.tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
