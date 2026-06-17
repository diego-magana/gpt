"""Seed-robustness check (build/repro script). For a given seed, train the full
architecture to a fixed reduced budget and record which head is most causally
important and whether it is a sharp previous-token head. Run once per seed; each
call appends to assets/seed_robustness.json.

The budget is deliberately small (3000 steps) — head specialization is already
unambiguous well before convergence, and the point is only to check that *some*
layer-0/1 head reliably becomes the dominant previous-token head, not to match
the 30k analysis model's loss.

    python seed_sweep.py --seed 0
"""
import argparse, json, sys
from pathlib import Path
import torch
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from gpt import make_dataset, set_seed
from gpt.data import get_batch
from gpt.analysis import eval_batches, attention_summary, classify_head, head_ablation
from models import GPT, GPTConfig

OUT = ROOT / "assets" / "seed_robustness.json"
STEPS = 3000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    seed = ap.parse_args().seed

    g = set_seed(seed)
    ds = make_dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(STEPS):
        xb, yb = get_batch(ds, "train", cfg.block_size, 16, "cpu", g)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    batches = eval_batches(ds, cfg.block_size, 32, 8)
    base, delta, _ = head_ablation(model, ds, batches)
    _, stats = attention_summary(model, batches)
    nL, nH = cfg.n_layer, cfg.n_head
    top = max(((delta[l, h], l, h) for l in range(nL) for h in range(nH)))
    td, tl, th = top
    s = stats[(tl, th)]
    rec = {
        "seed": seed, "steps": STEPS, "base_loss": round(base, 4),
        "top_head": [tl, th], "top_delta": round(td, 4),
        "top_prev": round(s["prev"], 3), "top_entropy": round(s["entropy"], 3),
        "top_label": classify_head(s),
        "top_is_prev_token_L01": bool(s["prev"] > 0.35 and tl in (0, 1)),
    }
    data = json.loads(OUT.read_text()) if OUT.exists() else []
    data = [d for d in data if d["seed"] != seed] + [rec]
    data.sort(key=lambda d: d["seed"])
    OUT.write_text(json.dumps(data, indent=1))
    print(f"seed {seed}: top head L{tl}H{th} Δ={td:.4f} prev={s['prev']:.3f} "
          f"label='{classify_head(s)}' -> prev-token in L0/1: {rec['top_is_prev_token_L01']}")


if __name__ == "__main__":
    main()
