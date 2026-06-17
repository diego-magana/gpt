# gpt

![CI](https://github.com/diego-magana/gpt/actions/workflows/ci.yml/badge.svg)

A modular, instrumented rebuild of Karpathy's *"Let's build GPT"* capstone, taken
apart to see how the trained model actually works.

The short version of what I found: in this 4-layer, 4-head character model, almost
all of the predictive work runs through a single attention head — layer 1, head 1 —
which copies the previous character into the position that predicts the next one.
Two heads *look* like they do that job; only one of them actually matters, and it
took intervening on the network to tell them apart.

This is the third in a series after
[micrograd](https://github.com/diego-magana/micrograd) (a scalar autograd engine)
and [makemore](https://github.com/diego-magana/makemore) (n-gram to WaveNet
character models). The activation-patching analysis here is the one I pointed
forward to at the end of makemore.

---

## What I found

I trained the full model to convergence and ran four analyses on it — two that just
look at the network (attention patterns, the residual stream) and two that
intervene on it (head ablation, activation patching). The interesting part is that
looking and intervening disagree, and the disagreement is where the real result is.

**The model has two heads that look like previous-token heads.** Averaging attention
over 512 held-out sequences, layer 0 is mostly diffuse, layer 1 leans entirely on
the preceding token, and layers 2–3 do shorter-range, more self-weighted mixing. Two
heads stand out as sharp previous-token heads: L1 H1, which puts ≈ 0.98 of its
attention on the preceding token with almost no entropy, and L0 H3 at ≈ 0.82. On the
attention maps they read as the same kind of head.

![Mean attention maps](assets/attention_grid.png)

**Only one of them matters.** When I zero each head and remeasure validation loss,
L1 H1 costs ≈ 0.89 nats — seven times more than any other head. L0 H3, the *other*
sharp previous-token head, costs ≈ 0.04. Nearly identical attention patterns, an
order of magnitude apart in how much the model actually needs them. My read is
redundancy: L1 H1 carries a stronger version of the same signal one layer later, so
removing L0 H3 on its own barely registers. This is the part I'd point a reviewer
at first — the attention maps alone would have told me these two heads were
interchangeable, and they aren't close.

**Activation patching shows where L1 H1's information goes.** I take a clean context,
corrupt the character two positions before the end, and then patch the clean
residual-stream activations back in one site at a time, measuring how much of the
original prediction comes back. Through the embedding and block 0 the corrupted
information sits at its own position. Then block 1 moves it forward: recovery jumps
from ≈ 0.13 at the corrupted position to ≈ 0.75 at the prediction position, and the
later blocks carry it the rest of the way. Block 1 is where L1 H1 lives, so this is
the head's copy operation caught in the act.

![Activation patching recovery](assets/activation_patching.png)

**The prediction sharpens steadily with depth.** Reading each layer's residual stream
through the final unembedding (the logit lens), top-1 next-character accuracy climbs
0.06 → 0.12 → 0.21 → 0.34 → 0.51 across the four blocks, with the last block doing
the most. One wrinkle I didn't expect: the embedding-level logit-lens cross-entropy
is *worse* in this converged model than it was in an under-trained one, even though
embedding-level accuracy is unchanged. The embeddings specialize to feed the deeper
layers rather than to be read out directly, so the lens — which uses the final
unembedding — reads them less faithfully. A reminder that the logit lens is a lower
bound on what a layer knows, not a decoder of it.

Three alternative readings of the result — and the limits of each method — are at 
the end of the notebook.
[`notebooks/05_attention_analysis.ipynb`](notebooks/05_attention_analysis.ipynb).

---

## What it builds

The four analyses run on a model I build up one ingredient at a time in
[`notebooks/progression/`](notebooks/progression). To keep the comparison about the
*architecture* rather than about training length, I hold the three transformer
stages at a fixed 5,000-step budget:

| Stage | Params | Steps | Val loss (nats) |
|-------|-------:|------:|----------------:|
| Bigram baseline | 4,225 | 20,000 ¹ | 2.474 |
| + single attention head | 22,721 | 5,000 | 2.356 |
| + multi-head + feed-forward | 59,969 | 5,000 | 1.985 |
| **Full GPT** (4 blocks, residual + LayerNorm) | 209,729 | 5,000 | **1.798** |

The checkpoint I actually analyze is that same full architecture trained to
convergence — 30,000 steps, val ≈ 1.65 — because the analysis is more honest on a
model that's near its ceiling than on one still mid-descent. The 5k row above is the
controlled comparison point;
[`notebooks/progression/04_full_gpt.ipynb`](notebooks/progression/04_full_gpt.ipynb)
shows both.

¹ The bigram is a different kind of model with a fixed first-order floor, so I train
it to convergence (it plateaus by ~16k steps) instead of to the transformers' shared
budget. Every run uses seed 1337, batch size 16, learning rate 1e-3, AdamW, and
isolated fixed-seed evaluation; val loss is mean cross-entropy over held-out Tiny
Shakespeare. The committed 30k checkpoint reproduces a single-character-model
reference run to within ≈ 0.01 nats.

---

## Run it

```bash
pip install -e .                       # editable install; pulls torch, numpy, matplotlib
python train_gpt.py                    # reproduce assets/gpt.pth (~15 min on CPU, 30k steps)
jupyter lab notebooks/                 # run the progression, then 05_attention_analysis
```

The tests cover the package's correctness invariants — including the one the whole
patching analysis rests on, that splicing a run's own clean activations back in
changes nothing:

```bash
pytest                                 # 13 tests, runnable from any directory
```

I commit `assets/gpt.pth` (the trained 30k model) so the analysis notebook runs on
its own without retraining. Every other `*.pth` is gitignored; that one is a
deliberate carve-out. Delete it and run `python train_gpt.py` to rebuild it from
scratch.

---

## Repository layout

```
gpt/
├── gpt/                      reusable package
│   ├── data.py              char tokenizer, train/val split, reproducible batching
│   ├── layers.py            Head, MultiHeadAttention, FeedForward, Block + ActivationCache
│   ├── train.py             training loop, isolated eval, generation, seeding
│   └── analysis.py          attention stats, ablation, activation patching, logit lens
├── models/
│   ├── bigram.py            the baseline
│   └── gpt.py               GPT + GPTConfig, instrumented forward, checkpointing
├── notebooks/
│   ├── 05_attention_analysis.ipynb     ← the analysis (start here)
│   └── progression/         01 bigram → 02 single head → 03 multihead+FFN → 04 full GPT
├── assets/                  gpt.pth, loss history, generated figures
├── data/                    input.txt (Tiny Shakespeare)
├── tests/                   test_smoke.py, test_analysis.py
└── train_gpt.py
```

---

## Implementation notes

A few places where the decision that mattered wasn't the obvious one:

- **I loop over a `ModuleList` instead of `nn.Sequential` for the blocks.** The
  source stacks them in `Sequential`. I loop explicitly so the forward pass can
  capture each layer's attention and residual stream, route head-ablation flags to
  the right layer, and overwrite activations for patching. `Sequential` hides the
  loop and forbids exactly the per-layer access the whole analysis needs.
- **Evaluation draws from its own isolated RNG.** In the source, training and
  evaluation sample from the same global generator, so the trained weights quietly
  depend on how often you evaluate. I re-seed a separate generator inside
  `estimate_loss`, which decouples them — the loss curve is the same whether I
  evaluate every 250 steps or every 1000, and a chunked run, a single-shot run, and
  the notebooks all land on the same numbers.
- **Checkpoints describe themselves.** `GPT.save` stores the `GPTConfig` next to the
  weights and `from_pretrained` rebuilds the matching architecture. The most common
  way a checkpoint "loads but outputs garbage" is an architecture that drifted from
  the one that trained it; carrying the config closes that off.
- **Attention is scaled by `1/sqrt(head_size)`, not `1/sqrt(n_embd)`.** The variance
  of the query·key dot product grows with the *head* dimension, so that's what the
  scaling has to cancel. Get it wrong and nothing crashes — softmax just goes peaky
  at initialization and training stalls, which is the worst kind of bug.
- **Patching writes out of place, and the no-op case is tested.** A patch writes
  into a cloned tensor so a corrupted run can't overwrite the cached clean
  activations, and a test asserts that patching a run's own clean activations back
  in changes the output by nothing. Without that invariant every recovery number is
  suspect, so I pinned it down with a test rather than trusting it.
- **Recovery ratios are gated on a usable denominator.** Patching divides by the
  clean-minus-corrupt metric gap; I drop the examples where the corruption barely
  moved the prediction (about 150 of 192 survive) so the ratio doesn't blow up on
  noise.
- **The blocks are pre-norm.** LayerNorm sits inside each residual branch and leaves
  the skip path clean, so gradients reach the early layers — which is why notebook 03
  (multi-head + FFN with no residuals) is the harder model to optimize despite being
  shallower.

---

## Attribution

The architecture and training recipe follow Andrej Karpathy's *"Let's build GPT:
from scratch, in code, spelled out"* and
[nanoGPT](https://github.com/karpathy/nanoGPT); the corpus is Tiny Shakespeare. What
I added is the interpretability instrumentation (activation cache, head-ablation and
activation-patching APIs), the reproducibility engineering (isolated evaluation,
self-describing checkpoints), and the four-part analysis this README leads with.
