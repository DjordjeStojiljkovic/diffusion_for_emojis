# Emoji diffusion

A small DDPM trained on 64x64 emoji, in three iterations. Each one is additive —
the earlier models and their checkpoints keep working.

1. **Unconditional** — `train.ipynb`, `unet.py`. Draws *an* emoji.
2. **Class-conditional** — `train_cond.ipynb`, `unet_cond.py`. Draws an emoji of
   a requested class ("give me a `food-drink`") via a learned `nn.Embedding`.
3. **Text-conditional** — `train_text.ipynb`, `unet_text.py`. Draws an emoji
   from *any* text, via a frozen CLIP embedding of the emoji name.

## Layout

| file | what it is |
| --- | --- |
| `dataloader.py` | `EmojiDataset` (images only) and `ConditionedEmojiDataset` (images + class index from a metadata CSV) |
| `unet.py` | the noise-predicting U-Net, `model(x, t)` |
| `unet_cond.py` | `ConditionedUNet(UNet)` — adds `nn.Embedding` for the class, `model(x, t, y)` |
| `unet_text.py` | `TextConditionedUNet(UNet)` — conditions on frozen CLIP text embeddings; `y` is either bank indices or raw embeddings |
| `emoji_text.py` | builds/caches the CLIP name bank, matches free prompts to real names, and scores that retrieval |
| `emoji_llm.py` | small local LLM: sentence → list of emoji-sized phrases |
| `download_llm.py` | fetches that LLM into `models/` |
| `diffusion.py` | linear-beta DDPM: forward noising, loss, ancestral sampling, DDIM sampling, classifier-free guidance |
| `trainer.py` | the training loop with EMA, and batched sample generation |
| `metrics.py` | FID, KID, CMMD, precision/recall, CLIP score, CLIP zero-shot accuracy, nearest-neighbour memorization, diversity |
| `runlog.py` | per-run output directories and the figures (grids, loss curves, comparison table) |
| `make_subsets.py` | carves the two curated subsets out of openmoji |
| `train.py` | headless version of the original unconditional run |

Every module has a `__main__` smoke test — `python metrics.py`, `python
trainer.py`, etc. — that runs in seconds and checks its own contract.

## How the conditioning works

Both conditional models sum a conditioning vector into the timestep embedding:

```python
emb = self.time_mlp(t) + self.label_emb(y)       # unet_cond.py: learned per class
emb = self.time_mlp(t) + self.text_proj(bank[y]) # unet_text.py: frozen CLIP text
```

Every residual block already consumes that embedding as a per-channel bias, so
the signal reaches the whole network without touching any block. In both, the
conditioning table has one extra row — a null class / empty prompt that training
drops onto 10% of the time, which is what makes classifier-free guidance
possible without training a second model.

**Why CLIP text rather than a name table.** Emoji names are unique: exactly one
image per name. A learned `nn.Embedding` over 500 names therefore has nothing to
generalise from and can only memorise. CLIP places "grinning face" near "winking
face", so the model learns face-ness from every smiley at once — and because the
conditioning space is shared with CLIP, sampling accepts *any* prompt, not only
names seen in training.

**CLIP is never trained.** The bank is computed once, cached, and held as a
buffer (so checkpoints are self-contained and no gradient reaches it). The only
new trained component is `text_proj`, ~198k parameters, 1.2% of the model. The
text encoder isn't even loaded during training.

**The bank cache is fingerprinted, for a reason.** A cache keyed only on the
name list will happily hand back vectors from a completely different encoder,
and every downstream number still *looks* fine — this silently cost a full
30k-step training run once. `emoji_text.build_bank` now records which encoder
and prompt template produced a bank and rebuilds on any mismatch, calls
`check_encoder` (is `cos(dog, puppy) > cos(dog, rocket)`?) before embedding, and
rejects a finished bank whose mean pairwise cosine looks like noise. The
playground cell adds the matching runtime check: a checkpoint carries the bank
it trained on, so it warns if that disagrees with the bank in memory.

## Text → emoji, without pipelining it

Two independent pieces, deliberately not yet wired end to end:

- **CLIP matching** (`emoji_text.match`) turns free text into a real emoji name.
  Because it picks from a fixed list it cannot invent a name that doesn't exist.
  `emoji_text.retrieval_accuracy` scores this stage on its own, using openmoji's
  `description` keywords as a free labelled test set.
- **LLM decomposition** (`emoji_llm.decompose`) splits a sentence into several
  emoji-sized phrases — the one thing retrieval cannot do, since a single CLIP
  embedding of a whole sentence blends into one concept. Its output is always
  run back through CLIP matching, so hallucination can't reach the model.

## Data

[openmoji](https://openmoji.org/) (CC BY-SA 4.0), 1321 images at 64x64 in 8
groups, under `datasets/openmoji/`. `make_subsets.py` builds two curated
subsets of common, recognisable emoji:

| subset | images | conditioned on |
| --- | --- | --- |
| `datasets/openmoji_10` | 10 | 2 groups (smileys-emotion, food-drink) |
| `datasets/openmoji_200` | 200 | 8 groups, 25 each |
| `datasets/openmoji_500` | 500 | 500 names, one image each |

The 200 picks are hand-listed — a random draw of openmoji is mostly keycaps,
clock faces and Japanese buttons. The 500 subset grows those 200 by filling
round-robin across groups and subgroups (so 66 mammals can't crowd out every
bird), after dropping the subgroups in `EXCLUDE_SUBGROUPS`: keycaps, arrows,
maths, clock faces, colour swatches and the 25 near-identical `family: ...`
variants. Those are exactly the images that teach a text-conditioned model that
very different prompts map to nearly the same picture.

The subsets are nested — 10 ⊂ 200 ⊂ 500, asserted in the script — so runs stay
comparable.

## Running it

```bash
pip install -r requirements.txt      # torchvision + open_clip; transformers only for the LLM
python make_subsets.py               # writes datasets/openmoji_10, _200 and _500
python download_llm.py               # optional: ~3 GB into models/, for emoji_llm.py

jupyter lab train_cond.ipynb         # class-conditional: 10-image and 200-image runs
jupyter lab train_text.ipynb         # text-conditional: the 500-image run
```

Both notebooks have a `QUICK = True` flag that runs every cell end-to-end in a
couple of minutes for checking plumbing, and both write everything to disk as
they go:

```
runs/<name>/config.json       hyperparameters, cumulative wall-clock, final loss
runs/<name>/losses.csv        per-step loss, extended across resumed rounds
runs/<name>/loss.png          loss curve
runs/<name>/metrics.json      every metric
runs/<name>/ckpt.pt           rolling: raw + EMA weights + optimiser state
runs/<name>/ckpt_step<N>.pt   immutable snapshot per training round
runs/<name>/samples/*.png     previews during training, final per-class grids,
                              guidance sweep, nearest-neighbour figure
results/summary.md            cross-run table + config + figure index
results/metrics_all.json      the same, machine-readable
```

Training is resumable: `trainer.train` takes `ema`, `start_step`,
`optimizer_state` and `losses`, so §12 of `train_text.ipynb` continues a run
rather than restarting it — step numbering and loss history carry on, and each
round leaves both an updated `ckpt.pt` (with Adam's moments, so the next round
picks up exactly where it stopped) and a snapshot that later rounds never
overwrite.

`results/summary.md` is the file to open when writing the presentation.

## Reading the metrics

| metric | question | direction |
| --- | --- | --- |
| FID | do samples match the data distribution? | lower |
| KID | same, unbiased at small sample counts | lower |
| CMMD | same, on CLIP features — reliable at these sample counts | lower |
| precision / recall | fidelity vs coverage, separated | higher |
| CLIP score | does the sample match its class prompt? | higher |
| CLIP accuracy | does CLIP classify the sample as the requested class? | higher |
| NN similarity | is the sample just a copy of a training image? | **lower** |
| diversity | mean pairwise distance within a class | higher |

The text run adds two levels of conditioning score, because "which of 500 names
is this?" and "does this look like an emoji at all?" are different questions:

| metric | question |
| --- | --- |
| `name_top1` / `name_top5` | does CLIP pick the right name out of all 500? (chance 0.2%) |
| `name_clip_score` | how well does the sample match its own name prompt? |
| `retrieval_text_only` | accuracy of prompt → name *before* any image is drawn |

Caveats that belong in any writeup:

- **FID is unreliable here.** It needs thousands of samples; these runs have 10
  and 200. It is reported because people expect it, but KID, CMMD and the CLIP
  metrics carry the argument. FID values are also computed from torchvision's
  InceptionV3, not the original TF weights, so they are comparable only with
  each other — never with published numbers.
- **CLIP is not a perfect judge of 64x64 pictograms.** Every conditioning metric
  is also computed on the real data (`clip_accuracy_real`, `clip_score_real`);
  that is the ceiling to read the generated score against.
- **Run A is supposed to memorise.** High nearest-neighbour similarity with 10
  training images is the expected outcome, not a bug — it is why the metric is
  there.

## Compute notes

DDIM (`Diffusion.ddim_sample`) is used for previews and metric batches: ~10x
fewer network calls than the 1000-step ancestral sampler for a small quality
cost. The final "hero" figures use full ancestral sampling.

Measured: run A (4000 steps, batch 10) took **8.4 min on a GPU**. On CPU a step
costs ~1.8 s at batch 32, so the larger runs are GPU jobs — expect **1–2 hours**
for run B (20k steps at batch 32) and **2–3 hours** for the text run (30k steps),
plus ~10 min for the final figures, metrics and the first-time CLIP/Inception
weight downloads.

Text conditioning costs nothing extra per step: the bank is precomputed, so the
step is a tensor index plus one ~200k-parameter projection.

The notebook picks the device automatically and has a `QUICK` flag that runs
every cell end-to-end in about two minutes for checking plumbing. Weights are
rewritten to `runs/<name>/ckpt.pt` every `CHECKPOINT_EVERY` steps, so an
interrupted run loses at most that many steps; `load_run` in §2 restores a
finished run without retraining it.
