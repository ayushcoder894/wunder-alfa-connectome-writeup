# Wunder Fund Alfa Connectome: our solution

**Team:** Ayush Gupta ([@ayushcoder894](https://github.com/ayushcoder894)), TODO_NAME_2 (@TODO), TODO_NAME_3 (@TODO)
**Result:** public leaderboard 0.66938 (about rank 45 of 5,800). Our rule-clean variant scored 0.66924.
**Competition:** streaming prediction of two price-movement indicators from order books of two instruments, scored by a
global weighted Pearson correlation; inference on 1 CPU core in 60 minutes for 39.4 million rows.

This is a write-up, not a code release. Snippets are simplified sketches of the parts we think are unusual; they show the
idea, not our exact implementation.

## Contents

1. [The task in one paragraph](#1-the-task-in-one-paragraph)
2. [What the data turned out to be](#2-what-the-data-turned-out-to-be)
3. [Validation: the part we got wrong first](#3-validation-the-part-we-got-wrong-first)
4. [The base model](#4-the-base-model)
5. [Labels: teachers and look-ahead targets](#5-labels-teachers-and-look-ahead-targets)
6. [Adapting to the validation period](#6-adapting-to-the-validation-period)
7. [Unit 3: the hard market](#7-unit-3-the-hard-market)
8. [The final ensemble](#8-the-final-ensemble)
9. [Fitting it into one CPU core and 60 minutes](#9-fitting-it-into-one-cpu-core-and-60-minutes)
10. [What did not work](#10-what-did-not-work)
11. [After the deadline: why new models kept "fading"](#11-after-the-deadline-why-new-models-kept-fading)

---

## 1. The task in one paragraph

Each sequence is 20,000 consecutive moments of two instruments, `i0` and `i1`: 11 anonymised bid and 11 ask "price-like" and
"volume-like" columns per instrument, 4 trade-price and 4 trade-volume columns per instrument, and 8 additional columns
`a0`-`a7` (112 inputs). The first 99 rows are warm-up. On every later row the model must output two numbers, `t0` and `t1`,
"indicators of the future price movement of i0". Only a hidden subset of rows is scored. The metric is the weighted Pearson
correlation of prediction and target per target (weights = |target|, both clipped to [-2, 2]), pooled over all scored rows of
all sequences, averaged over the two targets. Train: 10,607 sequences; validation: 1,873 with the scored mask; test: 1,970,
streamed row by row to the submission. State must be reset between sequences.

## 2. What the data turned out to be

Most of our real progress came from understanding the data, not from model search.

### 2.1 Inputs are quantile-transformed

Every input column is the output of a `QuantileTransformer(output_distribution="normal")`: values are bounded, and a
sentinel `-5.199337 = Φ⁻¹(1e-7)` marks the floor. Price columns are not prices: within a tick unit they jump between a small,
fixed vocabulary of exact vectors shared by all sequences. We built a token table of those vectors (98.5 % coverage on
validation) and decoded an exact tick path for `i0`.

### 2.2 Four markets ("tick units")

Sequences come from four tick sizes (0.1254, 0.2413, 0.4685, 0.6569 in target units). The unit is identifiable from the
first 99 rows at 99.95 % (one error in 1,873) with a small gradient-boosted model on warm-up statistics. The units behave
very differently:

| Unit | Validation sequences | Typical WP | Share of metric weight |
|---|---|---|---|
| 0 | 21 % | 0.80-0.81 | about 0.21-0.28 |
| 1 | 19 % | 0.77-0.78 | about 0.12-0.16 |
| 2 | 23 % | 0.73-0.74 | about 0.08-0.11 |
| 3 (large tick) | 37 % | 0.44-0.45 | about 0.48-0.56 |

Unit 3 carries about half of the metric and is predicted at 0.44. Almost every decision we made was really a unit-3
decision.

### 2.3 The targets have an exact structure

- `t0 = F - ask`, `t1 = bid - F`, with one reference price `F` on the tick grid. Targets sit exactly on the per-unit tick grid.
- Targets change **only** on rows where i0's price columns change (0 exceptions), and `a6` resets **if and only if** a target
  changed (P = 1.000 both ways, all units).
- `F` is the end point of the next full-tick price run (exact on 96 % of unit-3 rows, 82 % of unit 2).
- When `F` lies beyond a quote, the far target is pinned: one target is `+k` ticks and the other exactly `-(s+2)` ticks, with
  `s` the spread code (unit 3: 99.97 %). So `t0 + t1 = -spread` fails exactly in those "beyond" states.
- In unit 3 about 94 % of scored rows are in one "rest" price code, where the book shape does not reveal on which side `F`
  sits. That is the core difficulty of unit 3.

### 2.4 The additional columns are clocks

`a0` is the log of the real time step between rows (Σ exp(3·a0) rebuilds wall-clock time); `a2`/`a3` are EMAs of it;
`a5`, `a6` are wall-clock timers since an event (`a6`: since the last target change; `a5`: since a class of i0 trades); `a7`
is recent event intensity. None of this gave a feature gain by itself (the recurrent model already reads them), but it
explained the data.

### 2.5 The scored mask is not random

Scored rows come in bursts (75 % of gaps between scored rows are 1 row), depend on the target state (in unit 3, spread-2
states are almost never scored, 3-tick states are over-sampled), and are much easier than unscored rows (unit 3: 0.43 vs
0.19 WP for the same model). We trained a small GRU on validation to estimate P(scored | inputs, targets) and used it as a
row weight in training (+0.0005 paired).

### 2.6 Sequences are blocks of whole trading sessions

Sequences are consecutive 20,000-row blocks of longer sessions (1,505 continuation links inside validation, 368 sessions,
only 41 sessions for unit 3). Carrying state from the previous block is worth +0.0002 at most, and the rules forbid joining
sequences at test time, so we never used links at inference. Their real importance was for validation (next section).

```python
# Sketch: linking validation blocks into sessions (used only for building honest folds).
# A block's last decoded book state matches the next block's first state exactly.
end_key   = {seq: hash(last_book_state(seq))  for seq in valid_seqs}
start_key = {seq: hash(first_book_state(seq)) for seq in valid_seqs}
by_start  = defaultdict(list)
for s, k in start_key.items():
    by_start[k].append(s)
links = {s: by_start[k][0] for s, k in end_key.items() if len(by_start[k]) == 1}
sessions = connected_components(links)        # each component = one trading session
```

## 3. Validation: the part we got wrong first

Three mistakes cost us weeks; fixing them was worth more than any model change.

1. **Random hold-outs leak.** A random 373-sequence hold-out overstated a validation fine-tune by 4-5x: +0.0027 on the random
   split, +0.0006 on a split that keeps whole chains apart. Our first fine-tuned package (estimated 0.665) scored 0.6585.
2. **Chain folds still leak.** Folds built from the obvious chain map put the same session on both sides for 96 % of unit-3
   sequences. We rebuilt the session map (it found 44 % more links) and moved every gate to **session folds**.
3. **Soft labels can leak through the teacher.** A cross-fitted teacher trained on 80 % of validation still saw most
   held-out sessions; arms that used it looked +0.005 better on unit 3 than they were. Teachers for gates must be train-only.

What held up:

- For joint (train + validation) packages, public ≈ full-validation score − 0.0274, stable within ±0.0015 over a dozen
  submissions. Train-only models transfer about 0.003 worse.
- Paired public deltas between related packages have SD ≈ 0.0006; public vs private for the same model differs with
  SD ≈ 0.009. We stopped reading public deltas under ~0.0015 as signal.
- Gains on unit 3 (half the metric) need two to three seeds: two fine-tunes of one parent that differ only by seed are
  0.0013 apart on unit 3.

```python
# Paired session bootstrap used for every gate (weights |y|, clipped as the metric).
def wp(p, y, w):
    p, y = np.clip(p, -2, 2), np.clip(y, -2, 2)
    mp, my = (w * p).sum() / w.sum(), (w * y).sum() / w.sum()
    return (w * (p - mp) * (y - my)).sum() / np.sqrt((w * (p - mp)**2).sum() * (w * (y - my)**2).sum())

def paired_boot(pa, pb, y, sess, n=300, rng=np.random.default_rng(0)):
    u, ix = np.unique(sess, return_inverse=True); d = []
    for _ in range(n):
        c = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[ix]   # resample whole sessions
        d.append(np.mean([wp(pa[:, k], y[:, k], np.abs(np.clip(y[:, k], -2, 2)) * c)
                          - wp(pb[:, k], y[:, k], np.abs(np.clip(y[:, k], -2, 2)) * c) for k in range(2)]))
    return np.percentile(d, [5, 95]), np.mean(np.array(d) > 0)
```

## 4. The base model

All shipped members are small recurrent networks trained on full 20,000-row sequences with truncated BPTT in 1,000-row
chunks. The ingredients that survived paired tests, in the order we found them (validation gains, one seed unless noted):

| Change | Gain | Note |
|---|---|---|
| LSTM instead of GRU at equal cost | +0.002 | 128 → 192 wide paid only once labels were cleaner |
| Self-supervised auxiliary heads (future decoded price moves at 5-200 rows) | +0.001 | training only |
| Random weight perturbation (N(0, (0.01·|w|)²) on alternate steps) | +0.0005 | SAM and heavier noise did not help |
| Piecewise-linear encoding (PLE) of each input, 8 bins | +0.0008 | factored, zero-initialised |
| Causal adaptive input normalisation (`ain`) | +0.0007 | 672 parameters, proven on 2 seeds |
| LSTM with projection (400 cells → 128) | +0.0007 | |
| Two streams: one per instrument, merged at the top | +0.0004 pooled, **+0.0019 unit 3** | used for unit 3 |

### 4.1 Loss

The loss is the metric itself on each chunk, plus a little MSE for scale:

```python
def wcorr(pred, y, rw):                      # differentiable weighted Pearson
    y = y.clamp(-2, 2); w = y.abs() * rw     # rw = P(scored | x, y) + floor
    W = w.sum()
    yc, pc = y - (w * y).sum() / W, pred - (w * pred).sum() / W
    return (w * yc * pc).sum() / torch.sqrt((w * yc**2).sum() * (w * pc**2).sum() + 1e-12)

loss = sum(1 - wcorr(out[:, k], y[:, k], rw) for k in range(2)) / 2 \
     + 0.1 * (y.clamp(-2, 2).abs() * (out - y.clamp(-2, 2))**2 * rw[:, None]).mean() \
     + distillation_terms + aux_terms
```

### 4.2 Piecewise-linear encoding, factored

PLE (Gorishniy et al., 2022) turns each scalar into a vector of "how far through each quantile bin". With 224 inputs
(values and first differences) × 8 bins the encoding is wide, so we project it to 64 extra LSTM inputs with a zero-initialised
matrix. At export the projection is folded into the LSTM's input weights per code, so inference only adds the rows of
bins that changed.

```python
class PLE(nn.Module):
    def __init__(self, edges, d_out=64):              # edges: [n_cols, n_bins + 1], train quantiles
        super().__init__()
        self.register_buffer("lo", edges[:, :-1]); self.register_buffer("hi", edges[:, 1:])
        self.proj = nn.Linear(edges.shape[0] * (edges.shape[1] - 1), d_out, bias=False)
        nn.init.zeros_(self.proj.weight)              # starts identical to the plain LSTM
    def forward(self, x):                             # x: [B, T, n_cols]
        z = ((x[..., None] - self.lo) / (self.hi - self.lo)).clamp(0, 1)
        return self.proj(z.flatten(-2))
```

### 4.3 Causal adaptive input normalisation

Train and validation differ strongly (a classifier separates them at AUC 0.99). `ain` normalises every column by running
statistics of the sequence itself, mixed with the global normalisation by a few learned scalars per column. It is causal (only
past rows) and cheap enough to share between all models at inference.

```python
class CausalAdaptiveNorm(nn.Module):
    """x_t -> mix of (global z-score) and (z-score against this sequence's running mean / var); learned per column."""
    def __init__(self, n, alphas=(1/64, 1/1024)):
        super().__init__()
        self.alphas = alphas
        self.mix = nn.Parameter(torch.zeros(len(alphas), n))   # 0 = plain global normalisation at init
    def forward(self, x, state):                               # state: running means / vars per alpha
        out = x
        for i, a in enumerate(self.alphas):
            m, v = state[i]
            m = (1 - a) * m + a * x; v = (1 - a) * v + a * (x - m) ** 2
            state[i] = (m, v)
            out = out + self.mix[i] * ((x - m) / torch.sqrt(v + 1e-3) - x)
        return out, state
```
(The real version runs the recursion over a chunk with a scan and has one more cross-column term; the idea is the same.)

### 4.4 Two streams

i0 and i1 columns go through separate LSTM stacks; a top LSTM reads both. The partner instrument carries little
information on its own (scrambling i1's prices costs only 0.009), but separating it stopped it from interfering with i0's
representation, and on unit 3 that was worth +0.0019.

## 5. Labels: teachers and look-ahead targets

Raw targets are noisy (a target is constant between price events and then jumps). Distillation gave clean, repeatable gains:

- **Causal teacher ensemble** (`teachmix`, 9 models averaged): the student pulls its main head toward the teacher with the
  same weighted Pearson.
- **Look-ahead teachers:** the same LSTM, but given summaries of the next 20 or 50 rows of the decoded price path. They score
  0.742 / 0.800 on validation (causal models: 0.69). Labels from 20-50 rows of look-ahead helped; 100-500 rows or event-time
  look-ahead hurt (the teacher sees too much the student cannot know).
- **A big teacher:** two wide LSTMs (16M and 33M parameters, 6-14 hours of CPU each, unshippable) averaged as `xlmix`
  (0.6946 validation). Students distilled from it gained +0.0008, and the same labels in the validation fine-tune gave
  +0.0006 on public.
- Soft label for fine-tunes: `0.2 · y + 0.8 · teacher` worked better than either alone.

## 6. Adapting to the validation period

The organisers confirmed that public and private test data lie **after** validation in time, and validation lies after
train. So the validation period is the most test-like data we have.

1. **Joint training.** Every shipped parent is trained on train + validation. Train-only models transferred to public
   about 0.002-0.003 worse than their joint twins.
2. **Validation fine-tunes per unit.** Parents are fine-tuned on validation sequences of one unit group, with warm-up +
   cosine LR (peak 2e-4, 6 epochs), loss weight 0.1 on unscored rows, and soft labels. A constant LR collapsed the models
   (−0.04): they had ended pre-training at ~1e-6.
3. **Weight soups.** Three to six fine-tunes of the same parent, differing only by seed, are averaged in weight space,
   then mixed 0.75 / 0.25 with the parent (a WiSE-FT-style hedge). Independent-seed soups failed; same-parent soups work.

```python
def soup(parent, finetunes, alpha=0.75):
    sd = {}
    for k, v in parent.items():
        if v.is_floating_point():
            ft = torch.stack([f[k].double() for f in finetunes]).mean(0)
            sd[k] = (alpha * ft + (1 - alpha) * v.double()).to(v.dtype)
        else:
            sd[k] = v.clone()
    return sd
```

Public effects: units 0-2 fine-tune +0.0010 (J6), with the big-teacher labels +0.0006, the first unit-3 fine-tune +0.0004.

## 7. Unit 3: the hard market

Unit 3 is information-limited: every model family makes the same errors there (residual correlation 0.997), a capacity jump
from 0.8M to 33M parameters adds +0.0003, and hand-built event features, Markov models and boosted trees all stay below the
LSTM. Two things worked.

### 7.1 Bid/ask mirror augmentation, done properly

Mirroring the book (swap bid and ask, negate the direction, swap `t0` and `t1`) doubles unit-3 data. The naive column swap
does not work: the volume columns are a mixed, quantile-transformed encoding, and a model trained on naively swapped rows
predicts direction backwards on real data (AUC 0.398). We decoded the representation first:

- price codes → their mirror codes through the shared codebook;
- volume columns: decoded into 10 price levels per side plus one "latest update" column, by calibrating each column's
  monotone curve so that one-tick moves act linearly, then finding the level-shift maps (an improving move sends level `j`
  to `j+1`);
- trade codes → their partner states.

Rows mirrored through the decoder keep their meaning (a model trained only on mirrored rows reaches AUC 0.862 on real rows
vs 0.863 trained on real rows). Mirrored copies are used in pre-training a unit-3 parent and inside the unit-3 validation
fine-tune.

```python
def mirror_u3(x, y, tables):
    xm = x.copy()
    xm[:, PRICE_I0] = tables.price_mirror[code_of(x[:, PRICE_I0])]      # exact partner code vectors
    bid, ask = decode_depth(x[:, VOL_BID]), decode_depth(x[:, VOL_ASK])  # [T, 11] levels per side
    xm[:, VOL_BID], xm[:, VOL_ASK] = encode_side(ask), encode_side(bid)  # swap sides in decoded space
    xm[:, TRADE_I0] = tables.trade_mirror[trade_code(x[:, TRADE_I0])]
    ym = y[:, ::-1].copy()                                               # t0 <-> t1
    return xm, ym
```

Public: mirror inside the unit-3 fine-tune +0.0002; decoder-based mirror instead of the first version +0.0001; members
from a mirror-pretrained parent at least neutral and more diverse.

### 7.2 Diversity, not size

The unit-3 blend that worked is three members with different parents and different data views at equal weights (a
depth-decoder-mirror fine-tune, a mirror fine-tune of a second seed, and a soup of fine-tunes of a mirror-pretrained
parent). Seed and parent diversity was the only steady unit-3 gain.

## 8. The final ensemble

A small classifier on the 99 warm-up rows picks the unit; each unit gets its own blend:

| Route | Members (weights) |
|---|---|
| Unit 0 | G0 0.45 · RGRU 0.15 · tsall 0.40 |
| Unit 1 | G0 0.40 · RGRU 0.15 · tsall 0.45 |
| Unit 2 | G0 0.20 · RGRU 0.30 · tsall 0.50 |
| Unit 3 | G1d ⅓ · X ⅓ · GM ⅓ |
| Doubtful unit (≈0.2 % of sequences) | G0 only |

- **G0:** joint LSTMP-400 with `ain` and PLE, validation fine-tuned with big-teacher labels, refitted on units 0-2.
- **RGRU:** a second architecture from a separate pipeline: a 2×256 GRU on raw inputs, int8. It was the only partner that
  added a large, nested-validated gain on units 0-2 (+0.0017 validation, +0.0006 public): **pipeline diversity beat
  architecture diversity.**
- **tsall:** a two-stream model fine-tuned on validation.
- **G1d / X / GM:** the unit-3 members of section 7.

About 15 small recurrent models in total, ~19 MB zipped.

**A rule note.** One of our best-scoring variants chose a lighter unit-3 route on every 4th unit-3 sequence with a counter
that runs across sequences, which conflicts with "no state across sequences". The rule-clean variant picks that light route
from a hash of the sequence's own warm-up rows (or not at all) and scored 0.66924:

```python
def light_route(warmup_rows):                       # depends only on this sequence
    h = zlib.crc32(np.ascontiguousarray(warmup_rows[:99], dtype=np.float32).tobytes())
    return h % 4 == 0
```

## 9. Fitting it into one CPU core and 60 minutes

39.4 million rows in 60 minutes is about 91 µs per row for everything, including Python's per-row callback. Speed decided
how many members we could ship, so inference engineering was a large part of the work.

- **A C kernel** compiled at start-up (gcc, AVX2 / FMA / F16C, a NumPy fallback if compilation fails), with a self-timing guard.
- **Weights in fp16 and int8.** Recurrent matrices in int8 with per-output scales and int16 activations; int8 matched fp16
  scores exactly on our tests. Inference is bound by weight bytes per row, so halving bytes halves time.
- **Incremental input work.** Only ~10 of 224 inputs change per row; the PLE contribution is updated only for changed codes,
  and the adaptive-normalisation core is computed once per row and shared by all models.
- **Packed int8 tiles.** The last speed step (−11 %, bit-identical) packs two int8 weights per 16-bit lane so one
  `vpmaddubsw` serves 32 weights with no sign-extension instructions:

```c
/* low byte = w_a + 128 (unsigned), high byte = w_b (signed); x = int8 activations broadcast as 16-bit pairs */
__m256i p   = _mm256_maddubs_epi16(packed_w, x_pairs);   /* (w_a + 128) * x + ... (unsigned x signed)  */
__m256i s   = _mm256_madd_epi16(p, ones16);              /* widen to int32                            */
/* the +128 offset is removed once per row with a precomputed 128 * sum(x) term; w_b comes from a srai */
```

- **A watchdog.** Per-route µs/row are measured during the run; if the projected total exceeds 59 minutes, the remaining
  sequences switch to a lighter policy. Several early submissions silently fell back to a small model because the start-up
  timing guard was too tight on a noisy host; after that we always kept a 15 % margin and timed packages in Docker with
  one pinned core (the scorer ran about 1.4x slower than Docker).

## 10. What did not work

Each of these was tested against a matched control, most on the full recipe:

- **Architectures:** local attention, TCN, LRU / diagonal state-space, minGRU / minLSTM stacks, sLSTM, LayerNorm-LSTM,
  CNNs on the book "image", event-clocked layers, memory cores, multi-head averaging, per-unit specialists. Every one led at
  epoch 0 and faded to a seed or worse by epochs 2-4 (see section 11 for why).
- **Online learning,** in every form we could make legal: per-sequence recalibration from labels that become known later
  (once a label is causally known it carries nothing), weights carried across sequences (harmful at every learning rate).
- **Features:** order-flow, decoded book levels as inputs, Markov / n-gram event grammars, partner lead-lag, regime
  labels, reconstructed EMA signals. The LSTM already extracts them.
- **Post-processing:** smoothing within target segments (−0.065), per-unit / per-regime rescaling, stackers, gradient-boosted
  trees on hand features (0.38 vs 0.44 on unit 3).
- **Labels:** event-time look-ahead teachers, 100-500-row teachers, bootstrapped TD targets, multi-horizon label averaging.
- **Data:** re-weighting train sequences toward the validation regime (unit-3 train is 64 % "trending" sessions, validation
  96 % "bounce"), adding other units' data to unit 3, time-warp augmentation, target clipping.

## 11. After the deadline: why new models kept "fading"

Looking per unit instead of pooled explained the pattern:

| Big LSTM (16M) by epoch | Unit 0 | Unit 3 | Pooled |
|---|---|---|---|
| 1 | 0.8042 | 0.4375 | 0.6893 |
| 5 | 0.8079 | 0.4455 | 0.6937 |
| 8 | 0.8095 | 0.4262 | 0.6883 |

- More capacity and more training keep helping units 0-2; on unit 3 they turn into memorisation after a point (the same
  model's score on unit-3 **train** sequences rises from 0.515 to 0.606 between epochs 5 and 8 while validation falls).
  Unit 3 is half the pooled metric, so a faster learner reaches the turning point sooner and its pooled lead disappears.
- The old causal teacher helps early in training and becomes a drag late: removing it after the first epoch gave units 0-1
  **+0.0011 / +0.0007**, reproduced on a second seed (+0.0013 / +0.0006).
- Blends restricted to weights in [0, 1] missed that early checkpoints and the old teacher deserve **negative** weights. A
  nested (honest) per-unit committee with free weights reaches 0.6965 against 0.6946 for our best teacher.
- A causal transformer teacher (8 layers, d=512, 2,000-row context) beat the big LSTM at every matched epoch on every unit
  (unit 3 +0.003 at epoch 3), but did not yet add to the teacher blend when we stopped it.

```python
# Free-weight per-unit committee, cross-fitted by session (the honest version nests the source selection too).
def committee(sources, y, unit, fold):
    out = np.zeros_like(y)
    for u in range(4):
        for k in range(2):
            for f in range(5):
                tr, te = (unit == u) & (fold != f), (unit == u) & (fold == f)
                A = np.c_[[s[tr, k] for s in sources].T, np.ones(tr.sum())]
                w = np.abs(np.clip(y[tr, k], -2, 2))
                b = np.linalg.lstsq(A * np.sqrt(w)[:, None], y[tr, k] * np.sqrt(w), rcond=None)[0]   # weights may be < 0
                out[te, k] = np.c_[[s[te, k] for s in sources].T, np.ones(te.sum())] @ b
    return out
```

---

*Numbers are validation weighted Pearson on all 1,873 sequences unless marked public. Snippets are simplified.*
