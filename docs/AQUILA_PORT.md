# Porting the reservoir to QuEra Aquila (analog Rydberg array)

Status: Gates 0-5 closed, digital-arm fixture frozen. Gate 3's MSE result looked negative
enough to stop on, but a preliminary FID check reversed that -- MSE and FID dissociate for
the Rydberg arm exactly as `QRC_FUSION_FID_BIAS_AUDIT.md` documented for the digital arm.
Gate 4's sweep found `a/R_b=1.0` (a=9.757um, vs. the original 8um/0.82) improves both val_mse
and FID over the Gate 0-3 baseline. **Gate 5 then reversed that again**: the FID gain was
measured on exact (infinite-shot) statevector features; at the hardware shot ceiling
(`shots=1000`) it does not survive -- FID goes from 85.40 (exact) to 93.18, worse than
classical (92.31). A variance-matched Gaussian-noise control shows the degradation is generic
noise magnitude, not a structural property of projective measurement. **Gate 6 (full-scale
`samples=10000` FID/PRDC, 3 data seeds) is running** -- protocol, cost and what it can settle
are in its section below; its results table is filled in per cell as the queue drains. This
document is updated as later gates land; it is not a design doc written up front.

## Digital-arm regression fixture

`tests/_build_digital_fixture.py` (a one-off generator, not a test) ran the exact
`fashionmnist`/`seed=0`/`draw=0`/`n_train=500` cell on this machine and saved
`tests/fixtures/digital_arm_fixture.npz`: `qrc_features` output and every arm's fitted Ridge
coefficients. `tests/test_digital_arm_regression.py` reruns the identical computation and
asserts bit-for-bit equality against that fixture -- protecting against a *logic* regression
from `reservoir=` being added to `qrc_fusion_fair_core.py`, not claiming cross-machine
reproducibility. Determinism on this machine was checked directly (two independent runs
produced identical `repr()` output to full float64 precision) before trusting the fixture.

Measured once, directly, as the cross-machine sanity check: this machine's own numbers
differ from `results/qrc_fusion_fair/supervised_main.parquet` (produced on a different
GPU/torch build) by a small, consistent amount --

```
              committed (other machine)   this machine        diff
ridge_classical    0.5636011098413176   0.5636005596402978   -5.50e-07
ridge_qrc          0.5817008392756552   0.5817003376932807   -5.02e-07
ridge_random       0.5678036071177155   0.5678031493229592   -4.58e-07
ridge_interaction  0.5367942809311216   0.5367936834355703   -5.97e-07
```

Same sign, same order of magnitude, across all four arms -- consistent with a BLAS/cuDNN
version difference in float32 matmul reduction order, not a logic difference. This
comparison is not re-run automatically (nothing guarantees the other machine's environment
still exists to compare against); it is recorded here as a one-time measurement.

## Wiring `reservoir=` into the fair-comparison pipeline

Per the task's rule, the only change to `qrc_fusion_fair_core.py` is a `reservoir='digital'|
'rydberg'` parameter on `qrc_features` and `rollout`, dispatching to
`quera.features.rydberg_features`/`ReservoirParams` on the `'rydberg'` branch; the
`'digital'` branch (the default) is untouched code, confirmed by the fixture test above
passing unmodified after the edit. `rollout`'s `reservoir='rydberg'` path rejects
`reset_every_step=False` outright (`ValueError`) rather than accepting it silently -- the
Rydberg map has no persistent state to *not* reset, so that flag combination would promise
behavior that isn't implemented.

**The Rydberg map has no `draw`.** The digital arm's `draw` selects a random circuit from an
ensemble of reservoir unitaries (5 draws in the published protocol); the Rydberg Hamiltonian
is a fixed physical operating point (`ReservoirParams`), not a random family, so there is
nothing to draw from. Rather than inventing a fake ensemble, `qrc_fusion_fair_supervised.cell`
tags every row of a Rydberg cell `unitary_draw=-1`, reusing the convention
`ridge_classical`/`ridge_interaction` already use for draw-independent arms
(`qrc_fusion_fair_generation.py --draw` already defaults to -1 for exactly this reason) --
not a new sentinel. **Consequence for the paired statistics, stated up front so Gate 3's
number isn't misread**: `ridge_qrc` normally gets 15 cells (3 seeds x 5 draws) sharing only
3 independent classical baselines -- the audit's own "Ressalva de potência" already flags
the resulting df inflation. The Rydberg arm gets exactly one cell per (seed, n_train): 3
seeds means 3 independent pairs, not 15 correlated ones. A p-value from n=3 is not
comparable to the audit's qrc-vs-classical tables; report per-seed deltas and the mean, not
a headline p-value, for the Rydberg contrast.

`quera/encoding.py`'s `EncodingAffine` (`h = clip((raw-mean)/(k*std)+0.5, 0, 1)`, `k=4`
by default) is fit fresh inside each `cell()` call, on that cell's own training `(x, t)`
pairs only -- consistent with "fit on train, reuse unchanged at val/test" without needing a
separate persisted artifact across cells (each cell already refits the classical/random/
interaction readouts from scratch too, so this matches the existing per-cell-fit design
rather than introducing a new one).

### Performance fix (blocking, done before Gate 3)

The original `emulator._hamiltonian_matvec` issued `n_sites` (12) sequential
`torch.gather` calls per RK4 stage -- launch-bound on GPU, not compute-bound: a
`batch=2048` production call took 267s, and a `batch=1` call took 0.8s, i.e. cost was
*not* amortizing over the batch at all. Fixed by flattening all 12 flip-index tables into
one `(n_sites*dim,)` array and issuing a single batched `torch.gather` per stage (reshape
+ sum instead of a Python loop). Re-measured:

```
batch=1      0.803s   (803.0 ms/sample)
batch=256   15.518s   (60.6 ms/sample)
batch=2048 121.817s   (59.5 ms/sample)
```

Per-sample cost is now flat from `batch=256` to `batch=2048` (60.6ms vs 59.5ms) -- the fix
worked: cost now scales with genuine compute, not kernel-launch overhead. `batch=2048`
dropped from 267s to 122s (2.2x). All correctness tests (`tests/test_emulator.py`,
`tests/test_shot_convergence.py`) were re-run and still pass after this change; Gate 1 was
not re-run in full (same formula, only the batching of an already-tested matvec changed),
but the closed-form drive-factorization test exercises the same code path at full precision
and still passes.

## Gate 3 / preliminary Gate 6: MSE says no, FID says otherwise -- corrected

**This section originally concluded "no" from MSE alone and stopped, per Gate 3's literal
wording. That was wrong, and the mistake is worth stating plainly**: this exact project's
own `QRC_FUSION_FID_BIAS_AUDIT.md` documents MSE and FID *dissociating* for the digital arm
three separate times (s1, s5c/5d, and the interaction-vs-qrc contrast) -- a supervised-MSE
ranking predicting nothing about generation quality is this project's single best-established
empirical pattern, and Gate 3's MSE-only stopping condition was applied without checking
whether the same dissociation was happening again. It was. A quick, reduced-scale
`qrc_fusion_fair_generation.py --reservoir rydberg` run (below) reverses the conclusion:
**Gates 4-6 should not have been skipped.** They still haven't been run (this was one
preliminary cell, not the sweep), but the premise for skipping them was wrong.

### The MSE result that (wrongly, on its own) looked conclusive

At the task spec's own default operating point (`a=8um`, `Omega=6.283 rad/us`,
`S=9.0 rad/us`, `V=4`, `ramp_us=50ns`), `fashionmnist`, `n_train=500`, `seed=0`:

```
                    val_mse   test_mse   (delta test_mse vs classical)
ridge_classical      0.5410     0.5636    --
ridge_qrc (rydberg)  0.6483     0.6819    +0.1183
ridge_random         0.5430     0.5650    +0.0014
ridge_interaction    0.5198     0.5368    -0.0268
```

The Rydberg arm is not just "no better than classical" (the digital arm's own result at
this `n_train`, see `QRC_FUSION_FID_BIAS_AUDIT.md` s5c: `+0.00101`, a null) -- it is
**worse than a same-width (312-dim) random tanh projection control** by two orders of
magnitude in effect size (`+0.1183` vs `+0.0014`). This was not accepted at face value;
before writing it up, three checks ruled out an implementation bug:

1. **Feature statistics are well-behaved and consistent between splits**: `n_floored=0`
  (no dead columns, unlike the digital arm's 12-30 -- see `QRC_FUSION_FID_BIAS_AUDIT.md`
  s2b), and per-feature std/mean ranges match closely between train and val
  (`std in [0.065, 0.158]` both splits, `mean` ranges overlapping). A caching or
  train/val-inconsistency bug would show up here; it doesn't.
2. **The failure mode is a genuine train/val generalization gap**, not a fitting artifact:
  `train_mse=0.353` (much *better* than classical) vs `val_mse=0.648`, `test_mse=0.682`.
  Sweeping lambda past the value `fit_readout` selected (100) makes it monotonically
  *worse* (`0.648 -> 0.679` at 1e3 `-> 0.841` at 1e4 `-> 0.970` at 1e7, converging to the
  predict-near-zero floor) -- confirming 100 is already the grid's actual optimum, not a
  search failure, and that no amount of additional L2 regularization rescues this
  combination of features.
3. **Isolated by probe time**: each of the 4 `V_SLICES` probe-time blocks (78 features)
  used *alone* gives `val_mse` in `0.561-0.577` -- a mild negative, the same order of
  magnitude as the digital arm's own `n_train=500` result, not a red flag on its own. The
  degradation to `0.648` appears specifically when **combining** all 4 into 312 features.
  Since the `random` control is the same width (312) and does *not* show this
  degradation, the cause is the **correlation structure across probe times** (each is a
  different-duration snapshot of related dynamics driven by the same `x_t`, so the 4
  blocks are far more mutually correlated than 312 i.i.d. random features), not
  dimensionality alone -- a multicollinearity effect ridge regularization does not fully
  absorb at `n_train=500`.

This is a real, physically-interpretable negative in MSE: at this operating point the
`V_SLICES=4` probe times are redundant enough with each other that stacking them hurts a
linear readout more than a same-width random control would. **But `QRC_FUSION_FID_BIAS_AUDIT.md`
is exactly the document that should have stopped me from treating this as decisive** -- it
found the published digital-arm MSE negative was "in largest part a variance-floor
artifact" and that even the corrected, genuinely-null MSE result at `n_train=500` coexisted
with the QRC arm *winning* FID against classical on both datasets (s5d). MSE ranking a
reservoir arm behind a random control is unusual and worth taking seriously, but it is not,
on its own, in this project, evidence about generation quality.

### The FID check that reverses the conclusion

Ran `qrc_fusion_fair_generation.py --reservoir rydberg --method qrc` at reduced scale
(`samples=500`, not the audit's `10000` -- so these FIDs are elevated/noisier than any
published number and only the *relative* ordering within this table is meaningful, the
same caveat the existing code already applies to reduced-`n_gen` BreastMNIST cells) against
freshly-run `classical`/`random`/`interaction` cells at the identical `n_train=500`,
`seed=0`, `samples=500` settings:

```
method       reservoir   fid       diversity  precision  recall  density  coverage  degenerate
classical    digital     92.311    0.823      0.692      0.418   0.368    0.288     False
interaction  digital     88.456    0.991      0.670      0.516   0.426    0.352     False
qrc          rydberg     90.469    0.723      0.748      0.513   0.481    0.336     False
random       digital     103.504   0.633      0.670      0.371   0.359    0.272     False
```

**Rydberg beats classical (90.47 vs 92.31) and beats random by a wide margin (90.47 vs
103.50, comparable in size to the digital arm's own documented "beats random by 17-50 FID"
finding, s5d "O que sustenta um resultado QRC positivo"), and is within ~2 FID of
interaction** -- the same qualitative pattern the audit found for the *digital* reservoir:
loses or ties on structured-classical-control, but clearly beats a dimension-matched random
control, meaning the 312 features encode real, extractable structure rather than just
adding capacity. `degenerate=False` and `finite=True` for every cell (the guard rail that
would have caught the exact failure mode the original published study had); rydberg's
`precision` (0.748) and `density` (0.481) are the *best* of all four arms, `recall` (0.513)
second only to interaction's 0.516 -- not a FID win riding on one degenerate metric.

**This does not mean "the Rydberg gain is established."** It means the premise for stopping
at Gate 3 was wrong. What actually stands:

- One seed, `samples=500` (not the audit's `10000`) -- both FID and the MSE result above
  need the 3-seed replication `docs/AQUILA_PORT.md`'s wiring section already flags as
  necessary for real statistics; a ~2 FID gap to `interaction` from one data point at
  reduced sample size is not distinguishable from noise (the audit's own Fashion-MNIST
  QRC-vs-interaction contrast was itself a statistical tie, p=0.55-0.71, s5d).
- The MSE negative is still real and still needs an explanation (probe-time
  multicollinearity, above) -- it just isn't the generation-quality answer.
- **Gates 4-6 were incorrectly skipped and should run**: Gate 4's sweep (`a`, `Omega`, `S`,
  `V`) is exactly the mechanism to find out whether this FID result holds up, improves, or
  was a lucky single seed, and Gate 6 is where the real (not preliminary/reduced-scale) FID
  comparison belongs.

**One seed only** (`seed=0`), and `samples=500` not the audit's `10000` for the FID check --
both a real limit on precision, not on the reversed conclusion's direction (the effect sizes
here, +13 FID over `random` and -1.8 FID under `classical`, are large relative to typical
single-seed FID noise at this sample size, but "large relative to typical noise" is not the
same as "confirmed over 3 independent seeds").

**Deviations from the task's literal smoke-test commands**: run with `--device cuda`, not
`--device cpu` as specified, for both the supervised smoke cell and the generation cell. At
the measured CPU cost (~8s/sample-probe at the accuracy Gate 1 needed), the supervised cell
(500 train + 500 val + 2000 test rows x 4 probe times = 12,000 program-evolutions) would
take multiple hours instead of ~50 minutes; the generation cell (50 DDIM steps x 500
samples x 4 probe times) took ~101 minutes on GPU at a relaxed `rk4_safety=0.2` (vs. the
default `0.05`) -- justified because Gate 1's own cross-validation floor was ~1e-5, and
`rk4_safety=0.2`'s internal discretization error (~2.2e-6, measured on the closed-form drive
test) is already comfortably under that, so the relaxation trades away precision the
cross-validation couldn't back up anyway, not precision that mattered. Separately, Fashion-MNIST
Inception Score was unavailable for every cell (`experiments.phase_f.FashionCNN`, the class
`results/fmnist_classifier.pt` needs, is one of the phase0-phase5 files the README says were
deliberately excluded from this trimmed repo) -- a pre-existing repo-scoping gap, not
introduced by this work, made non-fatal in `qrc_fusion_fair_generation.py` with a
try/except that falls back to the same `NaN` the code already used when the checkpoint file
itself was absent; FID/PRDC do not depend on this classifier and are unaffected.

## Gate 4: emulation sweep -- and why it doesn't select by val_mse alone

Gate 3 established something specific to carry into Gate 4: `val_mse` is not merely noisy at
this operating point, it is *actively anti-correlated* with FID (rydberg loses to `random` on
val_mse yet beats `classical` on FID). "Escolher uma configuracao por validacao, nao por
teste" cannot mean "trust val_mse blindly" after that result, so Gate 4 treats val_mse as a
cheap filter/diagnostic, not the selector, and confirms any shortlist with the real metric
(FID) before calling it a choice. `experiments/gate4_sweep.py` runs the diagnostic pass;
`experiments/qrc_fusion_fair_generation.py` (now taking `--spacing-um/--omega/--s-detuning
/--v-slices/--t-scale/--tag`) runs the confirmation.

**Sweep design.** One-factor-at-a-time from the Gate 0-3 baseline (`a=8um`, `Omega=6.283`,
`S=9.0`, `V=4`, `t_scale=1.0`), not a full factorial (5 knobs x ~3 values each would be days
of wall clock at this solver cost, not a sweep) -- coverage and diagnosis, not an exhaustive
search. `a` and `Omega` are swept as `(Omega, a/R_b ratio)` pairs with
`a = ratio * blockade_radius(Omega)`, re-quantized to the 10nm grid, since `a` and `Omega` set
the same physical axis (how deep in blockade the cluster sits) and varying `a` alone at fixed
`Omega` while claiming to sweep "a/R_b" would silently conflate the two. `n_train=200` (val
fixed at 500, `qrc_fusion_fair_core`'s own `n_val=max(500, n_train//2)` rule) for speed;
`rk4_safety` left at the default `0.05`, not the relaxed `0.2` the generation runs use, since
the sweep is exploratory and its own cost (not accuracy) was the constraint here.

Beyond `val_mse`, each config also records `n_floored_columns` (uninformative here -- 0 in
every one of the 11 configs) and **cross-probe-time block correlation**: mean `|Pearson r|`
between the same observable at different probe times, averaged over all `C(V,2)` block pairs
-- the mechanism Gate 3's writeup blamed for the MSE damage (four probe times of the same
underlying trajectory are correlated; a config that decorrelates them has a physically
motivated reason to be preferred independent of what it does to val_mse alone).

**Results** (`results/qrc_fusion_fair/gate4_sweep.json`, sorted by val_mse):

| config | val_mse | block_corr | delta vs random (val_mse) |
|---|---|---|---|
| `ratio=1.0` (a=9.757um) | 0.6657 | 0.3097 | +0.098 |
| `s_detuning=4.5` | 0.6738 | 0.1847 | +0.106 |
| `v_slices=2` | 0.7090 | 0.2265 | +0.149 |
| `v_slices=6` | 0.7416 | 0.2801 | +0.170 |
| `t_scale=0.5` | 0.7463 | 0.2013 | +0.179 |
| `omega=12.0` (a=7.182um, same ratio as baseline) | 0.7523 | 0.1659 | +0.185 |
| baseline | 0.7598 | 0.2053 | +0.192 |
| `ratio=0.7` (a=6.830um) | 0.7622 | 0.3119 | +0.195 |
| `omega=3.14` | 0.7698 | 0.2968 | +0.202 |
| `s_detuning=18.0` | 0.7764 | 0.4135 | +0.209 |
| `t_scale=2.0` | 0.7837 | 0.2216 | +0.216 |

Block correlation does **not** track val_mse cleanly -- `ratio=1.0` has the *second-highest*
block correlation (0.310) yet the *best* val_mse, while `omega=12.0` has the *lowest* block
correlation (0.166) yet only a middling val_mse. The multicollinearity story from Gate 3 is a
plausible contributor, not the whole mechanism; treating it as a reliable standalone proxy
would have been the same mistake as trusting val_mse alone, just one level down. This
confirms the advisor's caution going in: there is no single cheap number here that can be
trusted without an FID check.

**FID confirmation on a 2-point shortlist** (baseline vs. the best-val_mse pick vs. the
lowest-block-correlation pick, chosen specifically so the check could discriminate between the
two candidate mechanisms), same protocol as Gate 3's preliminary FID check
(`seed=0`, `n_train=500`, `samples=500`, `rk4_safety=0.2`):

| config | val_mse (n_train=500) | FID | precision | recall | diversity |
|---|---|---|---|---|---|
| `ratio=1.0` | 0.6067 | **85.40** | 0.722 | 0.581 | 0.898 |
| baseline | 0.6483 | 90.47 | 0.748 | 0.513 | 0.723 |
| `omega=12.0` | 0.6639 | 98.35 | 0.646 | 0.385 | 0.646 |
| classical | -- | 92.31 | 0.692 | 0.418 | 0.823 |
| interaction | -- | 88.46 | 0.670 | 0.516 | 0.991 |
| random | -- | 103.50 | 0.670 | 0.371 | 0.633 |

`ratio=1.0` -- moving the cluster spacing from `0.82*R_b` to exactly `1.0*R_b` (a=9.757um
instead of 8um, `Omega`/`S`/`V` unchanged) -- **wins on both signals**: better val_mse at
*both* `n_train=200` (sweep) and `n_train=500` (confirmation, a robustness check the sweep
alone couldn't give), and the best FID of every arm tested so far, including `interaction`.
`recall`/`diversity` both rise sharply over baseline (0.513->0.581, 0.723->0.898) at a modest
precision cost (0.748->0.722) -- more of the real distribution's support is being reached, not
a degenerate collapse (`degenerate=False`, `frac_at_clip=0.0` for both).

`omega=12.0` -- the block-correlation-favored pick -- **loses on both signals**: worse val_mse
at `n_train=500` than the sweep's `n_train=200` predicted (0.752 -> 0.664, i.e. it *flips
sign* relative to baseline between the two scales) and the worst FID of any rydberg config
tried, worse even than `classical`. This is the discriminating result the two-point shortlist
was built to produce: val_mse's *within-rydberg-family* ranking held up under a completely
different n_train and a different metric (FID), while block correlation's ranking did not.
That does not rehabilitate val_mse for the *rydberg-vs-classical* comparison (Gate 3's
reversal there stands), only for comparisons *within* this sweep's local neighborhood --
report both directions rather than treating either signal as generally reliable.

**Selected operating point: `a=9.757um` (`a/R_b=1.0`), `Omega=6.283`, `S=9.0`, `V=4`,
`t_scale=1.0`** -- the rest of the sweep (`S`, `V`, `t_scale`) was not re-confirmed against FID
given the compute budget already spent (~2.5h sweep + ~2.6h for the two-point FID check); `S=4.5`
is the strongest untested candidate (second-best val_mse at both scales' proxy *and* low block
correlation, the one config where the two signals agreed) and is a reasonable next probe if the
Gate 5/6 numbers below don't already look conclusive on this operating point.

**What this does not establish**: one seed, `samples=500` not the audit's `10000`, same
caveat as Gate 3's preliminary check. The `a/R_b=1.0` win is consistent across two n_train
scales and two metrics, which is stronger evidence than Gate 3's single-point reversal had,
but still short of the 3-seed replication both the Gate 3 section and this one flag as needed
before either result is a settled number.

## Gate 5: shot noise end-to-end -- the exact-feature gain does not survive it

Gates 3-4 (and the preliminary Gate 6 FID check) were all run on **exact** statevector
expectations (`shots=None`) -- the Hamiltonian dynamics as an idealization, not what the
device actually reports. Gate 5 asks the concrete question a real Aquila run forces: does the
FID edge found at the Gate 4 operating point (`a/R_b=1.0`, `Omega=6.283`, `S=9.0`, `V=4`)
survive finite, noisy shots at `shots<=MAX_SHOTS=1000` (the single-task hardware ceiling)?

**Machinery added.** `quera.features.ReservoirParams.noise_mode` (`"shots"`|`"gaussian"`) and
`quera.features.gaussian_noise_like_shots` -- the Gate 5 control. Real shot noise
(`zz_features_sampled`) draws `Z_i`/`Z_iZ_j` from the *same* multinomial samples at a given
probe time, so they're structurally correlated within a probe time; the Gaussian control adds
i.i.d. noise of the same per-feature variance (`physical_variance_floor`) with none of that
structure. If the two behave the same downstream, whatever `S`-dependence shows up is generic
noise-as-regularizer, not something specific to projective measurement statistics.
`--shots/--vacancy-rate/--fp-rate/--fn-rate/--noise-mode` were added to
`qrc_fusion_fair_generation.py`; `fp_rate`/`fn_rate` stay at 0 throughout Gate 5 -- the task
spec gives a concrete vacancy rate (~1%/site) but no concrete detection-error rate, and the
hard rule is to ask rather than invent hardware numbers, so false-positive/negative detection
error is out of scope pending an explicit value.

**Cheap val_mse sweep** (`experiments/gate5_shot_noise.py`, `results/qrc_fusion_fair/gate5_sweep.json`,
`n_train=200`, Gate 4's geometry, `vacancy_rate=0.01` on the `shots` arm):

| shots | val_mse (shots) | val_mse (gaussian control) |
|---|---|---|
| exact | 0.6657 | -- |
| 50 | 0.7846 | 0.7867 |
| 100 | 0.7448 | 0.7500 |
| 300 | 0.7099 | 0.7060 |
| 1000 | 0.6789 | 0.6808 |

Two things fall out cleanly. First, val_mse converges toward the exact value as `S` grows, and
at `S=1000` is already close (0.679 vs. 0.666 exact) -- the RK4 evolution cost dominates this
sweep regardless of `S` (~280s/config, flat), so this curve was cheap to get in full. Second,
**the Gaussian control tracks real shot noise within ~0.5% at every `S` tested** -- the
structure of projective measurement (correlated Z/ZZ from the same shots, vacancy filtering)
makes no detectable difference here; the degradation is fully explained by noise magnitude.
This is a clean answer to the "regularizer or genuinely quantum" question Gate 5 was meant to
separate, and it means the Gaussian control did not need to be re-run at FID scale (see below)
-- the mechanism question was already settled at the cheap-sweep stage.

**FID confirmation** (same protocol as Gate 3/4's checks: `seed=0`, `n_train=500`,
`samples=500`, `rk4_safety=0.2`, real shot noise, `vacancy_rate=0.01`) at `shots=1000` (the
hardware ceiling -- the actual deployable operating point, not an idealization) and
`shots=100` (to see how far the degradation goes):

| config | FID | precision | recall | density | coverage |
|---|---|---|---|---|---|
| exact (Gate 4) | 85.40 | 0.722 | 0.581 | 0.410 | 0.339 |
| `shots=1000` | 93.18 | 0.690 | 0.556 | 0.403 | 0.334 |
| `shots=100` | 111.18 | 0.644 | 0.347 | 0.335 | 0.231 |
| classical | 92.31 | 0.692 | 0.418 | 0.368 | 0.288 |
| random | 103.50 | 0.670 | 0.371 | 0.359 | 0.272 |
| interaction | 88.46 | 0.670 | 0.516 | 0.426 | 0.352 |

**The gain does not survive.** At `shots=1000` -- the maximum a single Aquila task allows --
FID rises from 85.40 to 93.18, which is now *worse* than `classical` (92.31), not better; the
0.86-point gap to `classical` is well inside the noise floor the earlier audit calibrated (a
~2-point single-seed FID gap at this sample size was called statistically indistinguishable
from noise, `s5d`). At `shots=100` the reservoir is worse than every other arm, including
`random`. Every diagnostic degrades monotonically and smoothly from exact to `shots=1000` to
`shots=100` (precision, recall, density, coverage all fall in lockstep; `degenerate=False` and
`readout_max_diff=0` at both points), so this reads as a real information-loss effect, not a
pipeline artifact.

**"S minimo" -- the direct answer the task's final report asks for: there is no `S<=1000` at
which the exact-feature FID gain reliably survives.** The Gate 3/4 result is a property of the
noiseless statevector idealization, not of the device as it would actually be operated. This
is the same MSE-does-not-equal-FID lesson from Gate 3, playing out again one level down: here
it is *FID itself* (the metric that reversed Gate 3's negative) that turns out not to survive
the next layer of physical realism, which is exactly why Gate 5 (not just Gate 4) had to run
before calling any of this settled.

### How far above the ceiling would `S` have to go? (extrapolation, not measurement)

The two FID points measured here (`S=100`, `S=1000`) pin a clean power law for the gap against
the exact-feature FID:

```
gap(S) = FID(S) - 85.40 = 283.5 * S^-0.521
```

The exponent **0.521** lands almost exactly on the `1/sqrt(S)` law Gate 2 measured for the
features themselves (`std*sqrt(S)` in `0.92-0.94` across three orders of magnitude), i.e. the FID
degradation inherits the counting statistics directly, with no additional saturation or
threshold effect on top. Extrapolating it:

| target | `S` required | tasks at 1000 shots |
|---|---|---|
| tie `classical` (92.31) | ~1,250 | 2 |
| tie `interaction` (88.46) | ~6,000 | 7 |

Two caveats, and the second is the one that matters. First, every entry above needs **shots
aggregated across multiple task submissions** -- the operating mode this gate is explicitly not
scoped to, and which carries between-submission calibration drift that nothing here measured.
Second, **"ties `classical` at S~1250" is not a result**: Gate 6 measured the seed-to-seed SD of
`classical` at **2.91 FID**, and the gap that `S` would close is 0.86 -- a third of one SD, well
inside the noise. The only bar that would mean anything is `interaction`, and that is ~6,000
shots (7 tasks) *to tie*, not to win. The honest reading of the whole `S` axis is therefore that
**there is no hardware-feasible shot count at which the Gate 4 gain survives** -- it is a
property of the exact-statevector idealization, and this extrapolation is offered as the reason
to stop looking rather than as a target to chase.

**What this does not establish**: one seed, `samples=500`, and only two `S` values probed
directly at FID scale (`1000`, `100`) rather than the full `{50,100,300,1000}` grid the cheap
sweep covered -- so the power law above is a two-point fit whose exponent happens to agree with
independent theory, not a measured curve -- the cheap sweep's own val_mse curve is smooth and monotonic in `S`, so
interpolating between `100` and `1000` is a reasonable guess, not a measured claim.
Aggregating shots across multiple hardware task submissions (the `test_shot_convergence.py`
comment's `S=5000` point) could in principle recover more of the gap than a single `S=1000`
task can, but that is a multi-task operating mode outside what Gate 5 (or the spec's
single-task shot ceiling) was scoped to test, and is flagged here rather than quietly assumed
to fix the result.

## Gate 6: full-scale, 3-seed replication -- RUNNING

Every number in Gates 3-5 is **one data seed at `samples=500`**, and both of those sections say
so in their own "what this does not establish" paragraph. Gate 6 is the replication that attaches
a spread to them: `samples=10000` (the audit's own scale, not a reduced-scale proxy) across data
seeds `{0,1,2}`, at the Gate 4-selected operating point (`a=9.757um` i.e. `a/R_b=1.0`,
`Omega=6.283`, `S=9.0`, `V=4`, `t_scale=1.0`), for both of the arms whose single-seed results the
earlier gates disagree about:

- `exact` (`shots=None`) -- Gate 4's headline, that the Rydberg arm's FID beats `classical` and
  `interaction`.
- `shots=1000`, `vacancy_rate=0.01` -- Gate 5's headline, that the gain does not survive the
  single-task hardware shot ceiling.

`experiments/gate6_replication.py` is the driver (one subprocess per cell, resumable -- see
below); `experiments/gate6_analyze.py` aggregates. **`S=4.5` is deliberately out of scope**: it
is the strongest untested candidate from Gate 4 and is flagged as such at the end of that
section, but Gate 6 is replication *at the selected operating point*, and adding a new
configuration would make it another Gate 4 branch rather than a replication of the existing one.

### What did not need re-running, and how that was checked rather than assumed

The digital arms at this exact cell definition (`fashionmnist`, `n_train=500`, `samples=10000`,
seeds 0/1/2) are **already committed** from the original fair-comparison study: `classical`,
`interaction`, digital `qrc` (5 draws x 3 seeds) and digital `random` (5 draws x 3 seeds, width
84). Those were produced on the other machine, the same one whose supervised numbers differ from
this machine's in the 5e-7 range (see the fixture section), so reusing them needed an actual
comparability measurement, not an argument. Re-running `classical`/`seed=0` here gave

```
              committed (other machine)   this machine     diff
fid                     70.769                70.604      -0.165
diversity                0.8297                0.8297      0.0000
```

-- a 0.165 FID difference against a **seed-to-seed SD of 2.906** for that same arm, i.e. two
orders of magnitude inside the spread Gate 6 is measuring. The committed cells are used as-is;
the check cell is kept under the `gate6chk` tag rather than deleted.

What the committed cells do **not** provide is a random control matched to the Rydberg arm's
width: the digital `random` cells are 84-dim (the digital reservoir's width), while the Rydberg
arm emits `n_features(12)*V = 312`. Gate 3's "beats random by a wide margin" is a claim about a
*dimension-matched* control, so Gate 6 queues 312-dim random cells itself -- 3 seeds x 5 draws,
since at ~21s each there is no reason to accept a single-draw estimate of that arm's spread.

### Cost, and the order the queue runs in

Measured on this machine at the Gate 4 geometry (not extrapolated from Gate 3's denser-8um
figure, which is ~2.6x slower): `2343s` of generation for 500 samples = **4.69 s/sample**, so one
Rydberg cell at `samples=10000` is **~13 h** and the six of them are **~78 h (~3.3 days)** of
GPU. There is no honest way to shrink that and keep "full-scale" in Gate 6: `rk4_safety` is
already at Gates 3-5's relaxed `0.2`, and per-sample cost is flat in `--batch` from 256 upward
(the performance-fix section above), so the remaining cost is genuine RK4 compute.

The queue is therefore ordered by decision value and each cell lands in its own parquet as it
finishes (`qrc_fusion_fair_generation.py` SKIPs existing cells, so killing the driver and
re-running resumes rather than restarts): cheap 312-dim controls first, then
`(seed 0 exact, seed 0 shots)`, then seed 1's pair, then seed 2's. After ~26 h there is a
complete full-scale **seed-0 pair** -- a directly comparable replacement for the Gate 4 and
Gate 5 tables -- and every later seed closes a *within-seed* exact-vs-shots pair rather than
leaving a half pair behind.

### Statistics: what n=3 can and cannot support here

Per the rule fixed up front in the wiring section, **no headline p-value is reported for the
Rydberg contrast.** The Rydberg map has no `draw` ensemble, so it gets exactly one cell per
(seed, n_train): 3 independent pairs, not the digital arm's 15 correlated ones, and a p-value
from n=3 is not comparable to the audit's tables. `gate6_analyze.py` reports per-seed paired
deltas plus the mean and SD across seeds, and collapses draw-ensemble arms to a per-seed mean
first (averaging all 15 digital `qrc` cells directly would weight seeds by draw count and
understate exactly the seed-to-seed spread the pairing is about).

Two limits are structural to the existing pipeline and are stated rather than fixed, because
changing either would break comparability with every cell already committed:

- **The generation noise is seed-independent.** `qrc_fusion_fair_generation.py:76` draws it from
  a fixed `default_rng(ROOT + 999)`, shared by every arm and seed, so `--seed` varies the
  training/validation draw and the readout fitted from it -- *not* the sampler's noise. Gate 6's
  3 seeds therefore measure readout-fit variability, not generation-noise variability. (One
  incidental benefit: the 10000-sample noise is a strict superset of the 500-sample draws, so
  these cells extend the Gate 3-5 ones rather than being an independent redraw of them.)
- **PRDC is not full-scale even though FID is.** `PRDC_SUBSAMPLE = 1000` caps
  precision/recall/density/coverage at 1000 generated samples regardless of `--samples`, so the
  two metric families in the table below do not have the same sample backing.

The FID **null floor** (`metrics.fid_null_floor` -- what a *perfect* generator scores against a
finite real reference) is computed for Fashion-MNIST in `gate6_analyze.py` rather than by editing
the shared generation script. Its construction is conservative and the docstring says why: the
reference set has exactly 10000 images, so a floor at `n_generated=10000` against a full 10000-image
reference cannot be built from disjoint halves; it splits 5000/5000 and bootstraps 10000 draws
from the second half. Both departures push plug-in FID *up*, so the true floor is no larger than
the reported one -- it bounds how small a gap could possibly mean anything, and is not a number
to subtract from the FIDs.

One failure mode was checked in advance rather than discovered 13 h into a cell: the rollout's
`--probe-limit` guard (`qrc_fusion_fair_generation.py:131`) fails a cell outright if standardized
features leave the training distribution, and going from 500 to 10000 samples can only push the
*max* over samples up. The existing Rydberg `n500` cells report `probe_max_abs_z` of
**4.80-5.11** (mean `|z|` ~0.7) against the default limit of **50** -- an order of magnitude of
headroom, far more than a 20x sample increase can consume through the extreme-value tail. The
limit is therefore left at its default and no cell is expected to trip it; the driver's
per-cell `rc!=0` handling would keep the queue alive if one did.

### Results -- digital backdrop measured, Rydberg rows NOT yet run

**Nothing below replicates Gate 4 or Gate 5 yet.** Every number in this section comes from the
pre-existing committed digital cells; **zero full-scale Rydberg cells have finished**, so Gate 6
currently has no verdict on either the Gate 4 gain or the Gate 5 reversal. The table is filled in
from `results/qrc_fusion_fair/gate6_summary.json` as the queue drains. What the digital arms
provide right now is the fixed backdrop -- and, more usefully, the first same-protocol
measurement of the seed-to-seed spread -- that the Rydberg cells will land against:

| arm | n seeds | FID (mean +/- SD over seeds) | delta vs `classical` (mean +/- SD) |
|---|---|---|---|
| `interaction` | 3 | 66.11 +/- 1.36 | -6.24 +/- 1.61 |
| `qrc` (digital, 5 draws/seed) | 3 | 66.75 +/- 3.67 | -5.60 +/- 1.20 |
| `classical` | 3 | 72.35 +/- 2.91 | -- |
| `random` (84-dim, 5 draws/seed) | 3 | 90.10 +/- 3.23 | +17.75 +/- 4.51 |
| `random` (312-dim, 5 draws/seed) | 3 | 103.02 +/- 5.22 | +30.67 +/- 3.23 |
| `qrc` (rydberg, exact) | **3 of 3** | 68.15 +/- 5.00 | **-4.20 +/- 2.52** |
| `qrc` (rydberg, `shots=1000`) | **3 of 3** | 77.63 +/- 6.77 | **+5.29 +/- 4.04** |

(SDs are over the 3 per-seed means, draws collapsed within a seed first as described above --
not over all 15 raw cells, which for `qrc` digital would read 4.57 instead of 3.67 by counting
correlated draws as independent.)

**The FID null floor is 6.65 +/- 0.05**, an order of magnitude below every arm's FID and two
orders below the gaps between them -- so for this comparison the binding precision limit is
*not* the finite real reference, it is the seed-to-seed spread. That reframes what the earlier
gates were up against. The `classical` SD alone is **2.91 FID**, while Gate 5's decisive number
was a **0.86** FID gap between `shots=1000` and `classical` -- a gap that section called "well
inside the noise floor" on the strength of the audit's `samples=500` calibration. This is the
first direct, same-protocol measurement of that floor at full scale, and it agrees: the gap is
roughly a third of one seed-to-seed SD of the reference arm on its own.

Also visible already: the paired contrasts that *are* complete (`interaction` and digital `qrc`
vs `classical`) have per-seed deltas of consistent sign across all three seeds
(`-5.07/-8.07/-5.57` and `-6.99/-4.85/-4.96`), i.e. the paired delta is considerably tighter
than each arm's absolute SD. That is the payoff of pairing within seed, and it is the right
yardstick for reading the Rydberg rows when they land -- an absolute FID that moves by 3 across
seeds may still carry a delta that does not.

**First full-scale Rydberg cell (seed 0, exact): FID 63.78, and the reduced-scale protocol
held.** It took 13.07 h, as projected, and `probe_max_abs_z=5.64` against the limit of 50 --
the headroom argument above was right. Against the same seed's digital cells it beats `classical`
(70.77), beats `interaction` (65.70), and lands on top of digital `qrc` (63.78). The number worth
noting is not the absolute FID but the **paired delta: -6.99 at `samples=10000` against -6.91 at
`samples=500`**. The absolute FID moved by 22 points between the two sample sizes (plug-in FID is
strongly n-dependent, which is exactly why `fid_null_floor` exists), while the delta against
`classical` moved by 0.08. That is direct evidence the reduced-scale protocol Gates 3-5 ran on was
measuring the right quantity, and it is the first time this port has been able to check that
rather than assert it. **One seed is still one seed** -- seeds 1 and 2 decide whether it holds.

**Seed 0's pair is complete, and Gate 5's headline replicates at full scale.** The
`shots=1000` cell came in at **72.27** against the same seed's `classical` at 70.77 -- i.e. the
Rydberg arm is **worse than classical once shot noise is applied**, exactly as Gate 5 found at
`samples=500`. Both halves of the story hold their size across a 20x change in sample count:

| quantity | `samples=500` | `samples=10000` |
|---|---|---|
| exact, delta vs `classical` | -6.91 | **-6.99** |
| shot-noise penalty (exact -> `shots=1000`) | +7.77 | **+8.48** |
| `shots=1000`, delta vs `classical` | +0.86 | **+1.50** |

Every diagnostic degrades in the same direction as before (precision 0.713 -> 0.668, recall
0.572 -> 0.460, coverage 0.490 -> 0.381, diversity 0.909 -> 0.772), with `degenerate=False`
throughout. This is the second independent confirmation that the reduced-scale protocol Gates 3-5
ran on was measuring the right *deltas* even though its absolute FIDs were inflated by ~22 points.

**The exact arm is complete over all three seeds, and it splits cleanly by which control it is
measured against.** Seed 0 alone would have overclaimed on both -- this is Gate 6 doing the job it
was built for:

| seed | rydberg exact | `classical` | `interaction` | delta vs `classical` | delta vs `interaction` |
|---|---|---|---|---|---|
| 0 | 63.78 | 70.77 | 65.70 | **-6.99** | **-1.92** |
| 1 | 73.60 | 75.70 | 67.63 | **-2.10** | **+5.98** |
| 2 | 67.06 | 70.57 | 65.00 | **-3.51** | **+2.06** |
| mean | **68.15 +/- 5.00** | -- | -- | **-4.20 +/- 2.52** | **+2.04 +/- 3.95** |

Two readings, and they differ:

- **Against plain `classical` the sign is stable across all three seeds** (`-6.99`, `-2.10`,
  `-3.51`), mean `-4.20 +/- 2.52`. Per this document's own rule for the Rydberg arm -- n=3
  independent pairs, so per-seed deltas and their spread rather than a headline p-value -- the
  honest statement is: **consistently favorable in direction on every seed, with a magnitude
  comparable to its own spread.** The mean is 1.7x the SD; that is a real effect by the standards
  available here, not a settled effect size.
- **Against `interaction` the sign does not hold**: `-1.92`, `+5.98`, `+2.06`, mean `+2.04 +/-
  3.95`. Two of three seeds unfavorable and the mean unfavorable. The seed-0 claim that the
  Rydberg arm beats the structured classical control **does not replicate**.

That second row settles a question Gate 3 explicitly left open. Its preliminary check saw the
Rydberg arm land "within ~2 FID of interaction" and flagged that a 2-FID gap from one reduced-scale
cell was not distinguishable from noise. It was not: at full scale, over two seeds, the Rydberg arm
**loses to `interaction` on average**. What survives is exactly the pattern
`QRC_FUSION_FID_BIAS_AUDIT.md` documented for the *digital* arm -- beats plain classical, beats a
dimension-matched random control by a wide margin (`-4.54` vs `+30.67`), but does not beat a
structured classical block of comparable capacity.

One caveat on comparing those SDs. The Rydberg arm has **one cell per seed** (no `draw` ensemble),
so its per-seed number carries full cell-level noise, while digital `qrc` and `random` per-seed
means average 5 draws and are correspondingly tighter by roughly `sqrt(5)`. The fair comparisons
for spread are `classical` (SD 2.91) and `interaction` (delta SD 1.61), which also have one cell
per seed -- against those the Rydberg arm is still the noisiest of the three.

**Seed 1's `shots=1000` cell strengthens Gate 5 rather than softening it.** Where seed 0 put the
Rydberg arm only `+1.50` behind `classical` under shot noise -- inside the 2.91 SD, hence the
earlier caution in this document that it might soften to a statistical tie -- seed 1 puts it
`+9.53` behind:

| seed | exact | `shots=1000` | noise penalty | delta vs `classical` |
|---|---|---|---|---|
| 0 | 63.78 | 72.27 | +8.48 | **+1.50** |
| 1 | 73.60 | 85.23 | +11.63 | **+9.53** |
| mean | 68.69 | 78.75 | **+10.06 +/- 2.22** | **+5.51 +/- 5.68** |

**Both seeds are on the same side**: worse than plain `classical` at the hardware shot ceiling, by
+1.50 and +9.53. The speculation recorded earlier here -- that the result might soften to a tie --
was wrong; it went the other way. Note also that seed 0 was the favorable draw on *both* metrics
(largest exact gain and smallest shot penalty), so the single-seed picture Gates 3-5 were built on
sat at the optimistic end of the range on both axes at once, not just one.

The **noise penalty is the more stable quantity** (`+10.06 +/- 2.22`) than either arm's absolute
FID, which is what one expects if shot noise acts as an approximately configuration-independent
information loss -- consistent with Gate 5's finding that a variance-matched Gaussian control
reproduced its damage within ~0.5%.

**Seed 2 closes the arm** at `+4.83`, so all three seeds are unfavorable: `+1.50`, `+9.53`,
`+4.83`, mean **`+5.29 +/- 4.04`**, with a noise penalty of `+9.48 +/- 1.86` -- again the most
stable quantity in the table.

## Gate 6 verdict (all 21 cells complete)

| arm | FID (3 seeds) | delta vs `classical` | per-seed signs |
|---|---|---|---|
| `interaction` | 66.11 +/- 1.36 | **-6.24 +/- 1.61** | 3/3 favorable |
| `qrc` digital | 66.75 +/- 3.67 | -5.60 +/- 1.20 | 3/3 favorable |
| **rydberg exact** | 68.15 +/- 5.00 | **-4.20 +/- 2.52** | **3/3 favorable** |
| `classical` | 72.35 +/- 2.91 | -- | -- |
| **rydberg `shots=1000`** | 77.63 +/- 6.77 | **+5.29 +/- 4.04** | **3/3 unfavorable** |
| `random` 84-dim | 90.10 +/- 3.23 | +17.75 +/- 4.51 | 3/3 unfavorable |
| `random` 312-dim | 103.02 +/- 5.22 | +30.67 +/- 3.23 | 3/3 unfavorable |

Three statements, in decreasing order of how well they are supported:

1. **Shot noise removes the gain, and this is the firmest result here.** All three seeds put the
   Rydberg arm behind plain `classical` at the hardware shot ceiling. **(Revised by seeds 3-5 --
   see that section: at six seeds this contrast is a statistical tie, `+1.99 +/- 8.58`, because
   `classical` is far noisier than three seeds showed. The deficit against `interaction`,
   `+12.08 +/- 3.86` on 6/6 seeds, is what survives.)** The single-seed picture
   Gates 3-5 rested on was the *optimistic* end on both axes at once (seed 0 had both the largest
   exact gain and the smallest shot penalty), so replication moved this result away from a tie,
   not toward one.
2. **The exact-feature gain over plain `classical` is real**: favorable on all three seeds, mean
   `-4.20` against its own SD of 2.52. Per this document's rule for the Rydberg arm (n=3
   independent pairs, no headline p-value) the defensible claim is consistent direction with an
   effect size comparable to its spread -- not a pinned-down number.
3. **It does not beat a structured classical control.** Against `interaction` the sign flips
   across seeds (`-1.92`, `+5.98`, `+2.06`), mean `+2.04`. This reproduces, for the Rydberg
   reservoir, exactly what `QRC_FUSION_FID_BIAS_AUDIT.md` found for the digital one: beats plain
   classical, beats a dimension-matched random control by a wide margin, loses or ties against a
   classical block of comparable capacity. **The port reproduces the original study's conclusion
   rather than improving on it.**

The 312-dim random control being *worse* than the 84-dim one (`+30.67` vs `+17.75`) is worth
carrying into any write-up: "beats a dimension-matched random control" is a weaker statement than
it sounds, because at this `n_train` 312 random features are simply harder to fit than 84.


**The dimension-matched random control is a weaker bar than the 84-dim one, not a stronger one,
and that cuts against the Gate 3 claim it was built to test.** Widening the random tanh
projection from 84 to 312 makes it *worse*, consistently and by a wide margin
(`+30.67 +/- 3.23` vs `+17.75 +/- 4.51` FID over `classical`; the per-seed deltas do not
overlap). So "the Rydberg arm beats a dimension-matched random control" -- Gate 3's stated
reason for believing the 312 features encode real structure rather than mere capacity -- is a
*less* demanding test than beating the 84-dim control would be, because at this `n_train` 312
random features are simply too many for the readout to fit well. That does not void Gate 3's
reasoning (a control that shares the Rydberg arm's width is still the right one for isolating
width from structure), but it does mean the size of that margin cannot be read as the size of
the structural gain, and the Rydberg-vs-`interaction` contrast is the load-bearing one.

## Gate 7: telling the readout about the noise, and picking the operating point by SNR

Gate 5 left two untried levers, and this gate tests both. Neither had been tried because both are
invisible at `shots=None`: (1) the readout is never told its features are shot-noise estimates --
`fit_readout` standardizes the observable block with a floor calibrated for *exact* observables --
and (2) Gate 4 selected the operating point on exact features, so it could not have noticed that
`sigma^2 = (1 - <O>^2)/S` makes noise depend on the observable's own value.

### The diagnostic that motivated the gate

Two numbers from the existing record, put side by side: the informative variation of each feature
across samples is `std in [0.065, 0.158]` (Gate 3's recorded diagnostic), while shot noise at
`S=1000` is `sigma ~ 0.032`. Measured directly here rather than inferred, at the Gate 4 operating
point with `vacancy_rate=0.01`:

```
median per-feature SNR = 2.19      39% of the 312 features have SNR < 2
mean |<O>| (polarization) = 0.129
```

So the features are individually barely above their own noise, and -- the part that matters for
lever (2) -- **the observables sit almost exactly at zero**, which is precisely where
`sigma^2 = (1 - <O>^2)/S` is largest. At `|<O>| = 0.129` the variance reduction from polarization
is `1 - 0.129^2 = 1.7%`, i.e. this operating point is paying nearly the full `1/S` noise and
leaving the entire shot-efficiency axis unused.

### Part 1: errors-in-variables readout -- implemented, and it is a null result

`experiments/gate7_noise_readout.py` wires in `quera.features.physical_variance_floor` (which
Gate 2 built, tested, and left connected to nothing) as a proper errors-in-variables correction.
Noisy features attenuate ridge: `E[X'X] = X_true'X_true + n*Sigma`, and here `Sigma` is **known
analytically** rather than estimated, since `sigma^2 = (1-<O>^2)/S` is exact for any +/-1-valued
observable -- which covers the `<Z_i>` singles and the `<Z_i Z_j>` correlators alike. Subtracting
it back off the Gram matrix undoes the attenuation. The correction is unbiased but
higher-variance, so a shrinkage `tau in [0,1]` scales it and is selected on validation jointly
with `lambda`; `tau=0` is in the grid and reproduces `fit_readout` **exactly**, which
`tests/test_gate7_readout.py` asserts against `sklearn.Ridge` to `rtol=1e-7` (6 tests, all
passing). That identity is the point of the design: the gate cannot lose on validation, and any
gain is measured against the identical incumbent rather than a re-tuned one.

Measured at the Gate 4 operating point, `n_train=200`, reusing Gate 5's cached features:

| shots | val_mse plain | val_mse noise-aware | selected `tau` | shot-noise damage | recovered |
|---|---|---|---|---|---|
| exact | 0.66570 | 0.66570 | 0.00 | -- | -- |
| 1000 | 0.67885 | 0.67849 | 1.00 | 0.01315 | 0.00036 (**2.7%**) |
| 300 | 0.70989 | 0.70946 | 1.00 | 0.04419 | 0.00044 (**1.0%**) |
| 100 | 0.74484 | 0.74421 | 1.00 | 0.07914 | 0.00063 (**0.8%**) |
| 50 | 0.78457 | 0.78457 | 0.00 | 0.11887 | 0.00000 (**0%**) |

**The correction recovers 0-3% of what shot noise costs, and nothing at all at `S=50`.** This is a
null result and is reported as one. Validation *does* select the full correction (`tau=1.0`) at
three of the four shot counts, so the machinery is being used, not silently switched off -- it
simply has almost nothing to give. The most likely reason is that ridge's validation-selected
`lambda` was already absorbing the attenuation implicitly: shrinkage and an EIV correction pull on
the same quantity from opposite directions, and with `lambda` free to move over a 9-decade grid the
explicit correction is largely redundant. Making the noise known to the readout is therefore
**not** a lever on the Gate 5 result, and no FID confirmation run is warranted -- a 0.0004 val_mse
change is orders of magnitude below the 2.91-FID seed-to-seed spread Gate 6 measured.

The negative is still worth having on the record: it removes the most obvious "you just fitted it
wrong" objection to Gate 5. The information lost to counting statistics is genuinely lost, not
recoverable by a better estimator downstream -- consistent with Gate 5's own finding that a
variance-matched Gaussian control reproduces real shot noise's damage within ~0.5%.

### Part 2: sweeping for shot efficiency

`experiments/gate7_snr_sweep.py` scores operating points by

```
SNR_j = std_over_samples(<O_j>) / mean_over_samples(sqrt((1 - <O_j>^2)/S_eff))
```

with `S_eff = S*(1-v)^12`, the vacancy-filtered usable count `quera/sampling.py` is already tested
against. Because `sigma ~ 1/sqrt(S)`, an SNR improvement of `k` substitutes for `k^2` shots -- so a
30% noise reduction is worth ~2x the shot budget, which is roughly the `S~1250` Gate 5's
extrapolation says would tie `classical`, obtainable *without* leaving the single-task ceiling.

**Polarization is explicitly not the objective**: a config could in principle polarize its
observables by evolving so little that every atom stays in `|g>` -- `<Z>=+1`, zero noise, zero
information. The signal-to-noise *ratio* is what prices that trade-off, which is why it, and not
`|<O>|`, is what the sweep maximizes. (This paragraph originally named `t_scale -> 0` as that
degenerate case, in both the doc and the script docstring. **That was wrong.** `t_scale` is a
post-standardization multiplier on the encoding's two *timestep* channels, `quera/encoding.py:41`;
it does not shorten the evolution, which is set by `t_max_us/v_slices`. The error was caught by the
measurement it was supposed to predict -- see the `polar_t_scale=0.25` row below.)

Per the standing rule from Gates 4 and 5, **SNR is a diagnostic, not a selector**: Gate 3 showed
val_mse can rank this reservoir backwards against FID and Gate 4 showed block correlation can do
the same one level down, so a third cheap proxy earns no more trust than the other two. Any
shortlist needs an FID confirmation run before it is a choice.

Pass A reuses Gate 5's and Gate 4's cached feature arrays for the 11 configurations already run
(free); polarization-targeted points Gate 4 never tried are behind `--extra` at ~280s of GPU each,
deliberately opt-in so this pass does not compete with Gate 6's replication queue for the GPU.

**Results: partial** (`results/qrc_fusion_fair/gate7_sweep.json`). Two configs measured so far,
and they already frame the question:

| config | median SNR | polarization `|<O>|` | features with SNR<2 | val_mse (exact) |
|---|---|---|---|---|
| `gate03_baseline` (a=8.0um) | **2.69** | **0.261** | **1%** | 0.75983 |
| `gate4_selected` (a=9.757um) | 2.19 | 0.129 | 39% | **0.66570** |

**The point Gate 4 rejected is the more shot-efficient of the two, by a wide margin** -- double
the polarization, and 1% of its features below SNR 2 against 39%. That is the anticipated blind
spot showing up exactly where predicted: Gate 4 selected on exact features and could not see this
axis. But the second column is the catch, and it is why SNR is a diagnostic rather than a
selector: `a=8um` buys its shot efficiency by being *worse* on exact features (0.760 vs 0.666).
Priced against Gate 5's measured law, the SNR gain (2.19->2.69, worth ~1.5x effective shots) is
worth roughly 1.5 FID of noise robustness, while the exact-feature loss between these two points
was 5.07 FID (90.47 vs 85.40 in Gate 4's own confirmation table). **On the two points measured,
shot efficiency and exact-feature quality are anti-correlated, and the trade is a bad one.**

Two points do not establish an anti-correlation, and the configs that would break the tie are the
polarization-targeted ones that hold Gate 4's geometry fixed while raising `S` -- exactly the
`--extra` set. Those are queued (below); if one of them shows high SNR *without* the exact-feature
penalty, part 2 has a lever and earns an FID confirmation run. If they all trace the same
trade-off curve, the honest conclusion is that this reservoir's shot efficiency and its
expressivity are two ends of one knob, and Gate 5's negative is structural rather than a bad
configuration choice.

### Part 2 results: the mechanism is not the one this gate was premised on

The `--extra` configs were pulled forward ahead of the rest (see the scheduling note below) because
they are the ones that break the tie: they raise the detuning `S` while holding Gate 4's geometry
fixed. Measured (`n_train=200`, `seed=0`, SNR priced at `S=1000` with `vacancy_rate=0.01`):

| config | `S` | median SNR | signal std | noise std | `|<O>|` | val_mse (exact) |
|---|---|---|---|---|---|---|
| `gate4_selected` | 9 | 2.19 | 0.0815 | 0.0331 | 0.129 | **0.66570** |
| `polar_s=18_at_gate4` | 18 | 3.93 | 0.1432 | 0.0330 | 0.151 | 0.71160 |
| `polar_s=27.0` | 27 | 5.40 | 0.1925 | 0.0325 | 0.177 | 0.72057 |
| `polar_s=36.0` | 36 | **6.91** | **0.2355** | 0.0324 | 0.223 | 0.77082 |
| `polar_t_scale=0.25` | 9 | 2.07 | 0.0769 | 0.0331 | 0.129 | **0.64959** |

**The premise of this gate was wrong about the mechanism, and the data says so plainly.** The
section above motivated part 2 by `sigma^2 = (1-<O>^2)/S`: polarize the observables and they get
cheaper to measure. That is not what happens. Across a 4x range of detuning the **noise std barely
moves (0.0331 -> 0.0324, ~2%)**, exactly as the polarization column predicts it should
(`|<O>|` only reaches 0.223, and `1 - 0.223^2` is a 5% variance reduction). The entire 3.2x SNR
gain comes from the **signal** rising 2.9x: stronger detuning makes the observables respond far
more strongly to the input data, widening `<O>`'s dynamic range across samples.

The metric survives its own premise being wrong, which is the one piece of luck here: SNR is a
ratio and does not care which term moves. Had this gate swept on `|<O>|` -- the quantity the
motivating argument actually named -- it would have ranked `S=36` a 1.7x improvement instead of
the 3.2x it really is, and would have missed the effect almost entirely.

### Pricing the trade-off, and an interior optimum

Higher `S` costs exact-feature quality (val_mse 0.666 -> 0.771) while buying noise robustness. Both
sides can be put in FID units: an SNR gain of `k` is worth `k^2` effective shots, priced through
Gate 5's measured law `gap(S) = 283.5*S^-0.521`; the exact-feature cost is priced at
**54 FID per unit val_mse**, the rate implied by the two configurations Gate 4 confirmed with
actual FID runs (`85.40 @ 0.66570` vs `90.47 @ 0.75983`).

| config | effective shots | noise robustness gained | exact-feature cost | **net vs Gate 4 point** |
|---|---|---|---|---|
| `S=18` | 3,238 | -3.55 | +2.47 | **-1.08** |
| `S=27` | 6,103 | -4.73 | +2.95 | **-1.78** |
| `S=36` | 10,011 | -5.42 | +5.66 | +0.24 |
| `t_scale=0.25` (`S=9`) | 893 | +0.46 | **-0.87** | **-0.41** |

**`S=27` is an interior optimum**, and its estimated FID at `shots=1000` is `93.18 - 1.78 = 91.4`
-- below `classical`'s 92.31. That is the first thing in this port pointing at a *reversal* of
Gate 5's negative, and it is exactly what part 2 was built to look for.

**The negative control falsified its own design, and left something useful behind.**
`polar_t_scale=0.25` was included to be the degenerate case -- high SNR, no information. It is
neither: its SNR is slightly *worse* than the baseline's (2.07 vs 2.19) and its exact val_mse is
slightly *better* (0.64959 vs 0.66570). Its polarization is 0.129, identical to the baseline's, to
three decimals -- which is precisely what the corrected reading of `t_scale` predicts, since it
never touches the evolution. So this gate has **no working negative control**: the degenerate
"polarize without informing" case would need a short-evolution config, reachable only through
`t_max_us/v_slices`, which nothing here sweeps. The claim that SNR cannot be trusted as a
standalone selector therefore still rests on the general rule from Gates 3-5, not on a
demonstration inside this gate.

What it did produce is a *different* axis: giving the timestep channels less of `h`'s dynamic range
improves exact-feature quality at essentially no SNR cost (net -0.41 FID, entirely from the
exact-feature side). That is consistent with Gate 3's multicollinearity finding -- the timestep is
the component most shared across probe times, so shrinking its share is the cheapest way to
decorrelate the blocks. It is also **orthogonal to the detuning axis**, which makes
`S=27` combined with `t_scale=0.25` the obvious next configuration: it would stack a -1.78 and a
-0.41 that come from different mechanisms. Whether they actually add is not something this sweep
can answer.

**It is a hypothesis, not a result, and three things have to be said about it.** First, it chains
**two extrapolations, each fitted on two points**: Gate 5's power law and the 54 FID/val_mse rate.
Second, that rate was calibrated across configurations differing in *geometry*, and applying it to
a *detuning* change assumes it transfers across a different knob -- the weakest link in the chain.
Third, `-1.78` sits inside the **2.91 FID seed-to-seed spread Gate 6 measured**, so even if the
estimate is right it is not separable from noise at n=1 seed.

### The FID confirmation falsified it, and SNR is the third failed proxy

`experiments/gate7_fid_confirm.py` ran the confirmation at Gate 5's protocol unchanged (`seed=0`,
`n_train=500`, `samples=500`, `rk4_safety=0.2`, `shots=1000`, `vacancy_rate=0.01`, Gate 4's
geometry), so it drops straight into Gate 5's table:

| cell | `S` | shots | FID | precision | recall | density | coverage | diversity |
|---|---|---|---|---|---|---|---|---|
| `ratio1p0` (Gate 4) | 9 | exact | 85.40 | 0.722 | 0.581 | 0.410 | 0.339 | 0.898 |
| `shots1000` (Gate 5) | 9 | 1000 | **93.18** | 0.690 | 0.556 | 0.403 | 0.334 | 0.765 |
| `s27_shots1000` | 27 | 1000 | **99.93** | 0.738 | **0.302** | 0.452 | 0.290 | **0.595** |

**Predicted 91.4, measured 99.93 -- wrong by 8.5 FID and in the wrong direction.** `S=27` is worse
than the Gate 4 point under noise (93.18), worse than `classical` (92.31), and heading toward
`random` (103.50). The estimate is dead.

**Where it broke, from the PRDC columns.** Precision actually *rose* (0.738, the best of any
noisy cell) while recall collapsed (0.556 -> 0.302) and diversity fell with it (0.765 -> 0.595):
`S=27` generates a narrow distribution that covers much less of the real support. So the 2.4x
"signal" gain the SNR metric measured was largely **uninformative variance** -- wider dynamic range
in the observables with no matching predictive content. `SNR_j` as defined treats every bit of
across-sample variance as signal, and that is exactly the assumption that failed.

**The method error is worth more than the result.** The cell reports `val_mse=0.6639` against
`0.6204` for `S=9` at the same 1000 shots -- i.e. **val_mse under noise already ranked `S=27`
worse**, and measuring it was a few minutes of work: Pass B does precisely this at the Gate 4 point
and was never extended to the polarization configs. Instead the decision was routed through a
decomposition into "noise-robustness credit + exact-feature cost", chaining two two-point
extrapolations, when the direct measurement was cheap and available. The decomposition also
double-counts: val_mse measured *under noise* already contains both effects that were being priced
separately. **Any future candidate gets its val_mse measured at the target shot count before any
FID run is scheduled.**

**Correction, from cell 2 -- the paragraph above blamed the wrong half, and SNR is not a failed
proxy.** `s27_exact` came in at **96.52** (predicted 88.35), which splits the error cleanly:

| | predicted | measured | error |
|---|---|---|---|
| exact-feature cost (`S=9` -> `S=27`) | +2.95 FID | **+11.12 FID** | **+8.17** |
| noise penalty at `S=27` | 3.02 FID | **3.41 FID** | +0.39 |

**The SNR half was right.** `SNR -> effective shots -> Gate 5's power law` predicted the `S=27`
noise penalty to within 0.4 FID, against a `S=9` penalty of 7.77 (predicted 7.75). The SNR gain was
real and it bought real robustness: the noise penalty fell 7.77 -> 3.41, a factor of 2.3, tracking
the 2.47x SNR gain closely. `S=27` genuinely is the more shot-robust configuration, exactly as the
metric claimed.

**What broke was the val_mse -> FID conversion**, which carried the entire error -- the weakest link
this section flagged in advance. The real rate on the *detuning* axis is **213 FID per unit
val_mse**, against the **54** calibrated on the *geometry* axis: a factor of 3.9. `S=27` fails
because its exact-feature FID is catastrophically worse than its val_mse suggested, not because
shot noise hurt it more than predicted.

So the standing rule needs stating more precisely than "no cheap scalar predicts FID". Two of the
three failures (val_mse in Gate 3, block correlation in Gate 4) were *ranking* failures. This one
was not: SNR ranked and priced correctly. **The thing that does not transfer is the val_mse->FID
exchange rate across different knobs** -- 54 on geometry, 213 on detuning, and nothing establishes
what it is on a third. Any prediction that converts val_mse into FID across a knob it was not
calibrated on is unsupported, and that is the specific error to avoid, not the use of cheap
diagnostics as such.

### The target this makes quantitative

Beating `classical` (92.31) at `shots=1000` requires `exact_FID + noise_penalty < 92.31`, and both
terms are now measurable in advance -- the second reliably, via SNR:

| config | SNR | noise penalty | exact FID | total |
|---|---|---|---|---|
| `S=9` (Gate 4) | 2.19 | 7.77 | 85.40 | 93.18 |
| `S=27` | 5.40 | **3.41** | 96.52 | 99.93 |

`S=27` bought 4.36 FID of robustness and paid 11.12 for it. The detuning axis is therefore
**closed**: every step up in `S` costs more in exact-feature quality than it returns in
robustness, and `S=18`/`S=36` sit on the same curve. What would work is a knob that raises SNR
while costing little or nothing on exact features -- which is precisely what `t_scale=0.25`
looked like in the sweep (better exact val_mse at slightly lower SNR). **No prediction is offered
for it here**: that would mean transferring the exchange rate across yet another knob, the exact
error this section exists to record. Cell 3 measures it.

### Cell 3: the last axis closes, on the cleanest MSE-vs-FID dissociation in the project

`s9_ts025_shots1000` -- `t_scale=0.25` at Gate 4's geometry and `S=9`, `shots=1000`:

| config (`shots=1000`) | val_mse (n_train=500) | FID | recall | coverage | diversity |
|---|---|---|---|---|---|
| `t_scale=1.0` (Gate 4/5) | 0.62041 | **93.18** | 0.556 | 0.334 | 0.765 |
| `t_scale=0.25` | **0.61445** | **95.33** | 0.492 | 0.309 | 0.732 |

**val_mse improved and FID got worse.** Same geometry, same shot count, one knob moved, and the
two metrics point in opposite directions -- `-0.006` on val_mse against `+2.15` on FID. This is
the MSE/FID dissociation `QRC_FUSION_FID_BIAS_AUDIT.md` recorded three times for the digital arm
and Gate 3 recorded for the Rydberg arm, and it is the **most controlled instance this project has
produced**: the earlier ones compared different arms or different geometries, this one isolates a
single encoding parameter. Anyone tempted to shortcut a future gate with val_mse should read this
row first.

### Gate 7 verdict

Both parts are closed, and neither moved the Gate 5 result:

| lever | outcome |
|---|---|
| Part 1 -- errors-in-variables readout | **null**: recovers 0-3% of the shot-noise damage |
| Part 2 -- detuning axis (`S=18/27/36`) | **closed**: buys 4.36 FID of robustness, costs 11.12 |
| Part 2 -- encoding axis (`t_scale=0.25`) | **closed**: better val_mse, +2.15 FID worse |

The best `shots=1000` FID in this port is still the Gate 4/5 operating point at **93.18**, above
`classical`'s 92.31. **Gate 5's conclusion now stands on much more than its original single
measurement**: two independent attempts to overturn it, four dedicated FID cells (~6 h of GPU),
all failed, and each failed for a reason that is now understood rather than merely observed.

What Gate 7 contributes positively is one validated tool and one sharpened target. The tool:
**SNR predicts the shot-noise penalty well** (3.02 predicted vs 3.41 measured at `S=27`, against
7.77 at `S=9`), so the noise half of any future candidate can be priced before spending GPU. The
target: `exact_FID + noise_penalty < 92.31`, with the noise term now cheap to estimate and the
exact term requiring an actual FID run, because no val_mse-to-FID exchange rate has transferred
across knobs (54 on geometry, 213 on detuning, and `t_scale` moves the two in *opposite*
directions).

### Pass A, all 15 configs: why no cheap screen can find a better operating point

Pass A finished after Gate 6 released the GPU. Sorted by SNR, with the noise penalty each SNR
implies through Gate 5's law:

| config | SNR | signal std | `|<O>|` | val_mse | implied noise penalty |
|---|---|---|---|---|---|
| `polar_s=36` | 6.91 | 0.2355 | 0.223 | 0.77082 | 2.34 |
| `polar_s=27` | 5.40 | 0.1925 | 0.177 | 0.72057 | 3.02 |
| `omega=3.14` | 4.36 | 0.1490 | 0.285 | 0.76977 | 3.77 |
| `polar_s=18` | 3.93 | 0.1432 | 0.151 | 0.71160 | 4.20 |
| `s_detuning=18` | 3.73 | 0.1213 | 0.282 | 0.77644 | 4.44 |
| `ratio=0.7` | 3.12 | 0.1012 | 0.332 | 0.76222 | 5.34 |
| ... | | | | | |
| `gate4_selected` | 2.19 | 0.0815 | 0.129 | **0.66570** | 7.75 |
| `s_detuning=4.5` | 2.11 | 0.0701 | 0.262 | 0.67376 | 8.03 |
| `polar_t_scale=0.25` | 2.07 | 0.0765 | 0.129 | **0.64959** | 8.21 |

There is a **tendency** for higher SNR to cost exact-feature quality (`r = 0.40` between SNR and
val_mse, `0.47` in log-SNR) -- real but weak, so "shot efficiency and expressivity are two ends of
one knob" would be an overstatement. `s_detuning=4.5`, flagged at the end of the Gate 4 section as
the strongest untested candidate, lands close to `gate4_selected` on both axes and offers nothing
new; it is now measured rather than pending.

**The reason Pass A cannot select a candidate is sharper than the trade-off, and it is in the
three configs whose exact FID was actually measured:**

| config | val_mse | measured exact FID |
|---|---|---|
| `gate4_selected` | 0.66570 | **85.40** |
| `polar_s=27` | 0.72057 | **96.53** |
| `gate03_baseline` | 0.75983 | **90.47** |

**val_mse ranks these wrong.** `gate03_baseline` has the *worst* val_mse of the three and the
*middle* FID; `polar_s=27` has better val_mse and much worse FID. So val_mse does not order exact
FID even in sign once a different knob is involved -- the same dissociation cell 3 showed for
`t_scale`, now across geometry versus detuning.

That is decisive for the method, because the two terms are not comparable in size: the noise
penalty spans **2.3 to 8.2 FID** across the whole sweep, while measured exact FID already spans
**85.4 to 96.5** among just three configs. The small term is the one SNR prices reliably; the large
term has no cheap proxy at all. **So finding a better operating point requires one FID run per
candidate (~40 min at `samples=500`) and no screening step can shorten the list.** Gate 7 ends by
establishing what cannot be done cheaply, which is worth more than another failed shortcut.

### What still stands, and cell 3's replacement

Two things from part 2 are measurements, not estimates, and survive the falsification:

- **A detuning axis exists that multiplies SNR by 3.2x**, driven by signal amplification rather
  than the polarization this gate was premised on. Gate 4 could not have seen it. It is real; it
  just does not help, and now it is known not to help rather than untested.
- **`t_scale=0.25` improves exact-feature val_mse at no SNR cost** (0.64959 vs 0.66570), a
  different mechanism from detuning (it shrinks the timestep channels' share of `h`'s dynamic
  range, the component most shared across probe times -- consistent with Gate 3's multicollinearity
  finding).

Cell 3 was originally `S=27 + t_scale=0.25`, asking whether the two axes add. Cell 1 removed that
question: stacking a second axis onto a config now measured at 99.93 answers nothing. It was
replaced with **`t_scale=0.25` at Gate 4's own `S=9`, at `shots=1000`** -- the one axis still
standing, never checked against FID. Cell 2 (`s27_exact`, predicted 88.35) was left running: it
separates which half of the chain broke, and a failed cell 1 is uninterpretable without it.

**Scheduling note.** Pass A was first run alongside Gate 6's replication queue and the contention
was far worse than the ~280s/config the uncontended estimate suggested: **1596s per config, a 5.7x
penalty**, with both jobs thrashing the same GPU. Finishing the remaining 13 configs that way
would have cost ~5 h of contention *and* pushed Gate 6 out by about the same, so Pass A is instead
chained to start automatically when the Gate 6 driver exits, where it costs ~42 min of exclusive
GPU. This is recorded because the first cost estimate in this section was wrong by 5.7x, and the
reason (GPU contention, not a cache miss rate) is the useful part.

## Seeds 3-5: the `classical` baseline is far noisier than three seeds showed

`experiments/seeds_extension.py` doubled the replication to six data seeds. The digital backdrop
finished first and it invalidates a number this document has leaned on throughout.

**The `classical` arm's seed-to-seed SD is 7.99 over six seeds, not the 2.91 measured over three.**
Its per-seed FIDs are `70.77 / 75.70 / 70.57 / 82.10 / 71.54 / 90.47` -- seeds 0-2 happened to be an
unusually tight sample. The arms are not comparably stable:

| arm | FID (6 seeds) | **FID SD** | val_mse SD |
|---|---|---|---|
| `interaction` | 66.77 | **1.57** | 0.0099 |
| `qrc` digital | 68.69 | **3.43** | 0.0113 |
| `classical` | 76.86 | **7.99** | 0.0133 |
| `random` 312-dim | 105.80 | **8.31** | 0.0130 |

The outlier cells are not defective: `degenerate=False`, `frac_at_clip=0`, `readout_max_diff=0`,
and their `val_mse` (0.5642, 0.5752) sits mid-range among all six. What separates them is
**diversity** -- 0.7199 and 0.7194, the two lowest of the six. The plain classical readout
occasionally generates at reduced diversity, and FID punishes that.

**The val_mse column is the point.** Every arm's supervised MSE is equally stable across seeds
(SD 0.0099-0.0133, a 1.3x range) while their FID stabilities span 1.57 to 8.31, a 5x range. This is
the sharpest MSE/FID dissociation this project has produced: identical supervised stability,
wildly different generation stability. Any future work tempted to screen on val_mse should read
this table alongside Gate 7's cell-3 result.

### What this invalidates

**The "2.91 FID noise floor" quoted repeatedly above underestimates by ~2.7x.** Every calibration
made against it needs re-reading -- in particular, the standard for "an improvement large enough to
demonstrate" against `classical` is of order 15 FID, not the ~5 stated in Gate 7.

**`classical` is a poor reference arm, and pairing does not rescue it** because the variance sits
in that arm specifically: `interaction - classical` was `-6.24 +/- 1.61` over three seeds and is
`-10.09 +/- 8.32` over six. `interaction` (SD 1.57) is the reference that actually discriminates,
and it is also the harder control -- which makes it the one to lead with.

### It does not overturn the main result -- it sharpens it

The `shots=1000` arm is complete over all six seeds:

| seed | rydberg `shots=1000` | `classical` | `interaction` | vs `classical` | vs `interaction` |
|---|---|---|---|---|---|
| 0 | 72.40 | 70.77 | 65.70 | +1.63 | +6.70 |
| 1 | 85.36 | 75.70 | 67.63 | +9.66 | +17.73 |
| 2 | 75.27 | 70.57 | 65.00 | +4.70 | +10.27 |
| 3 | 77.27 | 82.10 | 67.20 | **-4.83** | +10.07 |
| 4 | 83.11 | 71.54 | 69.26 | +11.57 | +13.85 |
| 5 | 79.67 | 90.47 | 65.81 | **-10.80** | +13.86 |
| **mean** | **78.85 +/- 4.86** | 76.86 +/- 7.99 | 66.77 +/- 1.57 | **+1.99 +/- 8.58** | **+12.08 +/- 3.86** |

**This revises the three-seed claim.** Gate 6's verdict above reads "all three seeds unfavorable
[against `classical`] ... the firmest result here". At six seeds that contrast is a **statistical
tie**: `+1.99` with a standard error of `3.50`. Both sign flips happen where **`classical` itself
failed** (82.10 and 90.47, its two lowest-diversity cells), not where the Rydberg arm improved.

**Against `interaction` the result is unambiguous and got tighter**: worse on 6 of 6 seeds, by
`+12.08 +/- 3.86` (standard error 1.58).

Note also that the Rydberg arm under shot noise (SD 4.86) is **more stable across seeds than
`classical`** (SD 7.99). The reference was the unstable component all along.

**The practical conclusion is unchanged and better supported than before**: at the hardware shot
ceiling the Rydberg reservoir buys nothing -- indistinguishable from a plain linear readout, and
clearly behind a structured classical block. What changes is the *claim that can be defended*: not
"consistently worse than `classical`" (three seeds' luck) but "no advantage over `classical`, and a
large deficit against the control that actually discriminates".

### Final result: all arms, six seeds, everything complete

| arm | FID (6 seeds) | SD | vs `interaction` | vs `classical` |
|---|---|---|---|---|
| **`interaction`** | **66.77** | **1.57** | -- | -10.09 +/- 8.32 |
| `qrc` digital | 68.69 | 3.43 | +1.92 +/- 2.85 (5+/1-) | -8.17 +/- 6.57 |
| **rydberg exact** | 69.61 | 3.71 | **+2.85 +/- 3.01** (5+/1-) | -7.24 +/- 7.20 |
| `classical` | 76.86 | 7.99 | +10.09 +/- 8.32 | -- |
| **rydberg `shots=1000`** | 78.85 | 4.86 | **+12.08 +/- 3.86** (6+/0-) | +1.99 +/- 8.58 |
| `random` 312-dim | 105.80 | 8.31 | +39.04 +/- 8.78 | +28.95 +/- 2.95 |

**The port succeeded as a port.** Rydberg exact against the digital reservoir it replaces, paired
per seed: `-0.00, +2.75, +1.45, +0.90, +0.82, -0.36`, mean **`+0.93 +/- 1.11`** (SE 0.45). An
analog Rydberg Hamiltonian with a fixed operating point, no gate set, and data entering through the
per-site detuning reaches the same generation quality as the digital circuit ensemble it was ported
from -- to within one FID, and the digital arm gets 5 draws per seed to the Rydberg arm's one. That
is the positive result of this work and it is clean.

**Neither reservoir beats the structured classical control.** `interaction` -- 100 `x_t (x) te(t)`
products, 120 columns, no physics, 0.4s of generation against 13h -- is the best arm at 66.77 and
the most stable at SD 1.57. Digital `qrc` sits `+1.92` behind it, rydberg exact `+2.85`, both
5-of-6 unfavorable. This is the same conclusion `QRC_FUSION_FID_BIAS_AUDIT.md` reached for the
digital arm, now confirmed for the analog port at six seeds.

**Shot noise costs ~9.2 FID and settles the hardware question.** At `shots=1000` the Rydberg arm
falls to 78.85, `+12.08 +/- 3.86` behind `interaction` on **6 of 6 seeds**. Against `classical` it
is a tie (`+1.99 +/- 8.58`) but that contrast is uninformative for the reason this section
documents.

**Read the two reference columns against each other.** Every contrast against `interaction` (SD
1.57) has a tighter spread than the same contrast against `classical` (SD 7.99), and the
`interaction` estimates converged as seeds were added (`+2.04`, `+2.01`, `+2.17`, `+2.85` for the
exact arm) while the `classical` ones swung (`-4.20`, `-6.39`, `-5.01`, `-7.24`). Lead with
`interaction`: it is simultaneously the harder control and the one that discriminates.

## Reproducing the results on QuEra's own emulator (bloqade-analog)

Every number in Gates 3-7 was produced by `quera/emulator.py`, this project's own RK4 statevector
propagator. For a paper the physics should be attributable to the vendor's emulator, not to a
homegrown one, so the pipeline was made to run against **bloqade-analog** -- QuEra's own software --
with nothing else changed.

**The split is drawn at the Schrodinger solve and nowhere else.** `quera/_bloqade_worker.py` (in
`.venv-gate1`, where bloqade lives) returns raw `|psi|^2` populations; every downstream step --
the 78 Z/ZZ observables, shot sampling, vacancy filtering, the ridge readout, the DDIM rollout,
PRDC and FID -- stays in the code that produced the existing results. That matters for what can
be claimed: this is **one pipeline with the solver swapped**, not two implementations that
happened to agree. Agreement at the feature level propagates to FID by construction, because
everything after `rydberg_features` is literally the same code.

`quera/bloqade_backend.py` is the production-side handle (a subprocess over pipes, the same
cross-virtualenv pattern Gate 1 already used, so the production environment still never imports
bloqade); `backend='torch'|'bloqade'` on `ReservoirParams` and `--backend` on
`qrc_fusion_fair_generation.py` select it. **`backend` is part of the feature cache key** -- without
that a bloqade run would silently reuse torch-computed features and the whole comparison would be
circular while failing no test.

### The one thing that could go wrong silently

bloqade's `state_vector.space.configurations` puts **site 0 at the least significant bit**; this
project's `emulator.py` puts it at the **most significant** (see Gate 1, where all three
implementations' conventions were established empirically). A wrong reversal yields a permuted but
entirely plausible state -- normalised, in range, failing no assertion. The reversal is explicit in
the worker and pinned by `tests/test_bloqade_backend.py`, which compares the **312 production
features** (not raw populations) and holds the per-site `<Z_i>` block to a tighter `5e-5` precisely
because that is where a permutation would surface. Those tests run with `use_cache=False`
deliberately: with the cache on they passed in 0.7s without invoking either solver, which would
have kept them passing after any change to the worker.

### Results

Same cell, same seed, same protocol, solver swapped:

| cell | `quera/emulator.py` | **bloqade-analog** | difference |
|---|---|---|---|
| `samples=500` (Gate 4 point) | 85.401802 | **85.406498** | **+0.0047** |
| `samples=10000` (Gate 6 seed 0) | 63.781992 | **63.782115** | **+0.000123** |

At full scale the two emulators disagree by **0.0001 FID**, against the **2.91 FID** seed-to-seed
SD Gate 6 measured -- roughly **24,000x inside the experiment's own noise**. `val_mse` matches to
the sixth decimal and the ridge selects the same `lambda` in both. Agreement is *tighter* at
`samples=10000` than at `500`, as expected: FID over more samples averages away per-sample feature
differences that a 500-sample estimate still carries.

Across all three seeds of the exact arm the two solvers never disagree by more than **0.0013 FID**
(`+0.000123`, `+0.001327`, `+0.000047`), and the 3-seed means coincide to three decimals:
`68.148 +/- 4.999` (torch) against `68.148 +/- 5.000` (bloqade).

Raw populations were checked directly too, on production programs: `max|dprob| ~ 3e-6` and
`max|d<n_i>| ~ 2e-5`, consistent with the `~1e-5` floor Gate 1 measured *between the two official
reference simulators themselves*. The residual is the known disagreement between independent
Rydberg AHS implementations, not an error on either side.

### The `shots=1000` arm does not reproduce to the same precision, and should not

Predicted here, before measuring, that the sampled arm would agree as tightly as the exact one --
the reasoning being that `sampling.py` draws from the populations with an identically seeded
generator, so identical populations give identical bitstrings. **The premise was false: the
populations are not identical**, they differ at `~3e-6`, which is enough to move some draws across
a multinomial category boundary. And the DDIM rollout is a 50-step feedback loop, so one flipped
bitstring at step 1 changes `x` at step 2, hence `h`, hence everything after it. Measured on seed 0:

| metric | torch | bloqade | difference |
|---|---|---|---|
| FID | 72.265 | 72.535 | **+0.269** |
| recall | 0.460 | 0.526 | +0.066 |
| coverage | 0.381 | 0.419 | +0.038 |
| val_mse | 0.620408 | 0.620762 | +0.000354 |

The readout is barely touched (`val_mse` differs at the fourth decimal, same selected `lambda`);
the divergence is in the rollout, as the chaotic-amplification account predicts.

**Seed 2 settles that this is realization noise and not a solver bias: its difference is
`-0.2457`, the opposite sign.** Over the three seeds the differences are `+0.2695`, `+0.2497`,
`-0.2457` -- mean `+0.09`, mean absolute `0.25`. A systematic error in either solver would keep its
sign; this does not. Arm means: `77.630 +/- 6.766` (torch) against `77.721 +/- 6.846` (bloqade), a
0.09 FID gap, **3% of one seed-to-seed SD**.

**This is not solver disagreement, it is two realizations of the same stochastic process**, and it
yields a number nobody had measured: the **shot-realization noise of a `shots=1000` cell is
~0.25 FID**. Against the quantities it has to be read next to -- a seed-to-seed SD of **2.91**, a
noise penalty of **9.48**, a deficit versus `classical` of **+5.29** -- it is 3-9% and changes no
conclusion. It is also a useful decomposition for a write-up: the sampled arm's variance is
dominated by the train/val seed, not by the shot draw.

### Cost, and a correction to this document's own figure

The full-scale bloqade cell took **7.6 h on 20 CPU cores against 13.0 h for the GPU path** -- the
vendor emulator on CPU is **1.7x faster** than `quera/emulator.py` on an RTX 2080 Ti, plausibly
because bloqade's adaptive `dop853` takes far larger steps than our fixed-step RK4 at
`rk4_safety=0.2`.

**The "~8s per CPU sample" figure this document used to justify building a custom emulator was
measured at Gate 1's tightened tolerances, not at production settings.** At production accuracy the
real cost is **0.139s per program**. That error made an end-to-end vendor-emulator run look like
185 days when it is 7.6 h, and it stood unexamined until it was measured. The custom emulator was
still needed for the batched GPU rollout that Gates 3-7 were developed against, but the claim that
bloqade was infeasible for production runs was wrong.

Memory, not cores, is the binding constraint: bloqade grows from ~250MB to 700MB+ RSS per worker
over a few thousand solves (it appears to cache compiled programs, and `h` differs every task), so
22 workers with loose recycling took a 15.7GB machine to 400MB free in four minutes. `20` workers
with `BLOQADE_MAXTASKS=50` sits on a flat ~6GB-available plateau. Keeping workers alive longer was
tried as a speedup and measured as none (~78% of the pure-solve floor either way).

### Status

`experiments/bloqade_replication.py` has drained: **all six full-scale cells (both arms x 3 seeds)
exist in both solvers.** The vendor-emulator replication is complete for Gate 6 as published.
`experiments/seeds_extension.py` extends to seeds 3-5 -- Rydberg cells there run on bloqade only,
since the equivalence no longer needs re-demonstrating.

## Environment

Two virtualenvs, both Python 3.12, neither touches the machine's existing conda envs:

- `.venv` -- the production environment. `torch` (CUDA, verified against the local RTX
  2080 Ti), `scikit-learn`, `scipy`, `pandas`, `numpy`, `pytest`. This is what
  `experiments/*.py` and `quera/*.py` run under, and what CI-equivalent test runs use.
- `.venv-gate1` -- Gate 1 only. Adds `bloqade-analog` and `amazon-braket-sdk` on top of the
  same base. **`numpy` must be pinned to `2.1.3`** in this env: `bloqade-analog==0.16.9`
  crashes on import against `numpy>=2.5` (`beartype.roar.BeartypeDecorHintNonpepNumpyException`
  from a `beartype` type-hint check on a numpy scalar-dtype API that changed upstream). This
  is a `bloqade-analog`/`beartype` compatibility bug, not a project defect; pinning numpy in
  the isolated Gate 1 env is the workaround and does not touch the production env's numpy.
  `amazon-braket-sdk` has no such constraint.

Both packages installed and imported successfully on Python 3.12 in one shared env once
numpy was pinned -- the two-interpreter fallback (3.13 main / 3.11 for Gate 1) the task
anticipated was not needed.

## Gate 1 dependency probe (before building the emulator)

Two things had to be checked before trusting Gate 1's 1e-6 tolerance is even a coherent
target: whether each reference can return *exact* expectation values (not just shot
samples), and whether the two references agree with each other once given the same
physical input.

**Exact state access requires bypassing both libraries' public, documented entry points:**

- `bloqade-analog`'s `BloqadePythonRoutine.run(shots=...)` is shot-based only. The exact
  final state vector is reachable through `run_callback`, which hands the emulator's
  internal `StateVector` object to a user callback before any sampling happens. This is
  the documented extension mechanism, not a private API, but it is the less-traveled path.
- `amazon-braket-sdk`'s public surface for the local AHS simulator is
  `LocalSimulator("braket_ahs").run(program, shots=...)`, which also only returns sampled
  shots -- `RydbergAtomSimulator.run()` (in
  `braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator`) computes the full
  state-vector trajectory internally (via `numpy_solver.rk_run` for <=1000 basis
  configurations, `scipy_solver.scipy_integrate_ode_run` above that) and then discards it,
  keeping only `states[-1]` long enough to sample from. There is no public flag to get the
  state back; Gate 1's test calls `rk_run`/`scipy_integrate_ode_run` directly, replicating
  the four lines of unit conversion and validation `RydbergAtomSimulator.run()` does before
  handing off to the solver. **This is a private-API dependency** -- an
  `amazon-braket-sdk` upgrade could rename or restructure these internals without notice.
  Flagged here rather than discovered silently when Gate 1's test starts failing on an
  unrelated dependency bump.

**The two references do not agree on the Rydberg C6 constant.** `amazon-braket-sdk`'s
default (`RydbergAtomSimulator.RYDBERG_INTERACTION_COEF`) is a rounded
`5.42e-24` rad/s.m^6 (`5.42e6` rad/us.um^6). `bloqade-analog`'s hardcoded constant
(`bloqade.analog.constants.RB_C6 = 2*pi*862690`) is `~5.420453e6` rad/us.um^6 -- which *is*
the task spec's own defining formula (`862690 x 2pi MHz.um^6`) at full precision, confirming
`quera.device.C6_RAD_US_UM6 = 5.4203e6` is the right constant to build the emulator against.
Comparing the two references at their respective defaults would fail Gate 1 on a ~1.5e-4
relative C6 mismatch that has nothing to do with either emulator's correctness. **Gate 1
must explicitly pass a matched interaction coefficient into both**: bloqade has no override
(the constant is hardcoded, so bloqade is always run at its own value, which matches the
spec), and braket's `rydberg_interaction_coef` kwarg is set from `quera.device.C6_RAD_US_UM6`
converted to SI at the call site.

**Numerical cross-check.** A toy 2-atom program (10 um apart -- close enough that the C6
term is comparable to Omega, so this actually exercises the interaction term, not just the
drive) was run through both references with the C6 mismatch corrected. `bloqade`'s adaptive
`dop853` solver (`atol=1e-7`, `rtol=1e-14`) and `braket`'s fixed-step RK6 agree to ~1e-4 at
1,000 time steps and converge monotonically toward each other as the braket step count
increases (~5e-6-9e-7 agreement at 100,000 steps over a 1.1 us program, ~32s wall time) --
consistent with two independent implementations of the same physics, differing only in
integration discretization. For n>=10 sites the public `RydbergAtomSimulator.run()` path
auto-switches to the adaptive `scipy` solver (`len(configurations) > 1000`, i.e.
`2**n_sites > 1000`), which reaches tight agreement fast (~3s for the real 12-atom cluster)
without the fixed-step step-count cost the 2-atom toy case needed -- confirmed on the actual
cluster in Gate 1 below, not just assumed from the toy program.

**Site-index convention differs between every pair of the three implementations**, verified
empirically (never assumed) before trusting any cross-comparison:
`bloqade`'s `state_vector.space.configurations` integers put site 0 at the *least*
significant bit (from `AtomType.integer_to_string`'s `state_int % n_level` peeling off site 0
first); `braket`'s `get_blockade_configurations` returns strings with site 0 at *character
index 0*; this project's own `emulator.py` puts site 0 at the *most* significant bit
(matching `denoiser_qrc.step`'s `idx >> (n-1-q)` convention). All three agree that "site 0" is
the first row of `positions_um` / first `add_position`/`add(...)` call, so the physical
observable `<n_i>` for a given atom is unambiguous even though the three raw amplitude
vectors are permutations of each other. `tests/_gate1_reference_worker.py` computes each
reference's `<n_i>`/`<n_i n_j>` using that reference's own convention rather than trying to
align raw amplitude vectors -- confirmed correct by an asymmetric two-atom check (see
Gate 1 section) where bloqade and braket's `|rg>`/`|gr>` populations matched to 4 decimal
places before any tolerance tuning.

## What is not portable from the digital reservoir, and why

- **The `clifford`/`alpha_dial`/`doped`/`haar` circuit families** (`magicqrc/circuits.py`).
  These are discrete-gate constructions (Clifford products, T-gate doping, Haar sampling)
  built for a digital, matrix-product gate model. Aquila has no single-qubit or two-qubit
  gate primitive to compile them onto -- it exposes one fixed-form global/local analog
  Hamiltonian (drive + detuning + van der Waals), not a gate set. The "magic dial" concept
  the families implement (dynamics-generated magic at fixed depth) has no analog on a
  device whose only tunable dynamics are continuous pulse shapes.
- **Encoding the data in the initial state.** `denoiser_qrc._input_density` prepares
  `rho_in` from `x_t` and evolves it forward; the *dynamics* are data-independent. On
  Aquila the initial state is always `|gg...g>` by hardware fixed-fact -- there is no
  state-preparation channel. Data has to enter the *generator* of the dynamics instead
  (the per-site detuning pattern `h_j`), which is why `encoding.py` maps `x_t` to `h`
  rather than to a density matrix. This is a qualitatively different map, not a
  reparametrization of the same one: the nonlinearity source moves from the tensor product
  (`_input_density`'s docstring) to the matrix exponential.
- **Memory qubits.** `denoiser_qrc.py`'s docstring notes `n_data` is fixed at 5 by the
  quadrature encoding while `n_qubits=6`, so one qubit is a memory qubit carrying state
  across calls when the reservoir isn't reset. The Rydberg port is stateless by
  construction (`rollout(..., reset_every_step=True)` already made the digital arm
  stateless too, ref: `denoiser_qrc.py:_input_density` docstring and
  `qrc_fusion_fair_core.py:rollout`'s docstring) and every atom is a data or timestep
  channel -- 12 in, 12 used, nothing held in reserve. There is no notion of a qubit that
  is *not* re-prepared each probe-time program, since each probe time is already its own
  from-scratch program (`pulses.py`'s module docstring).

## Gate 0

`quera/device.py` centralizes every numeric constant from the task's Aquila spec table
(nowhere else in `quera/` may define one -- `layout.py` and `pulses.py` both import from
it). `quera/pulses.py` builds one independent ramp-up/plateau/ramp-down program per probe
time, quantized to the 1 ns time grid and 10 nm position grid *before* validation (never
after -- rounding a value that already passed a check can silently reintroduce the
violation it just cleared). `tests/test_device_validation.py` (16 tests) proves the five
required failure modes -- missing ramp, 30 ns segment, 5 us duration, 3 um spacing,
positive `Delta_local` -- plus range/quantization/shot-count checks, all fail closed. A
slew-limit test needed to call `_validate_omega` directly rather than go through the full
`validate()`: at `MIN_SEGMENT_US=50ns` and `OMEGA_MAX=15.8`, the worst-case slew
(`15.8/0.05=316 rad/us^2`) is under `OMEGA_SLEW_MAX=400`, so a program that satisfies the
minimum-segment constraint can never violate the slew constraint -- the two limits are
jointly consistent by construction, not by coincidence, and that fact is worth keeping in
mind for Gate 4's `V` sweep (shorter probe times still get a full 50 ns ramp, they just
leave less room for the plateau).

`quera.device.blockade_radius`/`pair_coupling` were checked against all five of the task
spec's stated derived numbers (`R_b(2pi)=9.76um`, `R_b(15.8)=8.37um`,
`J(15)=0.48, J(20)=0.085, J(22)=0.048 rad/us`) and match to the spec's stated precision,
confirming `C6_RAD_US_UM6=5.4203e6` is being used consistently with the rest of the table.

## Gate 1

`quera/emulator.py` propagates `|gg...g>` as a batched torch statevector (`(batch, 2^n)`,
no density matrix -- unitary, no dissipation, fixed initial state). The drive term is
applied without assembling a matrix: flipping qubit `q` is a fixed index permutation
(`torch.gather`), and the diagonal (detuning + van der Waals) is built once per program and
reused at every integration step, because `Delta_global`/`Delta_local` are held constant for
an entire probe-time program (`pulses.py`) -- only `Omega(t)` moves within a program. RK4
was chosen over Krylov/Lanczos for a simpler, fully-batched implementation (a Gershgorin
bound on `||H||` picks the sub-step count per pulse segment; no subspace-size hyperparameter
to tune); `tests/test_emulator.py` checks this two ways that need no optional dependency:

- **Free correctness test**: at zero detuning/interaction and `phi=0`, `H(t)` is
  proportional to the *same* fixed operator (`sum_q X_q`) at every instant, so the
  time-ordered exponential has a closed form -- a per-qubit rotation by the pulse's area/2,
  giving an exact target statevector with no reference simulator involved. Matches to
  `2.2e-6` at the default step size, `8e-9` at 4x finer steps -- this isolates the
  drive/index-permutation half of the matvec from the diagonal half.
- **Step-halving convergence**: halving the RK4 safety threshold shrinks the change in the
  final state by >8x per halving, consistent with RK4's `dt^4` global-error scaling.

`quera/program.py` exports a `Program` to `bloqade.analog` (native rad/us, um units) and to
`braket.ahs.AnalogHamiltonianSimulation` (SI units) -- lazily importing both, so
`quera.program` stays importable with neither installed.

**`tests/test_emulator_vs_reference.py`** compares `<n_i>` and `<n_i n_j>` from all three
implementations on 20 random programs on the real 12-atom, `a=8um` cluster (random `h` in
`[0,1]^12`, random plateau duration in `[0.1, 1.0]us`) -- not a toy system. It shells out to
`.venv-gate1` via `tests/_gate1_reference_worker.py` (a subprocess, not an import) so the
production `.venv` never needs bloqade/braket installed. Result:

```
worst-case disagreement over 20 points:
  mine vs bloqade-analog:  9.3e-6
  mine vs braket_ahs:      4.3e-6
  bloqade-analog vs braket_ahs: 1.09e-5
```

This is the two references checked against each other first, per the spec's "if the two
references disagree, stop and report" -- they agree to `1.09e-5`, so Gate 1 proceeds; that
number, not either individual comparison to this emulator, is the actual test of whether the
cross-validation is meaningful at all. **The task's aspirational `1e-6` tolerance was not
reached** -- the achieved figures are one order of magnitude looser, in the low `1e-5`
range. This was chased down, not accepted at first measurement: matching the C6 constant
(above) was necessary and improved agreement from ~1e-4 to ~1e-5 on the 2-atom probe case;
tightening both references' solver tolerances an order of magnitude past their defaults
(bloqade `atol` 1e-7->1e-10, braket `atol`/`rtol` 1e-8/1e-6->1e-10/1e-10, braket's internal
time grid 1000->4000 points) on a spot-checked worst-case point left the disagreement
essentially unchanged (~5e-6 before and after). That is evidence the residual is not a
solver-tolerance artifact on either side, but the practical floor between two independently
implemented Rydberg AHS simulators at this problem size -- plausibly a smaller residual
physical-constant or unit-rounding difference neither library exposes as a settable
parameter. `tests/test_emulator_vs_reference.py` asserts `2e-5` (with margin over the
measured `1.09e-5`/`9.3e-6`), documented in the test file rather than silently substituted
for the spec's number.

Site-index convention (above) was verified, not assumed, before any of these comparisons
were trusted: confirmed with an asymmetric two-atom program where bloqade's and braket's own
labeled outputs (`|rg>`/`|gr>`, `"rg"`/`"gr"`) agreed with each other to 4 decimal places
before any tolerance tuning, which also incidentally pre-validated that the two references
agree with each other on a case simple enough to hand-check.

**Known follow-up, out of scope for Gate 1**: a batched GPU call with `batch=2048` on the
production cluster geometry took ~267s (vs. ~8s for a single CPU sample at the same
accuracy) -- the RK4 loop's per-step Python overhead (12 sequential `torch.gather` calls per
stage, hundreds of stages per program) dominates over the actual batched tensor work, so the
GPU's parallelism is not being exploited. Fine for Gate 1's single-program checks; will need
addressing (e.g. vectorizing the per-qubit gather loop, or precomputing a single combined
permutation+phase tensor) before Gate 3's supervised sweep, which needs thousands of
programs.

## Gate 2

`quera/sampling.py` draws noisy bitstrings from `|psi|^2` (batched `torch.multinomial`,
decoded with this project's own MSB-first site convention -- self-consistent within
`sampling.py`/`features.py`, never compared to bloqade's/braket's raw indices, see the Gate 1
site-convention note). Three independently switchable noise sources, all default off:
shot noise (inherent to finite-sample counting), per-site vacancy (`~1%`, filtered **per
cluster** -- a shot is usable only if all 12 atoms survived, matching the spec's own
`0.99**12=88.6%` worked number, checked directly in
`test_vacancy_filtering_matches_spec_number` rather than trusted), and detection error
(false-positive/false-negative rates applied separately, since they act on disjoint true-bit
populations). `quera/features.py` computes the 78 `<Z_i>`/`<Z_i Z_j>` observables
(`Z=1-2n`, `|g>=+1`, matching the qubit convention with `|r>` playing `|1>`) two ways: exact,
from the statevector, and sampled, by counting from noisy bitstrings -- "igual ao hardware"
per the spec.

**`tests/test_shot_convergence.py`** fixes one program/statevector and draws 300 independent
Monte Carlo trials at each `shots in {50, 100, 300, 1000, 5000}` (the 300 trials are extra
batch rows of the *same* `psi`, so this is one vectorized `sample_bitstrings` call per shot
count, not a Python loop over trials). Measured:

```
S=   50  empirical_std=0.13100  std*sqrt(S)=0.9263  mean_bias=0.00631
S=  100  empirical_std=0.09288  std*sqrt(S)=0.9288  mean_bias=0.00493
S=  300  empirical_std=0.05371  std*sqrt(S)=0.9303  mean_bias=0.00254
S= 1000  empirical_std=0.02906  std*sqrt(S)=0.9188  mean_bias=0.00127
S= 5000  empirical_std=0.01334  std*sqrt(S)=0.9430  mean_bias=0.00051
```

`std*sqrt(S)` stays within `0.92-0.94` across three orders of magnitude in `S` -- a clean
`1/sqrt(S)` law, close to the task spec's own `sigma(<Z_i>)~=1/sqrt(S)` estimate (the
observed constant is slightly under 1 because the 78-feature average includes the `<Z_i Z_j>`
correlators, whose variance differs somewhat from a single Z's). The log-log slope
(`-0.5` predicted) is asserted, not just "decreasing", and `mean_bias` shrinking with `S`
(not just `empirical_std`) confirms the sampler is unbiased, not just low-variance. `S=5000`
exceeds `device.MAX_SHOTS=1000` -- deliberately: it represents shots aggregated over several
`<=1000`-shot hardware tasks, which this counting-statistics test doesn't need to distinguish
from a single task, not an oversight of the per-task shot limit (`pulses.py`/`device.py`
still enforce it for anything that becomes a submittable program).

`quera.features.physical_variance_floor` (`sigma^2~=(1-<O>^2)/shots`) is implemented and
tested but not wired into anything yet -- it is the correct floor for counting-based
features, documented against the digital arm's `VARIANCE_FLOOR=1e-6` (calibrated for exact
observables, per `QRC_FUSION_FID_BIAS_AUDIT.md`) without touching that constant, per the
task's hard rule not to change the digital arm's behavior.
