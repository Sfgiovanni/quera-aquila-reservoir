# QRC---Diffusion: the "fair" fusion comparison

Bias-corrected comparison of a QRC-reservoir-augmented ridge readout against
plain classical baselines for one-step diffusion denoising, on Fashion-MNIST
and BreastMNIST latents.

This repo mirrors the earlier `qrc_classical_feature_fusion` study
(`ridge_classical` vs `ridge_classical_qrc_B`) after two mechanical defects
were found and fixed: a train/generation state mismatch that made 11 of 15
hybrid FID cells collapse to a single repeated image, and a 1e-10 variance
floor that dominated the reported MSE gap. See
[`QRC_FUSION_FID_BIAS_AUDIT.md`](QRC_FUSION_FID_BIAS_AUDIT.md) for the full
writeup, before/after tables, and the corrected conclusion.

## Layout

- `experiments/qrc_fusion_fair_core.py` — shared readout fitting / balanced
  pairing / GPU readout code.
- `experiments/qrc_fusion_fair_supervised.py` — supervised MSE comparison
  (the stage that found the classical `x_t ⊗ te(t)` interaction block
  matches the QRC arm).
- `experiments/qrc_fusion_fair_generation.py` — end-to-end sample generation
  and FID computation per (dataset, method, seed, draw, n_train) cell.
- `experiments/qrc_fusion_fair_analyze.py`, `qrc_fusion_fair_samples.py` —
  aggregation and sample-grid plotting.
- `run_qrc_fusion_fair.sh`, `run_breast_ntrain_sweep.sh` — the sweeps used to
  produce `results/qrc_fusion_fair/`.
- `results/qrc_fusion_fair/` — parquet/json outputs (per-cell FID, paired
  generation stats, supervised summary).
- `figures/qrc_fusion_fair/` — sample grids from the corrected runs.
- `autoencoder.py`, `data.py`, `ddpm.py`, `denoiser_classical.py`,
  `denoiser_qrc.py`, `latent_scaling.py`, `metrics.py`,
  `experiments/phase4.py`, `experiments/qrc_kernel_core.py`,
  `experiments/time_injection.py`, `magicqrc/` — runtime dependencies needed
  to actually run the two scripts above.
- `checkpoints/autoencoder_d10.pt`,
  `checkpoints/breastmnist_sweep/B_dim10_hflip_600ep_seed0.pt` — the trained
  autoencoders used to encode/decode latents.
- `results/fmnist_classifier.pt` — small classifier used for Fashion-MNIST
  FID feature extraction.

Not included: the rest of this project's experiments (phase0–phase5 sweeps,
the earlier biased `qrc_classical_feature_fusion` study, the VQC/time-injection
tracks, etc.) — this repo is scoped to the fair comparison only. A cached
`results/magic_fid_reference_features_n10000.npz` (~71MB) that speeds up the
Fashion-MNIST reference FID pass was also left out; the code recomputes it on
first run if absent.

## Reproducing

Needs a Python env with `torch`, `scikit-learn`, `scipy`, `pandas`, `numpy`.

```bash
./run_qrc_fusion_fair.sh
./run_breast_ntrain_sweep.sh
python -m experiments.qrc_fusion_fair_supervised
python -m experiments.qrc_fusion_fair_analyze
```
