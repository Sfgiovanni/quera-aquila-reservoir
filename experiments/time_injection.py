"""Does a reservoir that sees the diffusion timestep beat ridge + interaction?

The diagnosed gap (ESTADO_DO_PROJETO.md 2.3): `features()` builds cat([x_t, h, te]) and the
reservoir never sees t, so h = h(x_t) is time-independent and a linear readout over concatenated
blocks can only produce an eps_hat that is ADDITIVE in the timestep. The correct denoiser needs the
coefficient on x_t to vary with t. Measured: x (x) te alone recovers 99.4% of an MLP's advantage,
while the reservoir contributes +0.00027 without it and +0.00006 with it.

Hypothesis: a reservoir's nonlinear dynamics mixes whatever is injected into it, so a reservoir
receiving both x_t and t should generate the interaction natively. That is the one capability a
block-concatenated ridge structurally lacks.

Honest prior, recorded in DECISIONS_time_injection.md before running: ridge + interaction already
captures 99.4% of the available gain at 1200 parameters, so even a perfect native interaction
leaves little headroom. A small effect or a clean null is expected. The value is settling whether
the reservoir's redundancy is architectural or fundamental.

Decisive comparison: qrc_t_injected_A vs ridge_interaction.

Controls, each from a documented failure:
  * unitary draw crossed with data seed (draw contributes SD 1.12 vs 0.30 for the seed);
  * select and report on FID -- val MSE is reported but non-predictive (4 of 5 cases anti-transferred);
  * one autoencoder checkpoint for every arm, paired within run (AE is not reproducible across runs);
  * the t-channel scale is swept on disjoint seeds, not guessed;
  * the Gaussian gate and the FID floor accompany every number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge

from autoencoder import decode_numpy, encode_numpy, load_autoencoder
from data import load_breast_mnist, load_fashion_mnist
from ddpm import cosine_schedule
from denoiser_classical import sinusoidal_embedding
from experiments.phase4 import eps_data, features, sample_base, sample_qrc
from metrics import (
    fid_null_floor, fmnist_probs, frechet_distance_lowrank, inception_features_and_probs,
    inception_score,
)

SEED0 = 20260802
LATENT_DIM = 10
DIFFUSION_STEPS = 200
DDIM_STEPS = 50
LATENT_CLAMP = 0.5
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)

# arm -> (uses_reservoir, uses_interaction, time_mode, n_qubits)
ARMS = {
    "ridge_plain":                  (False, False, None,      None),
    "ridge_interaction":            (False, True,  None,      None),
    "qrc_current":                  (True,  False, None,      6),
    "qrc_t_injected_A":             (True,  False, "input",   7),
    "qrc_t_injected_B":             (True,  False, "unitary", 6),
    "qrc_t_injected_A_interaction": (True,  True,  "input",   7),
}


def interact(x, te):
    return (x[:, :, None] * te[:, None, :]).reshape(len(x), -1)


class Readout:
    """Ridge on an optionally interaction-expanded design, with sklearn's .predict() contract so
    the validated samplers in experiments/phase4.py drive every arm unchanged. The expansion runs
    inside predict(), which keeps train and sample paths building identical columns."""

    def __init__(self, model, use_interaction):
        self.model, self.use_interaction = model, use_interaction

    @staticmethod
    def expand(design, use_interaction):
        if not use_interaction:
            return design
        x, te = design[:, :LATENT_DIM], design[:, -10:]
        return np.column_stack([design, interact(x, te)])

    def predict(self, design):
        return self.model.predict(self.expand(design, self.use_interaction))


def fit(dtrain, y, dval, yv, use_interaction):
    a, b = Readout.expand(dtrain, use_interaction), Readout.expand(dval, use_interaction)
    best = None
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(a, y)
        sc = float(np.mean((m.predict(b) - yv) ** 2))
        if best is None or sc < best[0]:
            best = (sc, m)
    return Readout(best[1], use_interaction), best[0], int(best[1].coef_.size)


def _knn_radius(x, k):
    d = np.linalg.norm(x[:, None] - x[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]


def prdc(real, fake, k=5):
    r_rad, f_rad = _knn_radius(real, k), _knn_radius(fake, k)
    d = np.linalg.norm(fake[:, None] - real[None], axis=2)
    return (float((d <= r_rad[None, :]).any(axis=1).mean()),
            float((d.T <= f_rad[None, :]).any(axis=1).mean()),
            float((d <= r_rad[None, :]).sum(axis=1).mean() / k),
            float((d.min(axis=0) <= r_rad).mean()))


def energy(a, b, size=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = a[rng.choice(len(a), min(size, len(a)), replace=False)]
    y = b[rng.choice(len(b), min(size, len(b)), replace=False)]
    return float(2 * np.linalg.norm(x[:, None] - y[None], axis=2).mean()
                 - np.linalg.norm(x[:, None] - x[None], axis=2).mean()
                 - np.linalg.norm(y[:, None] - y[None], axis=2).mean())


def mde(sd, n, power=0.8, alpha=0.05):
    """Minimum detectable effect for a paired t-test: what a null here can and cannot exclude."""
    return float((stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(power, n - 1)) * sd / np.sqrt(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breastmnist", choices=("breastmnist", "fashionmnist"))
    parser.add_argument("--stage", default="main", choices=("scale", "main"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--t-scales", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--t-scale", type=float, default=1.0)
    parser.add_argument("--prdc-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--autoencoder", default=None)
    args = parser.parse_args()
    fashion = args.dataset == "fashionmnist"
    samples = args.samples or (10000 if fashion else 1000)
    repeats = 1 if fashion else 100
    ae_path = args.autoencoder or ("checkpoints/autoencoder_d10.pt" if fashion
                                   else "checkpoints/breastmnist_sweep/B_dim10_hflip_600ep_seed0.pt")
    tag = f"_{args.dataset}_{args.stage}"
    # Stage 1 selects t_scale on seeds/draws disjoint from stage 2, so the scale is not chosen on
    # the evaluation seeds. Declared in DECISIONS_time_injection.md before running.
    base_seed = 100 if args.stage == "scale" else 0

    if fashion:
        train_x, _, test_x, _ = load_fashion_mnist("data/fashion-mnist/raw")
        train_x = train_x[:20000]
        ae = load_autoencoder(ae_path, args.device)
        z = encode_numpy(ae, train_x); train_z, val_z = z, z[-5000:]
    else:
        (train_x, _), (val_x, _), (test_x, _) = load_breast_mnist()
        ae = load_autoencoder(ae_path, args.device)
        train_z, val_z = encode_numpy(ae, train_x), encode_numpy(ae, val_x)

    _, alpha_bar = cosine_schedule(DIFFUSION_STEPS)
    f_test, _ = inception_features_and_probs(test_x, batch_size=128, device=args.device)
    f_train, _ = inception_features_and_probs(train_x, batch_size=128, device=args.device)
    mu_f = f_test.mean(0)
    basis = np.linalg.svd(f_test - mu_f, full_matrices=False)[2][:32]
    rsub = np.random.default_rng(0).choice(len(f_test), min(args.prdc_size, len(f_test)), False)
    proj_real = (f_test[rsub] - mu_f) @ basis.T

    sigma = float(train_z.std())
    x0_clip = LATENT_CLAMP / sigma
    dz = np.repeat(train_z / sigma, repeats, axis=0)
    dvz = np.repeat(val_z / sigma, repeats, axis=0)
    noise = np.random.default_rng(SEED0 + 999).normal(size=(samples, LATENT_DIM))
    mean_z, cov_z = train_z.mean(0), np.cov(train_z, rowvar=False)
    floor = fid_null_floor(f_test, f_train, samples)
    print(f"{args.dataset} stage={args.stage}: sigma={sigma:.4f} x0_clip={x0_clip:.3f} "
          f"rows={len(dz)} samples={samples}", flush=True)
    print(f"FID floor {floor['fid_floor_mean']:.2f} +/- {floor['fid_floor_sd']:.2f}", flush=True)

    def score(images, latents, **meta):
        gen, _ = inception_features_and_probs(images, batch_size=128, device=args.device)
        gsub = np.random.default_rng(1).choice(len(gen), min(args.prdc_size, len(gen)), False)
        p, r, d, c = prdc(proj_real, (gen[gsub] - mu_f) @ basis.T)
        row = {"fid": frechet_distance_lowrank(f_test, gen), "precision": p, "recall": r,
               "density": d, "coverage": c, "latent_energy": energy(train_z, latents),
               "latent_sd": float(latents.std(0).mean()), **meta}
        if fashion:
            probs = fmnist_probs(images, "results/fmnist_classifier.pt",
                                 batch_size=1024, device=args.device)
            row["is_fmnist"], row["is_sd"] = inception_score(probs)
        return row

    rows = []
    scales = ([float(v) for v in args.t_scales.split(",")] if args.stage == "scale"
              else [args.t_scale])
    # Only reservoir arms depend on the unitary draw; the ridge arms are evaluated once per seed.
    arms = (["qrc_t_injected_A"] if args.stage == "scale"
            else [a for a, spec in ARMS.items() if spec[0]])

    for s in range(args.seeds):
        seed = SEED0 + base_seed + s
        x, eps, t = eps_data(dz, alpha_bar, seed)
        xv, epsv, tv = eps_data(dvz, alpha_bar, seed + 100)
        te = sinusoidal_embedding(torch.as_tensor(t), 10, DIFFUSION_STEPS).numpy()
        tev = sinusoidal_embedding(torch.as_tensor(tv), 10, DIFFUSION_STEPS).numpy()
        plain, plainv = np.column_stack([x, te]), np.column_stack([xv, tev])

        if args.stage == "main":
            rng = np.random.default_rng(2000 + s)
            g = np.clip(rng.multivariate_normal(mean_z, cov_z, samples), -LATENT_CLAMP, LATENT_CLAMP)
            rows.append(score(decode_numpy(ae, g), g, arm="gaussian", mechanism="-", seed=s,
                              unitary_draw=-1, t_scale=np.nan, n_qubits=0, n_params=0,
                              val_mse=np.nan, wall_clock_s=0.0))
            print("TIMEINJ " + json.dumps(rows[-1]), flush=True)
            for arm in ("ridge_plain", "ridge_interaction"):
                started = perf_counter()
                model, vm, npar = fit(plain, eps, plainv, epsv, ARMS[arm][1])
                lat = sample_base(noise, model, alpha_bar, args.device,
                                  steps=DDIM_STEPS, x0_clip=x0_clip)
                enc = lat * sigma
                rows.append(score(decode_numpy(ae, enc), enc, arm=arm, mechanism="-", seed=s,
                                  unitary_draw=-1, t_scale=np.nan, n_qubits=0, n_params=npar,
                                  val_mse=vm, wall_clock_s=perf_counter() - started))
                print("TIMEINJ " + json.dumps(rows[-1]), flush=True)

        for draw in range(args.draws):
            slice_seed = SEED0 + 500 + base_seed + draw     # matched across QRC arms
            for arm in arms:
                use_res, use_inter, time_mode, nq = ARMS[arm]
                for ts in (scales if time_mode == "input" else [np.nan]):
                    started = perf_counter()
                    kw = dict(encoding="quadrature", correlations=True, n_qubits=nq,
                              slice_seed=slice_seed, time_mode=time_mode,
                              t_scale=(ts if time_mode == "input" else 1.0))
                    model, vm, npar = fit(
                        features(x, t, alpha_bar, 4, seed, batch=1024, **kw), eps,
                        features(xv, tv, alpha_bar, 4, seed, batch=1024, **kw), epsv, use_inter)
                    lat = sample_qrc(noise, model, alpha_bar, 4, seed, args.device,
                                     steps=DDIM_STEPS, x0_clip=x0_clip, **kw)
                    enc = lat * sigma
                    rows.append(score(decode_numpy(ae, enc), enc, arm=arm,
                                      mechanism={"input": "A", "unitary": "B"}.get(time_mode, "-"),
                                      seed=s, unitary_draw=draw, t_scale=ts, n_qubits=nq,
                                      n_params=npar, val_mse=vm,
                                      wall_clock_s=perf_counter() - started))
                    print("TIMEINJ " + json.dumps(rows[-1]), flush=True)
            pd.DataFrame(rows).to_parquet(f"results/time_injection{tag}_checkpoint.parquet",
                                          index=False)

    frame = pd.DataFrame(rows)
    frame.to_parquet(f"results/time_injection{tag}.parquet", index=False)
    frame.to_csv(f"results/time_injection{tag}.csv", index=False)

    summary = {"dataset": args.dataset, "stage": args.stage, "floor": floor,
               "n_seeds": args.seeds, "n_draws": args.draws,
               "wall_clock_note": "inflated by a concurrent job on the same GPU; FID unaffected"}

    if args.stage == "scale":
        g = frame.groupby("t_scale").agg(fid=("fid", "mean"), fid_sd=("fid", "std"),
                                         val_mse=("val_mse", "mean"))
        print("\n--- t_scale sweep (selected on FID, on seeds disjoint from stage 2) ---", flush=True)
        print(g.round(4).to_string(), flush=True)
        best = float(g.fid.idxmin())
        summary["scale_curve"] = {str(k): v for k, v in g.fid.to_dict().items()}
        summary["selected_t_scale"] = best
        print(f"\nselected t_scale = {best:g}", flush=True)
    else:
        aggs = dict(fid=("fid", "mean"), fid_sd=("fid", "std"), n_params=("n_params", "first"),
                    val_mse=("val_mse", "mean"), latent_energy=("latent_energy", "mean"),
                    recall=("recall", "mean"), precision=("precision", "mean"),
                    wall=("wall_clock_s", "mean"))
        # Inception Score is only meaningful where a domain classifier exists -- Fashion-MNIST has
        # one (FashionCNN); BreastMNIST does not, and ImageNet classes say nothing about ultrasound.
        if "is_fmnist" in frame.columns:
            aggs["is_fmnist"] = ("is_fmnist", "mean")
            aggs["is_sd"] = ("is_fmnist", "std")
        g = frame.groupby("arm").agg(**aggs)
        order = ["gaussian", "ridge_plain", "ridge_interaction", "qrc_current",
                 "qrc_t_injected_A", "qrc_t_injected_B", "qrc_t_injected_A_interaction"]
        has_is = "is_fmnist" in g.columns
        print(f"\n{'arm':30s} {'params':>8s} {'FID':>15s} "
              + (f"{'IS':>14s} " if has_is else "")
              + f"{'energy':>9s} {'recall':>7s} {'val MSE':>9s} {'s/seed':>8s}", flush=True)
        for a in order:
            if a not in g.index:
                continue
            r = g.loc[a]
            is_col = f"{r.is_fmnist:7.3f}+/-{r.is_sd:5.3f} " if has_is else ""
            print(f"{a:30s} {int(r.n_params):8d} {r.fid:8.2f}+/-{r.fid_sd:5.2f} " + is_col
                  + f"{r.latent_energy:9.4f} {r.recall:7.3f} {r.val_mse:9.5f} {r.wall:8.1f}",
                  flush=True)

        # Paired over (seed, draw) cells; ridge arms are draw-independent so they broadcast.
        def cell(arm):
            sub = frame[frame.arm == arm]
            if sub.unitary_draw.iloc[0] == -1:
                base = sub.set_index("seed").fid
                return pd.Series({(s, d): base[s] for s in range(args.seeds)
                                  for d in range(args.draws)})
            return pd.Series({(int(r.seed), int(r.unitary_draw)): r.fid
                              for r in sub.itertuples()})

        # Primary reference is ridge_interaction, as pre-registered. On Fashion-MNIST the best
        # classical arm is ridge_plain instead (57.72 vs 76.01), so both are reported there; the
        # primary comparison is unchanged.
        contrasts = {}
        for ref_arm in ("ridge_interaction", "ridge_plain"):
            print(f"\n--- contrasts vs {ref_arm} ---", flush=True)
            ref = cell(ref_arm)
            for arm in order:
                if arm in (ref_arm, "gaussian") or arm not in set(frame.arm):
                    continue
                a = cell(arm)
                common = ref.index.intersection(a.index)
                delta = a[common] - ref[common]
                t_stat, p = stats.ttest_rel(a[common], ref[common])
                dispersion = float(frame[frame.arm == arm].fid.std())
                m = mde(float(delta.std()), len(delta))
                verdict = ("BEATS" if delta.mean() < 0 and abs(delta.mean()) > dispersion and p < 0.05
                           else "NULL (within dispersion)" if abs(delta.mean()) <= dispersion
                           else "worse" if delta.mean() > 0 else "inconclusive")
                contrasts[f"{arm}_vs_{ref_arm}"] = {
                    "delta": float(delta.mean()), "sd": float(delta.std()), "p": float(p),
                    "dispersion": dispersion, "mde": m, "n_cells": int(len(delta)),
                    "verdict": verdict}
                print(f"  {arm:30s} d={delta.mean():+7.2f} +/-{delta.std():5.2f}  p={p:.4f}  "
                      f"dispersion={dispersion:.2f}  MDE={m:.2f}  -> {verdict}", flush=True)
        summary["contrasts"] = contrasts

        # Variance decomposition for the QRC arms: draw vs seed vs residual.
        decomp = {}
        for arm in order:
            sub = frame[(frame.arm == arm) & (frame.unitary_draw >= 0)]
            if sub.empty:
                continue
            grid = sub.pivot_table(index="seed", columns="unitary_draw", values="fid")
            grand = grid.values.mean()
            decomp[arm] = {
                "sd_total": float(grid.values.std(ddof=1)),
                "sd_from_unitary_draw": float((grid.mean(axis=0) - grand).std(ddof=1)),
                "sd_from_data_seed": float((grid.mean(axis=1) - grand).std(ddof=1)),
            }
        summary["variance_decomposition"] = decomp
        print("\n--- variance decomposition (QRC arms) ---", flush=True)
        for a, v in decomp.items():
            print(f"  {a:30s} total={v['sd_total']:.2f}  draw={v['sd_from_unitary_draw']:.2f}  "
                  f"seed={v['sd_from_data_seed']:.2f}", flush=True)

        summary["cells"] = {a: frame[frame.arm == a][
            ["fid", "precision", "recall", "density", "coverage", "latent_energy", "val_mse",
             "n_params", "wall_clock_s"] + (["is_fmnist"] if "is_fmnist" in frame.columns else [])
        ].mean().to_dict() for a in order if a in set(frame.arm)}

    Path(f"results/time_injection{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
