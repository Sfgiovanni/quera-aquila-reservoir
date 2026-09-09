"""Gate 2: sampled Z/ZZ features must converge to the exact ones as shots -> infinity, at the
`1/sqrt(shots)` rate shot noise predicts -- not just "get closer".
"""
from __future__ import annotations

import numpy as np
import torch

from quera.emulator import build_geometry_cache, evolve
from quera.features import (gaussian_noise_like_shots, physical_variance_floor, zz_features_exact,
                            zz_features_sampled, n_features)
from quera.layout import cluster_3x4
from quera.pulses import build_probe_program
from quera.sampling import apply_readout_noise, sample_bitstrings

DEVICE = "cpu"
POSITIONS = cluster_3x4()
N = len(POSITIONS)
SHOT_COUNTS = (50, 100, 300, 1000, 5000)  # 5000 aggregates several <=MAX_SHOTS hardware
                                          # tasks -- this test is about the counting
                                          # statistics, not a single-task submission.
N_TRIALS = 300


def _fixed_psi():
    rng = np.random.default_rng(7)
    h = rng.uniform(0.0, 1.0, N)
    program = build_probe_program(POSITIONS, h, plateau_us=0.5)
    geom = build_geometry_cache(POSITIONS, DEVICE, real_dtype=torch.float64)
    psi = evolve(program, h[None, :], DEVICE, dtype=torch.complex128, geometry=geom)
    return psi, geom


def test_feature_count_matches_spec():
    assert n_features(N) == 12 + 66 == 78


def test_shot_noise_converges_at_inverse_sqrt_rate():
    psi, geom = _fixed_psi()
    exact = zz_features_exact(psi, geom)[0]  # (78,)
    psi_rep = psi.expand(N_TRIALS, -1)  # independent trials as extra batch rows

    empirical_std = {}
    mean_bias = {}
    gen = torch.Generator().manual_seed(0)
    for shots in SHOT_COUNTS:
        sampled, _ = zz_features_sampled(psi_rep, geom, shots, generator=gen)  # (N_TRIALS, 78)
        empirical_std[shots] = float(sampled.std(dim=0).mean())
        mean_bias[shots] = float((sampled.mean(dim=0) - exact).abs().mean())

    # log-log slope should be close to the -1/2 shot-noise prediction, not just "decreasing".
    log_s = np.log(SHOT_COUNTS)
    log_std = np.log([empirical_std[s] for s in SHOT_COUNTS])
    slope, _ = np.polyfit(log_s, log_std, 1)
    assert -0.65 < slope < -0.35, f"shot-noise std scaling slope={slope:.3f}, expected ~-0.5"

    # sigma(<Z_i>) ~= 1/sqrt(shots) is the task spec's own stated formula -- check the
    # smallest and largest S bracket it within a factor of 3 (features aren't all Z_i alone,
    # ZZ correlators included, so this is a magnitude check, not an exact match).
    assert 1 / 3 < empirical_std[SHOT_COUNTS[0]] * np.sqrt(SHOT_COUNTS[0]) < 3
    assert 1 / 3 < empirical_std[SHOT_COUNTS[-1]] * np.sqrt(SHOT_COUNTS[-1]) < 3

    # Sampling is unbiased: mean over trials should approach the exact value as shots grow,
    # much faster than the std shrinks (bias is O(0), only Monte Carlo noise remains).
    assert mean_bias[SHOT_COUNTS[-1]] < mean_bias[SHOT_COUNTS[0]]
    assert mean_bias[SHOT_COUNTS[-1]] < 0.05


def test_vacancy_filtering_matches_spec_number():
    """0.99**12 = 88.6% usable shots at vacancy_rate=0.01 -- the task spec's own worked
    number, checked directly rather than trusted."""
    psi, geom = _fixed_psi()
    bits = sample_bitstrings(psi.expand(2000, -1), geom, shots=500,
                             generator=torch.Generator().manual_seed(1))
    _, usable = apply_readout_noise(bits, vacancy_rate=0.01,
                                    generator=torch.Generator().manual_seed(2))
    frac_usable = float(usable.float().mean())
    assert abs(frac_usable - 0.99**N) < 0.01


def test_detection_error_biases_toward_symmetric_point():
    """Independent false positive/negative rates should measurably bias the sampled <Z_i>
    toward 0 relative to the noiseless case, and the bias should grow with the error rate."""
    psi, geom = _fixed_psi()
    psi_rep = psi.expand(400, -1)
    gen = torch.Generator().manual_seed(3)
    clean, _ = zz_features_sampled(psi_rep, geom, 2000, generator=gen)
    noisy_small, _ = zz_features_sampled(psi_rep, geom, 2000, generator=gen, fp_rate=0.01, fn_rate=0.02)
    noisy_large, _ = zz_features_sampled(psi_rep, geom, 2000, generator=gen, fp_rate=0.05, fn_rate=0.08)

    shift_small = (clean.mean(0) - noisy_small.mean(0)).abs().mean()
    shift_large = (clean.mean(0) - noisy_large.mean(0)).abs().mean()
    assert shift_large > shift_small > 0


def test_gaussian_control_matches_shot_noise_variance_not_structure():
    """Gate 5's control: same per-feature variance as real shot noise, independent across
    features rather than correlated within a probe time (real shot noise's `Z_i`/`Z_iZ_j`
    come from the same multinomial draws, so are correlated; the Gaussian control is not, by
    construction)."""
    psi, geom = _fixed_psi()
    exact = zz_features_exact(psi, geom)[0]
    exact_rep = exact.expand(N_TRIALS, -1)
    shots = 300
    gen = torch.Generator().manual_seed(11)
    gauss = gaussian_noise_like_shots(exact_rep, shots, generator=gen)

    empirical_std = float(gauss.std(dim=0).mean())
    target_std = float(physical_variance_floor(exact, shots).sqrt().mean())
    assert abs(empirical_std - target_std) / target_std < 0.2

    mean_bias = float((gauss.mean(dim=0) - exact).abs().mean())
    assert mean_bias < 0.05  # unbiased, same as real shot noise


def test_gaussian_control_seed_deterministic():
    psi, geom = _fixed_psi()
    exact = zz_features_exact(psi, geom)[0].expand(50, -1)
    a = gaussian_noise_like_shots(exact, 100, generator=torch.Generator().manual_seed(5))
    b = gaussian_noise_like_shots(exact, 100, generator=torch.Generator().manual_seed(5))
    c = gaussian_noise_like_shots(exact, 100, generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_physical_variance_floor_shape_and_bounds():
    features = torch.tensor([0.0, 0.5, -0.9, 1.0])
    floor = physical_variance_floor(features, shots=100)
    expected = (1 - features**2) / 100
    assert torch.allclose(floor, expected)
    assert torch.all(floor >= 0)
