# Running the annealing reservoir on a real D-Wave QPU

The simulated annealing arm is complete: six seeds at `n_train=10000`, FID **73.90 +/- 1.62**, in
`results/qrc_fusion_fair/generation_cells_anneal_dwave_sqa_full/`. This document covers the port of
that arm to hardware, under a hard ceiling of **24 minutes of cumulative QPU access time**.

Status: **built and validated against `MockDWaveSampler`; not yet run on hardware** (no API token
configured at time of writing). Nothing here has spent quota.

## The result that makes this affordable

A naive reading says this is impossible. Data enters the reservoir through the local fields `h_i`,
so every row of every batch is a *different* Ising problem, and a full cell needs

| stage | rows |
|---|---|
| fit (`n_train` + `n_val`) | 15,000 |
| generation (`samples` x 50 DDIM steps) | 500,000 |
| **total** | **515,000** |

On a QPU a different problem means a different programming cycle, and programming (~16 ms) dwarfs
everything else: at the published operating point -- a single 5 ns probe, 8 reads -- the actual
annealing and readout come to ~0.3 ms per submission against ~16 ms of programming. Submitted one
row at a time, 515,000 rows is **2.3 hours** of QPU time. The budget is 24 minutes.

**The fix is that the reservoir is small and the chip is large.** A 12-spin problem needs a K12
clique embedding, which on Zephyr takes ~35 qubits in chains of ~3. Advantage2 has thousands, so
many disjoint copies fit side by side and one submission carries one row per copy -- the
programming cost is paid once and amortised across all of them. Measured on a defect-free Zephyr
Z15,4 graph:

```
tiles               148          (rows carried per programming cycle)
chain length        mean 2.94  max 6
qubits used         5214 of 7440
```

That turns the cell into 3,480 submissions at 18.24 ms, or **63.5 s of QPU time -- 1.06 of the 24
minutes**, a ~20x margin. The tiles are disjoint qubit sets with no couplers between them, so this
is parallelism, not approximation: the same physics run in separate regions of one chip.

The margin is large enough that the binding constraint is wall-clock, not quota: 3,480 cloud
round-trips at ~1 s each is ~1 h of waiting for ~1 min of annealing.

## The operating point is reachable, and only just

The published cell anneals for **5 ns**, which is *exactly* the bottom of Advantage2's
`fast_anneal_time_range` of `[0.005, 2000]` us, and below its standard `annealing_time_range` floor
of `0.5` us. So the hardware arm submits with `fast_anneal=True`, and `check_anneal_time` refuses a
duration outside the solver's range rather than letting it be silently rounded up to something 100x
longer.

**This was luck, not design, and the near miss is worth recording.** `anneal/schedule.py`'s
`INFORMATIVE_ANNEAL_US = (0.00005, 0.0002, 0.0005, 0.001)` -- the four-probe window where
consecutive anneal durations still give decorrelated feature maps -- is 0.05-1 ns, entirely *below*
every hardware floor. Had the full run used that tuple, none of its four probes could be reproduced
on hardware. It used a single 5 ns probe instead, which lands on the boundary. Any future sweep that
moves back into the sub-nanosecond window is a simulation-only result.

`h_range` and `j_range` are `[-4, 4]` and `[-1, 1]`, exactly the `H_FIELD_MAX` / `J_COUPLING_MAX`
the feature map already clips to -- so no rescaling is needed and the hardware arm sees the same
fields the simulated one did.

## The silent failure this port nearly shipped

A logical spin is several physical qubits held together by strong ferromagnetic couplers, and
`uniform_torque_compensation` -- Ocean's standard heuristic -- asks for a chain strength of **2.295**
on the production `J` (12 spins, `j_scale=0.5`). The solver's `j_range` is `[-1, 1]`.

With `auto_scale=True`, which is `DWaveSampler`'s default, Ocean resolves that by rescaling the
**entire problem** to fit. The chain couplers come into range, and so do the data-carrying fields:
`h` shrinks from `+-4.0` to `+-1.74`. The hardware arm would then run at 2.3x less signal than the
simulated one against a fixed thermal noise floor -- quietly, with no warning, and only on hardware.
The resulting FID gap would have looked exactly like decoherence.

Two changes close it:

- **Cap the chain strength at the `extended_j_range` floor** (`-2.0`, not `j_range`'s `-1.0` --
  chain couplers are negative, so the extended range's floor is the binding limit).
- **Submit with `auto_scale=False`**, and range-check `h` and `J` before sending, so an over-range
  field raises against `H_FIELD_MAX` / `J_COUPLING_MAX` instead of being silently rescaled.

The cap buys weaker chains, which cost chain *breaks* instead. That is the right trade: a chain
break is measured and reported on every cell (`qpu_mean_chain_break_fraction`), whereas a rescaling
is invisible. If that fraction comes back high, lower `--anneal-j-scale` -- a weaker `J` needs a
weaker chain -- rather than raising the strength back over the range.

## The budget guard

`anneal/qpu_budget.py`. The quota is spent, not rate-limited: overshooting cannot be refunded, so
the ledger is pessimistic wherever it could be optimistic.

- **Charge before submitting, correct afterwards.** The estimate is written and `fsync`ed before the
  problem leaves the process, then rewritten with the solver's reported `qpu_access_time`. A crash
  or a lost reply leaves the *estimate* charged -- the ledger over-counts under failure.
- **The margin is untouchable.** `remaining = cap * (1 - margin) - spent`, so the 24-minute cap with
  the default 10% margin stops at 21.6 minutes.
- **Cumulative and cross-process.** The ledger is `flock`ed for the whole read-modify-write, so two
  runs sharing a quota cannot both pass the check against the same stale balance. To start a fresh
  quota month, move the ledger aside -- do not raise the cap.
- **Estimates prefer the solver's own numbers.** `estimate_access_time_us` reads
  `problem_timing_data` when the solver publishes it and otherwise falls back to constants set
  *above* Advantage2 typicals, since an underestimate is what overspends.

Independently, the QPU path is the only backend whose features are content-addressed to
`cache/quera_features/`: a re-run or a resumed cell must not pay quota twice for fields it already
annealed. Verified in the tests -- a repeated `anneal_features` call issues zero new submissions.

## Running it, in three stages

The target is **six seeds**, matching the SQA arm so the contrast can be a paired per-seed delta
rather than a bare number against a `+/- 1.62` spread. Six seeds cost ~6.6 of the 24 minutes, so
quota is not what constrains this. Wall-clock is: ~1-3 h per seed, dominated by ~3,600 cloud
round-trips, not by the ~66 s of annealing inside them.

That asymmetry is why the runner stages the work. Firing all six at once risks discovering a bad
embedding after 18 hours and 6.6 unrecoverable minutes; each stage answers one question before the
next one pays.

```bash
dwave config create                        # once, or export DWAVE_API_TOKEN
MOCK=1 ./run_dwave_qpu_cell.sh --dry-run   # plan with no token and no QPU time
./run_dwave_qpu_cell.sh --dry-run          # plan against the real solver (still no QPU time)

./run_dwave_qpu_cell.sh --smoke            # stage 1: nt=500, samples=500  -> 3.8 s QPU
./run_dwave_qpu_cell.sh --seed0            # stage 2: full seed 0          -> 65.7 s
./run_dwave_qpu_cell.sh --rest             # stage 3: seeds 1-5, PAR at a time -> 5.5 min

./run_dwave_qpu_cell.sh --status           # ledger balance; spends nothing
```

| stage | QPU | cumulative | question it answers |
|---|---|---|---|
| `--smoke` | 3.8 s | 3.8 s | does the path work on real hardware at all? |
| `--seed0` | 65.7 s | 1.2 min | are chain breaks low and the FID plausible? |
| `--rest` | 5.5 min | 6.6 min | the publishable six-seed result |

**`--rest` is gated on stage 2 and refuses to run blind.** It reads the seed-0 cell and exits
non-zero if that cell is missing, if it tripped the degeneracy guard, or if its mean chain-break
fraction exceeds `MAX_CHAIN_BREAK` (default 5%) -- with the fix in the message, since the correct
response is to lower `--anneal-j-scale`, not to raise chain strength back over the solver's range.

**`--rest` runs `PAR` seeds concurrently** (default 2). The QPU serialises annealing regardless, but
network latency -- the actual bottleneck -- overlaps across processes, which is a larger win than
pipelining inside a single process would be. This is safe because the ledger `flock`s its whole
read-modify-write, so concurrent seeds cannot both pass the cap check against the same stale
balance.

Knobs: `PAR`, `REST_SEEDS`, `MAX_CHAIN_BREAK`, `CAP`, `MARGIN`, `LEDGER`, `N_TRAIN`, `SAMPLES`,
`READS`, `MOCK`.

If quota ever does get tight, cut `--samples`, never `--n-train`: fitting is 102 submissions
against generation's 3,500, so shrinking the training set saves almost nothing and breaks
comparability with the SQA arm.

`experiments/dwave_qpu_preflight.py` consumes no quota -- solver properties and node/edge lists are
metadata, and the tiling search is a classical graph computation on a local copy of the chip graph.
It prints the projection, says whether the cell fits the *remaining* budget, and if it does not,
reports the largest `--samples` that would. The runner refuses to submit if the preflight says no.

**Run the preflight against the real solver before trusting the 63.5 s figure.** It is measured on
a defect-free mock; a real chip has qubits and couplers out of service, so it will tile lower than
148 and cost proportionally more. At 40 tiles the cell is still only ~3.9 min.

## What to check in the output

Each cell writes `qpu_solver`, `qpu_tiles`, `qpu_submissions`, `qpu_rows`, `qpu_access_time_s` and
`qpu_mean_chain_break_fraction` alongside the usual FID columns.

`qpu_mean_chain_break_fraction` is the one that decides whether the result is physics. A logical
spin is several physical qubits, and when they come back disagreeing the value is a majority vote,
not a measurement. A cell with a high chain-break fraction is reporting embedding artefacts and must
not be averaged into a headline FID without saying so. The mock run sits at 0.2%.

## Expect the hardware number to be worse than 73.90, and know why before you look

The simulated arm is closed-system: unitary, zero temperature, no `1/f` flux noise, no bath. A real
QPU is at ~15 mK, decoheres, and has analog calibration error on every `h` and `J`. It also breaks
chains. Every one of those degrades the feature map relative to `anneal/emulator.py`, and the Gate 5
precedent on the Rydberg arm -- where shot noise alone cost ~9.2 FID -- says the gap can be large.

A worse hardware FID is therefore the expected outcome and not, on its own, evidence that the port
failed. The port succeeds if the hardware arm lands near the simulated one; the comparison that
carries the paper is the *paired per-seed* delta against the SQA arm at the identical operating
point, not the absolute number.
