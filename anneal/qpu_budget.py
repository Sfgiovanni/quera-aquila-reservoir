"""Hard ceiling on cumulative QPU access time, enforced across processes and runs.

The Leap quota is spent, not rate-limited: once it is gone it is gone until the month rolls over,
and a run that overshoots cannot be refunded. So this module is deliberately pessimistic in every
place where it could be optimistic:

1. **Charge before submitting, correct afterwards.** `charge()` writes the *estimated* cost to the
   ledger and fsyncs it before the submission leaves the process, then rewrites it with the
   `qpu_access_time` the solver actually reports. A crash, a kill -9 or a lost network reply
   therefore leaves the estimate charged rather than nothing -- the ledger over-counts under
   failure, which is the safe direction.
2. **The cap is the whole quota; the margin is what we refuse to touch.** `remaining()` is
   `cap * (1 - margin) - spent`, so the default 24-minute cap with a 10% margin will not spend
   past 21.6 minutes. The margin absorbs the gap between D-Wave's estimate and its bill.
3. **Locked.** The ledger is `flock`ed for the whole read-modify-write, so two runs sharing a quota
   cannot both pass the check against the same stale balance.

The estimate itself comes from `estimate_access_time_us`, which prefers the solver's own
`problem_timing_data` and falls back to constants chosen high rather than typical -- see there.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CAP_SECONDS = 24 * 60.0
DEFAULT_MARGIN = 0.10

# Fallbacks for solvers that do not publish `problem_timing_data`. Every one of these is set
# ABOVE the Advantage2 typical value, because an underestimate here is what overspends the quota.
FALLBACK_PROGRAMMING_US = 16_000.0        # Advantage2 typical is ~14.0 ms
FALLBACK_READOUT_US = 130.0               # per read; Advantage2 typical is ~30-45 us
FALLBACK_DELAY_US = 25.0                  # qpu_delay_time_per_sample
FALLBACK_THERMALIZATION_US = 1_000.0      # default programming thermalization


class BudgetExceeded(RuntimeError):
    """Raised instead of submitting when a call would take cumulative spend past the cap."""


def estimate_access_time_us(properties: dict, *, num_reads: int, anneal_us: float) -> float:
    """Upper-ish bound on `qpu_access_time` for one submission, in microseconds.

    D-Wave's own decomposition is

        access = programming + num_reads * (anneal + readout + delay + readout_thermalization)

    `problem_timing_data` carries the per-solver numbers when the solver publishes it. The readout
    time there is a model (`readout_time_model_parameters`) rather than a scalar; we take its
    maximum, since we cannot know the per-problem readout length before submitting and the maximum
    is the side that protects the quota.
    """
    timing = (properties or {}).get("problem_timing_data") or {}

    programming = float(timing.get("typical_programming_time", FALLBACK_PROGRAMMING_US))
    thermalization = float(timing.get("default_programming_thermalization",
                                      FALLBACK_THERMALIZATION_US))
    delay = float(timing.get("qpu_delay_time_per_sample", FALLBACK_DELAY_US))
    readout_thermalization = float(timing.get("default_readout_thermalization", 0.0))

    model = timing.get("readout_time_model_parameters")
    readout = max(float(v) for v in model) if model else FALLBACK_READOUT_US

    per_read = float(anneal_us) + readout + delay + readout_thermalization
    return programming + thermalization + num_reads * per_read


@dataclass
class QpuBudget:
    """Append-only ledger of QPU access time with a hard, cross-process ceiling."""

    path: Path
    cap_seconds: float = DEFAULT_CAP_SECONDS
    margin: float = DEFAULT_MARGIN

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- ledger io -------------------------------------------------------------------------
    def _blank(self) -> dict:
        return {"cap_seconds": self.cap_seconds, "margin": self.margin, "entries": []}

    def _read(self, handle) -> dict:
        handle.seek(0)
        raw = handle.read()
        if not raw.strip():
            return self._blank()
        return json.loads(raw)

    def _write(self, handle, state: dict) -> None:
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())

    @contextmanager
    def _locked(self):
        with open(self.path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # -- accounting ------------------------------------------------------------------------
    @staticmethod
    def _spent_us(state: dict) -> float:
        return sum(float(e["charged_us"]) for e in state["entries"])

    def spent_seconds(self) -> float:
        with self._locked() as handle:
            return self._spent_us(self._read(handle)) / 1e6

    def remaining_seconds(self) -> float:
        """Budget still spendable, i.e. the cap less the untouchable margin, less what is gone."""
        with self._locked() as handle:
            state = self._read(handle)
        usable = float(state.get("cap_seconds", self.cap_seconds)) * (
            1.0 - float(state.get("margin", self.margin)))
        return usable - self._spent_us(state) / 1e6

    def summary(self) -> dict:
        with self._locked() as handle:
            state = self._read(handle)
        spent = self._spent_us(state) / 1e6
        cap = float(state.get("cap_seconds", self.cap_seconds))
        usable = cap * (1.0 - float(state.get("margin", self.margin)))
        return {"cap_seconds": cap, "usable_seconds": usable, "spent_seconds": spent,
                "remaining_seconds": usable - spent, "submissions": len(state["entries"])}

    @contextmanager
    def charge(self, estimate_us: float, label: str = ""):
        """Reserve `estimate_us`, hand back a recorder for the real cost, settle on exit.

        Raises `BudgetExceeded` *before* yielding if the reservation would breach the cap, so the
        caller never submits. Usage::

            with budget.charge(est, "fit batch 3") as record:
                result = sampler.sample(...)
                record(result.info["timing"]["qpu_access_time"])
        """
        estimate_us = float(estimate_us)
        with self._locked() as handle:
            state = self._read(handle)
            cap = float(state.get("cap_seconds", self.cap_seconds))
            usable_us = cap * (1.0 - float(state.get("margin", self.margin))) * 1e6
            spent_us = self._spent_us(state)
            if spent_us + estimate_us > usable_us:
                raise BudgetExceeded(
                    f"refusing to submit {label or 'batch'}: estimate {estimate_us / 1e6:.3f}s "
                    f"+ spent {spent_us / 1e6:.3f}s exceeds usable budget "
                    f"{usable_us / 1e6:.3f}s (cap {cap:.0f}s, margin {self.margin:.0%}). "
                    f"Remaining: {(usable_us - spent_us) / 1e6:.3f}s.")
            entry = {"label": label, "charged_us": estimate_us, "estimate_us": estimate_us,
                     "status": "pending", "started": time.time()}
            state["entries"].append(entry)
            index = len(state["entries"]) - 1
            self._write(handle, state)

        actual: list[float] = []

        def record(access_time_us: float) -> None:
            actual.append(float(access_time_us))

        try:
            yield record
        finally:
            # Settle to the measured cost when we got one; otherwise the estimate stands charged.
            with self._locked() as handle:
                state = self._read(handle)
                entry = state["entries"][index]
                if actual:
                    entry["charged_us"] = actual[-1]
                    entry["status"] = "done"
                else:
                    entry["status"] = "unreported"
                entry["finished"] = time.time()
                self._write(handle, state)
