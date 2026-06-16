"""
Arrhythmia State Machine
Manages random cardiac arrhythmia / health deterioration episodes.
Episodes fire according to a Poisson process (mean ~2.5 hrs).
Transitions are smoothly ramped — no abrupt jumps in vitals.
"""

import time
import random
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

ARRHYTHMIA_MEAN_INTERVAL_S = float(os.getenv("ARRHYTHMIA_MEAN_INTERVAL_S", "9000"))


class PatientState(str, Enum):
    STABLE        = "stable"
    PVC           = "pvc"               # Premature ventricular contractions
    TACHYCARDIA   = "sinus_tachycardia"
    BRADYCARDIA   = "sinus_bradycardia"
    AFIB          = "atrial_fibrillation"
    DESATURATION  = "spo2_desaturation"
    HYPERTENSIVE  = "hypertensive_spike"
    RESP_DISTRESS = "respiratory_distress"


@dataclass
class Episode:
    """Represents a single arrhythmia / health deterioration episode."""
    state: PatientState
    start_time: float                   # epoch seconds
    duration_s: float                   # total planned duration
    label: str = ""
    severity: str = "mild"             # mild | moderate | severe

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_s

    @property
    def is_active(self) -> bool:
        return time.time() < self.end_time

    def ramp_factor(self, ramp_seconds: float = 12.0) -> float:
        """Returns 0→1 during ramp-in, 1 during plateau, 1→0 during ramp-out."""
        now = time.time()
        elapsed = now - self.start_time
        remaining = self.end_time - now

        if elapsed < 0:
            return 0.0
        ramp_in = min(1.0, elapsed / ramp_seconds)
        ramp_out = min(1.0, remaining / ramp_seconds)
        return min(ramp_in, ramp_out)


# --- Episode type definitions ---
#  Each entry: (state, min_duration_s, max_duration_s, weight, severity)
EPISODE_TYPES = [
    (PatientState.PVC,           30,   120, 3.0, "mild"),
    (PatientState.TACHYCARDIA,   60,   300, 2.0, "mild"),
    (PatientState.BRADYCARDIA,   45,   240, 1.5, "moderate"),
    (PatientState.AFIB,          90,   360, 1.0, "moderate"),
    (PatientState.DESATURATION,  30,   180, 2.5, "mild"),
    (PatientState.HYPERTENSIVE,  60,   300, 1.5, "moderate"),
    (PatientState.RESP_DISTRESS, 60,   240, 1.0, "mild"),
]

EPISODE_LABELS = {
    PatientState.PVC:           "Premature Ventricular Contractions",
    PatientState.TACHYCARDIA:   "Sinus Tachycardia",
    PatientState.BRADYCARDIA:   "Sinus Bradycardia",
    PatientState.AFIB:          "Atrial Fibrillation",
    PatientState.DESATURATION:  "SpO₂ Desaturation",
    PatientState.HYPERTENSIVE:  "Hypertensive Spike",
    PatientState.RESP_DISTRESS: "Respiratory Distress",
}


class ArrhythmiaStateMachine:
    """
    Manages patient state transitions.
    Uses Poisson-distributed episode timing.
    """

    def __init__(self, mean_interval_s: float = ARRHYTHMIA_MEAN_INTERVAL_S,
                 afib_history: bool = False):
        self.mean_interval_s = mean_interval_s
        self.afib_history = afib_history
        self.current_episode: Optional[Episode] = None
        self.episode_history: list[dict] = []
        self._next_episode_time: float = self._schedule_next()

    def _schedule_next(self) -> float:
        """Schedule next episode using exponential distribution (Poisson process)."""
        wait = random.expovariate(1.0 / self.mean_interval_s)
        # Clamp between 30 min and 4 hours
        wait = max(1800, min(wait, 14400))
        return time.time() + wait

    def _pick_episode(self) -> Episode:
        """Randomly pick an episode type, weighted by frequency."""
        types = EPISODE_TYPES[:]
        # Boost AFib if patient has history
        if self.afib_history:
            types = [(s, mn, mx, w * 3.0 if s == PatientState.AFIB else w, sv)
                     for (s, mn, mx, w, sv) in types]

        states, min_durs, max_durs, weights, severities = zip(*types)
        chosen_idx = random.choices(range(len(states)), weights=weights, k=1)[0]
        chosen = states[chosen_idx]
        duration = random.uniform(min_durs[chosen_idx], max_durs[chosen_idx])
        severity = severities[chosen_idx]

        # Occasionally make it more severe
        if random.random() < 0.15:
            duration *= 1.5
            severity = "severe"

        return Episode(
            state=chosen,
            start_time=time.time(),
            duration_s=duration,
            label=EPISODE_LABELS[chosen],
            severity=severity,
        )

    def update(self) -> Tuple[PatientState, Optional[Episode], float]:
        """
        Call this every simulation tick.
        Returns: (current_state, current_episode_or_None, ramp_factor)
        """
        now = time.time()

        # Check if current episode expired
        if self.current_episode is not None:
            if not self.current_episode.is_active:
                logger.info(
                    f"Episode ended: {self.current_episode.label} "
                    f"(duration: {self.current_episode.duration_s:.0f}s)"
                )
                self.episode_history.append({
                    "state": self.current_episode.state.value,
                    "label": self.current_episode.label,
                    "start_time": self.current_episode.start_time,
                    "duration_s": self.current_episode.duration_s,
                    "severity": self.current_episode.severity,
                })
                self.current_episode = None
                self._next_episode_time = self._schedule_next()

        # Trigger new episode if scheduled
        if self.current_episode is None and now >= self._next_episode_time:
            self.current_episode = self._pick_episode()
            logger.warning(
                f"Episode starting: {self.current_episode.label} "
                f"(planned duration: {self.current_episode.duration_s:.0f}s, "
                f"severity: {self.current_episode.severity})"
            )

        if self.current_episode is not None:
            ramp = self.current_episode.ramp_factor()
            return self.current_episode.state, self.current_episode, ramp

        return PatientState.STABLE, None, 0.0

    def get_vital_modifiers(self, state: PatientState, ramp: float
                            ) -> dict:
        """
        Returns delta modifiers for each vital sign given the current state.
        Values are scaled by ramp_factor (smooth onset/offset).
        """
        base = {
            "hr_delta":   0.0,
            "sbp_delta":  0.0,
            "dbp_delta":  0.0,
            "spo2_delta": 0.0,
            "rr_delta":   0.0,
            "irregularity": 0.0,   # 0=regular, 1=fully irregular (AFib)
            "pvc_burden": 0.0,     # fraction of beats that are PVCs
        }

        mods = {
            PatientState.STABLE: {},
            PatientState.PVC: {
                "hr_delta": -3.0,
                "sbp_delta": -6.0,
                "dbp_delta": -3.0,
                "spo2_delta": -0.8,
                "pvc_burden": 0.18,   # ~18% of beats are PVCs
            },
            PatientState.TACHYCARDIA: {
                "hr_delta":   +32.0,
                "sbp_delta":  +12.0,
                "dbp_delta":  +8.0,
                "spo2_delta": -1.5,
                "rr_delta":   +3.0,
            },
            PatientState.BRADYCARDIA: {
                "hr_delta":   -25.0,
                "sbp_delta":  -14.0,
                "dbp_delta":  -8.0,
                "spo2_delta": -2.5,
                "rr_delta":   -2.0,
            },
            PatientState.AFIB: {
                "hr_delta":   +28.0,
                "sbp_delta":  -8.0,
                "dbp_delta":  -5.0,
                "spo2_delta": -3.0,
                "rr_delta":   +2.0,
                "irregularity": 1.0,
            },
            PatientState.DESATURATION: {
                "hr_delta":   +8.0,
                "sbp_delta":  +5.0,
                "dbp_delta":  +3.0,
                "spo2_delta": -6.0,
                "rr_delta":   +4.0,
            },
            PatientState.HYPERTENSIVE: {
                "hr_delta":   +12.0,
                "sbp_delta":  +32.0,
                "dbp_delta":  +18.0,
                "spo2_delta": -0.5,
                "rr_delta":   +1.5,
            },
            PatientState.RESP_DISTRESS: {
                "hr_delta":   +10.0,
                "sbp_delta":  +6.0,
                "dbp_delta":  +4.0,
                "spo2_delta": -4.0,
                "rr_delta":   +6.0,
            },
        }

        chosen = mods.get(state, {})
        for key, val in chosen.items():
            base[key] = val * ramp

        return base

    @property
    def seconds_until_next(self) -> float:
        if self.current_episode is not None:
            return 0.0
        return max(0.0, self._next_episode_time - time.time())
