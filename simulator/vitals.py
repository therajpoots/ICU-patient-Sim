"""
Vitals & Signal Generator
Produces realistic ECG, PPG, Respiratory waveforms using biosignal-simulator,
plus computed SpO2, HR, Blood Pressure with ICU-grade noise layering.
"""

import time
import math
import random
import threading
import logging
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
import biosignal_simulator as bss

from simulator.patient_profile import PatientProfile
from simulator.arrhythmia_states import PatientState, ArrhythmiaStateMachine

logger = logging.getLogger(__name__)

SAMPLING_RATE = 250   # Hz


def _add_baseline_wander(signal: np.ndarray, sr: int,
                          freq: float = 0.1, amp: float = 0.12) -> np.ndarray:
    """Adds sinusoidal baseline drift + slow random walk component."""
    t = np.arange(len(signal)) / sr
    # Primary wander frequency
    wander = amp * np.sin(2 * np.pi * freq * t + random.uniform(0, 2 * math.pi))
    # Secondary slow drift
    wander += (amp * 0.4) * np.sin(2 * np.pi * (freq * 0.3) * t
                                    + random.uniform(0, 2 * math.pi))
    return signal + wander


def _add_powerline(signal: np.ndarray, sr: int, amp: float = 0.012) -> np.ndarray:
    """Adds 50Hz powerline interference."""
    t = np.arange(len(signal)) / sr
    return signal + amp * np.sin(2 * np.pi * 50 * t)


def _add_gaussian_noise(signal: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Adds white Gaussian noise."""
    return signal + np.random.normal(0, sigma, len(signal))


def _add_motion_artifact(signal: np.ndarray, sr: int,
                          prob_per_second: float = 0.003) -> np.ndarray:
    """Randomly inserts short motion burst artifacts."""
    out = signal.copy()
    n_samples = len(signal)
    n_seconds = n_samples / sr
    n_artifacts = np.random.poisson(prob_per_second * n_seconds)
    for _ in range(n_artifacts):
        start = random.randint(0, n_samples - 1)
        burst_len = int(sr * random.uniform(0.05, 0.3))
        end = min(start + burst_len, n_samples)
        amp = random.uniform(0.2, 0.8)
        out[start:end] += np.random.normal(0, amp, end - start)
    return out


def generate_ecg_segment(hr: float, duration_s: float, sr: int = SAMPLING_RATE,
                          profile: Optional[PatientProfile] = None,
                          state: PatientState = PatientState.STABLE,
                          pvc_burden: float = 0.0,
                          irregularity: float = 0.0) -> np.ndarray:
    """
    Generates a single ECG waveform segment using biosignal-simulator.
    - Maps states to biosignal-simulator rhythm types.
    - Applies custom noise layering.
    """
    if profile is None:
        profile = PatientProfile()

    # Map state & metrics to rhythm_type
    rhythm = "normal"
    if irregularity > 0.5 or state == PatientState.AFIB:
        rhythm = "afib"
    elif pvc_burden > 0.01 or state == PatientState.PVC:
        rhythm = "pvc"
    elif state == PatientState.TACHYCARDIA:
        rhythm = "tachycardia"
    elif state == PatientState.BRADYCARDIA:
        rhythm = "bradycardia"
    elif state == PatientState.VFIB:
        rhythm = "vfib"

    try:
        config = bss.ECGConfig(
            fs=float(sr),
            duration_s=duration_s,
            heart_rate=hr,
            rhythm_type=rhythm,
            lead_type="single",
            lead_name="II",
        )
        generator = bss.ECGGenerator(config)
        ecg = generator.generate()
    except Exception as exc:
        logger.warning(f"Error in biosignal_simulator ECGGenerator: {exc}. Falling back to simple simulator.")
        t = np.arange(int(duration_s * sr)) / sr
        ecg = np.sin(2 * np.pi * (hr / 60.0) * t)

    # --- Noise layers ---
    ecg = _add_baseline_wander(ecg, sr,
                                freq=profile.ecg_baseline_wander_hz,
                                amp=profile.ecg_baseline_wander_amp)
    ecg = _add_powerline(ecg, sr, amp=0.010)
    ecg = _add_gaussian_noise(ecg, sigma=profile.ecg_noise_amplitude)
    ecg = _add_motion_artifact(ecg, sr, prob_per_second=0.004)

    return ecg.astype(np.float32)


def generate_ppg_segment(hr: float, duration_s: float, sr: int = SAMPLING_RATE,
                          profile: Optional[PatientProfile] = None,
                          rr: Optional[float] = None,
                          state: PatientState = PatientState.STABLE) -> np.ndarray:
    """Generates a PPG (photoplethysmogram) waveform segment using biosignal-simulator."""
    if profile is None:
        profile = PatientProfile()
    
    # In Ventricular Fibrillation, cardiac output drops to zero, causing the pulse to vanish (flatline)
    if state == PatientState.VFIB:
        ppg = np.zeros(int(duration_s * sr))
    else:
        # Couple respiration rate if provided, otherwise default to baseline (typically ~15-16 bpm)
        if rr is None:
            rr = profile.rr_baseline
        
        try:
            config = bss.PPGConfig(
                fs=float(sr),
                duration_s=duration_s,
                heart_rate=hr,
                resp_rate=rr / 60.0,
                resp_modulation=0.15,
            )
            generator = bss.PPGGenerator(config)
            ppg = generator.generate()
        except Exception as exc:
            logger.warning(f"Error in biosignal_simulator PPGGenerator: {exc}. Falling back to simple simulator.")
            t = np.arange(int(duration_s * sr)) / sr
            ppg = np.sin(2 * np.pi * (hr / 60.0) * t) ** 2

    ppg = _add_baseline_wander(ppg, sr, freq=0.08, amp=0.08)
    ppg = _add_gaussian_noise(ppg, sigma=profile.ppg_noise_amplitude)
    ppg = _add_motion_artifact(ppg, sr, prob_per_second=0.003)
    return ppg.astype(np.float32)


def generate_rsp_segment(rr: float, duration_s: float, sr: int = SAMPLING_RATE,
                          profile: Optional[PatientProfile] = None) -> np.ndarray:
    """Generates a respiratory (RSP) waveform segment using biosignal-simulator."""
    if profile is None:
        profile = PatientProfile()
    try:
        config = bss.RespConfig(
            fs=float(sr),
            duration_s=duration_s,
            resp_rate_hz=rr / 60.0,
            amplitude=1.0,
        )
        generator = bss.RespGenerator(config)
        rsp = generator.generate()
    except Exception as exc:
        logger.warning(f"Error in biosignal_simulator RespGenerator: {exc}. Falling back to simple simulator.")
        t = np.arange(int(duration_s * sr)) / sr
        rsp = np.sin(2 * np.pi * (rr / 60.0) * t)

    rsp = _add_baseline_wander(rsp, sr, freq=0.04, amp=0.06)
    rsp = _add_gaussian_noise(rsp, sigma=profile.rsp_noise_amplitude)
    return rsp.astype(np.float32)



class VitalsEngine:
    """
    Continuously evolving vital signs model.
    Uses random walks constrained to physiological bounds.
    Modifiers from arrhythmia states are applied on top.
    """

    def __init__(self, profile: PatientProfile, state_machine: ArrhythmiaStateMachine):
        self.profile = profile
        self.state_machine = state_machine

        # Current "true" vital values (before modifiers)
        self._hr = profile.hr_baseline
        self._sbp = profile.sbp_baseline
        self._dbp = profile.dbp_baseline
        self._rr = profile.rr_baseline
        self._spo2 = profile.spo2_baseline

        self._last_update = time.time()
        self._lock = threading.Lock()

    def _random_walk(self, current: float, target_drift: float, sigma: float,
                     lo: float, hi: float, dt: float) -> float:
        """Ornstein-Uhlenbeck process: mean-reverting random walk."""
        theta = 0.05   # reversion speed
        new_val = (current
                   + theta * (target_drift - current) * dt
                   + sigma * math.sqrt(dt) * random.gauss(0, 1))
        return float(np.clip(new_val, lo, hi))

    def update(self, dt: float) -> dict:
        """
        Advance vitals by dt seconds.
        Returns a dict of current vital values with state modifiers applied.
        """
        with self._lock:
            state, episode, ramp = self.state_machine.update()
            mods = self.state_machine.get_vital_modifiers(state, ramp)

            p = self.profile

            # Update underlying random walks (slow physiological drift)
            self._hr = self._random_walk(
                self._hr, p.hr_baseline, p.hr_variability,
                p.hr_min, p.hr_max, dt)
            self._sbp = self._random_walk(
                self._sbp, p.sbp_baseline, p.sbp_variability,
                p.sbp_baseline - 15, p.sbp_baseline + 15, dt)
            self._dbp = self._random_walk(
                self._dbp, p.dbp_baseline, p.dbp_variability,
                p.dbp_baseline - 10, p.dbp_baseline + 10, dt)
            self._rr = self._random_walk(
                self._rr, p.rr_baseline, p.rr_variability,
                p.rr_min, p.rr_max, dt)
            self._spo2 = self._random_walk(
                self._spo2, p.spo2_baseline, p.spo2_variability,
                88, 100, dt)

            # Apply state modifiers (with ramp scaling)
            hr = float(np.clip(self._hr + mods["hr_delta"],
                               30, 200))
            
            # If in VFib, allow blood pressure to drop to near-zero (shock)
            min_sbp = 20 if state == PatientState.VFIB else 70
            min_dbp = 10 if state == PatientState.VFIB else 40
            min_spo2 = 10 if state == PatientState.VFIB else 82

            if state == PatientState.VFIB:
                # Interpolate between current baseline and shock floor based on ramp
                sbp_target = 25.0
                dbp_target = 15.0
                sbp = float(np.clip((1.0 - ramp) * self._sbp + ramp * sbp_target, min_sbp, 220))
                dbp = float(np.clip((1.0 - ramp) * self._dbp + ramp * dbp_target, min_dbp, 130))
            else:
                sbp = float(np.clip(self._sbp + mods["sbp_delta"], min_sbp, 220))
                dbp = float(np.clip(self._dbp + mods["dbp_delta"], min_dbp, 130))

            rr = float(np.clip(self._rr + mods["rr_delta"],
                               6, 40))

            # SpO2 is also coupled to RR (hypoxia link)
            rr_spo2_coupling = -max(0, (rr - p.rr_baseline) * 0.12)
            spo2 = float(np.clip(
                self._spo2 + mods["spo2_delta"] + rr_spo2_coupling,
                min_spo2, 100))

            # MAP (mean arterial pressure)
            map_val = dbp + (sbp - dbp) / 3.0

            return {
                "heart_rate": round(hr, 1),
                "systolic_bp": round(sbp, 1),
                "diastolic_bp": round(dbp, 1),
                "map": round(map_val, 1),
                "spo2": round(spo2, 1),
                "respiratory_rate": round(rr, 1),
                "patient_state": state.value,
                "episode_label": episode.label if episode else None,
                "episode_severity": episode.severity if episode else None,
                "episode_ramp": round(ramp, 3),
                "pvc_burden": round(mods["pvc_burden"], 3),
                "irregularity": round(mods["irregularity"], 3),
                "timestamp": time.time(),
            }
