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

from simulator.patient_profile import PatientProfile
from simulator.arrhythmia_states import PatientState, ArrhythmiaStateMachine

logger = logging.getLogger(__name__)

SAMPLING_RATE = 250   # Hz


def _add_baseline_wander(signal: np.ndarray, sr: int,
                          freq: float = 0.1, amp: float = 0.12, t_start: float = 0.0) -> np.ndarray:
    """Adds sinusoidal baseline drift + slow random walk component."""
    t = t_start + np.arange(len(signal)) / sr
    # Primary wander frequency
    wander = amp * np.sin(2 * np.pi * freq * t)
    # Secondary slow drift
    wander += (amp * 0.4) * np.sin(2 * np.pi * (freq * 0.3) * t)
    return signal + wander


def _add_powerline(signal: np.ndarray, sr: int, amp: float = 0.012, t_start: float = 0.0) -> np.ndarray:
    """Adds 50Hz powerline interference."""
    t = t_start + np.arange(len(signal)) / sr
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
                          pac_burden: float = 0.0,
                          irregularity: float = 0.0,
                          st_delta: float = 0.0,
                          t_start: float = 0.0) -> np.ndarray:
    """
    Generates a phase-continuous, mathematically realistic ECG (Lead II) waveform segment.
    Uses continuous phase tracking on the profile to prevent segment boundary step discontinuities.
    """
    if profile is None:
        profile = PatientProfile()

    n_samples = int(duration_s * sr)
    t = t_start + np.arange(n_samples) / sr
    ecg = np.zeros(n_samples, dtype=np.float32)

    # Initialize ECG phase if not present
    if not hasattr(profile, '_ecg_phase'):
        profile._ecg_phase = (t_start * (hr / 60.0)) % 1.0

    # Ventricular Fibrillation (VFIB) is chaotic, with no QRS waves
    if state == PatientState.VFIB:
        # Sum of multiple asynchronous sines to produce realistic fibrillation activity
        f_fib = 5.2 # Hz
        ecg = (
            0.26 * np.sin(2 * np.pi * f_fib * t) +
            0.12 * np.sin(2 * np.pi * (f_fib * 1.7) * t + 0.4) +
            0.08 * np.sin(2 * np.pi * (f_fib * 0.61) * t + 1.2)
        )
        # Advance the underlying phase to prevent freeze when transitioning back
        profile._ecg_phase = (profile._ecg_phase + n_samples * (hr / 60.0) / sr) % 1.0
    else:
        # Generate standard heartbeat components using a phase-continuous oscillator
        for i, ti in enumerate(t):
            base_phase = profile._ecg_phase

            # AFib has highly irregular RR intervals (rhythm irregularity)
            if state == PatientState.AFIB or state == PatientState.AFLUTTER or irregularity > 0.5:
                # Add low-frequency phase jitter to simulate irregularity
                phase_jitter = 0.09 * np.sin(2 * np.pi * 0.22 * ti) + 0.04 * np.cos(2 * np.pi * 0.95 * ti)
                phase = (base_phase + phase_jitter) % 1.0
            else:
                phase = base_phase % 1.0

            # Determine if current beat index is a PVC or PAC beat
            beat_idx = int(ti * (hr / 60.0))
            
            # Isolated ectopic check for PVCs/PACs in STABLE state
            is_stable_pvc = (state == PatientState.STABLE and ((beat_idx * 179 + 31) % 40 == 0))
            is_pvc_beat = ((state == PatientState.PVC or pvc_burden > 0.01) and (beat_idx % 4 == 0)) or is_stable_pvc
            
            is_stable_pac = (state == PatientState.STABLE and ((beat_idx * 131 + 47) % 45 == 0))
            is_pac_beat = not is_pvc_beat and (((state == PatientState.PAC or pac_burden > 0.01) and (beat_idx % 4 == 0)) or is_stable_pac)

            if is_pvc_beat:
                # Premature Ventricular Contraction (no P wave, wide QRS, large inverted T wave)
                p_pvc = (phase + 0.08) % 1.0
                r_pvc = 1.15 * np.exp(-((p_pvc - 0.16) / 0.045) ** 2) # wide R
                s_pvc = -0.55 * np.exp(-((p_pvc - 0.26) / 0.065) ** 2) # deep, wide S
                t_pvc = -0.38 * np.exp(-((p_pvc - 0.48) / 0.085) ** 2) # inverted T
                val = r_pvc + s_pvc + t_pvc
            elif is_pac_beat:
                # Premature Atrial Contraction (early P wave, narrow QRS)
                p_pac = (phase + 0.07) % 1.0
                p_wave = -0.10 * np.exp(-((p_pac - 0.10) / 0.02) ** 2)  # inverted/premature P
                q_wave = -0.06 * np.exp(-((p_pac - 0.185) / 0.009) ** 2)
                r_wave = 1.35 * np.exp(-((p_pac - 0.20) / 0.008) ** 2)
                s_wave = -0.32 * np.exp(-((p_pac - 0.22) / 0.012) ** 2)
                t_wave = 0.26 * np.exp(-((p_pac - 0.40) / 0.055) ** 2)
                st_wave = st_delta * np.exp(-((p_pac - 0.28) / 0.075) ** 2)
                val = p_wave + q_wave + r_wave + s_wave + t_wave + st_wave
            else:
                # Normal clinical QRS complex (ECG Lead II)
                p_wave = 0.12 * np.exp(-((phase - 0.12) / 0.035) ** 2)
                q_wave = -0.06 * np.exp(-((phase - 0.185) / 0.009) ** 2)
                r_wave = 1.35 * np.exp(-((phase - 0.20) / 0.008) ** 2) # sharp R spike
                s_wave = -0.32 * np.exp(-((phase - 0.22) / 0.012) ** 2)
                t_wave = 0.26 * np.exp(-((phase - 0.40) / 0.055) ** 2) # wider T wave
                st_wave = st_delta * np.exp(-((phase - 0.28) / 0.075) ** 2) # ST shift
                
                # Fibrillation waves (f-waves) replace P-waves in AFib
                # AFlutter has saw-tooth regular flutter waves
                if state == PatientState.AFIB or irregularity > 0.5:
                    f_wave = 0.05 * np.sin(2 * np.pi * 6.5 * ti) + 0.02 * np.cos(2 * np.pi * 14.2 * ti)
                    val = f_wave + q_wave + r_wave + s_wave + t_wave + st_wave
                elif state == PatientState.AFLUTTER:
                    # Sawtooth pattern (AFlutter)
                    flutter = 0.08 * (2 * (ti * 4.0 - math.floor(ti * 4.0 + 0.5)))
                    val = flutter + q_wave + r_wave + s_wave + t_wave + st_wave
                else:
                    val = p_wave + q_wave + r_wave + s_wave + t_wave + st_wave
            
            ecg[i] = val

            # Increment the phase for the next sample
            profile._ecg_phase = (profile._ecg_phase + (hr / 60.0) / sr) % 1.0

    # Add realistic noise layers
    ecg = _add_baseline_wander(ecg, sr, freq=profile.ecg_baseline_wander_hz, amp=profile.ecg_baseline_wander_amp, t_start=t_start)
    ecg = _add_powerline(ecg, sr, amp=0.005, t_start=t_start)
    ecg = _add_gaussian_noise(ecg, sigma=profile.ecg_noise_amplitude)
    ecg = _add_motion_artifact(ecg, sr, prob_per_second=0.002)

    return ecg.astype(np.float32)


def generate_ppg_segment(hr: float, duration_s: float, sr: int = SAMPLING_RATE,
                          profile: Optional[PatientProfile] = None,
                          rr: Optional[float] = None,
                          state: PatientState = PatientState.STABLE,
                          ppg_amplitude_factor: float = 1.0,
                          t_start: float = 0.0) -> np.ndarray:
    """
    Generates a phase-continuous, mathematically realistic PPG (photoplethysmogram) waveform segment.
    Features physiological respiratory modulation and a distinct dicrotic notch.
    """
    if profile is None:
        profile = PatientProfile()

    n_samples = int(duration_s * sr)
    t = t_start + np.arange(n_samples) / sr
    ppg = np.zeros(n_samples, dtype=np.float32)

    if rr is None:
        rr = profile.rr_baseline

    # Initialize PPG phase if not present
    if not hasattr(profile, '_ppg_phase'):
        profile._ppg_phase = (t_start * (hr / 60.0)) % 1.0

    # In VFIB or VTach (high severity), peripheral perfusion drops to near-zero flatline
    if state == PatientState.VFIB or (state == PatientState.VTACH and ppg_amplitude_factor < 0.15):
        ppg = np.random.normal(0, 0.005, n_samples)
        # Advance the underlying phase to prevent freeze
        profile._ppg_phase = (profile._ppg_phase + n_samples * (hr / 60.0) / sr) % 1.0
    else:
        # Standard photoplethysmographic wave
        resp_freq = rr / 60.0
        for i, ti in enumerate(t):
            phase = profile._ppg_phase % 1.0
            
            # Primary systolic pulse + secondary diastolic pulse (separated by dicrotic notch)
            systolic = 0.82 * np.exp(-((phase - 0.25) / 0.075) ** 2)
            diastolic = 0.24 * np.exp(-((phase - 0.44) / 0.11) ** 2)
            pulse = (systolic + diastolic) * ppg_amplitude_factor
            
            # Respiratory amplitude modulation & baseline wander
            amp_mod = 1.0 + 0.12 * np.sin(2 * np.pi * resp_freq * ti)
            base_mod = 0.06 * np.sin(2 * np.pi * resp_freq * ti)
            
            ppg[i] = pulse * amp_mod + base_mod

            # Increment phase
            profile._ppg_phase = (profile._ppg_phase + (hr / 60.0) / sr) % 1.0

    # Add minor noise
    ppg = _add_baseline_wander(ppg, sr, freq=0.06, amp=0.04, t_start=t_start)
    ppg = _add_gaussian_noise(ppg, sigma=profile.ppg_noise_amplitude)
    ppg = _add_motion_artifact(ppg, sr, prob_per_second=0.002)

    return ppg.astype(np.float32)


def generate_rsp_segment(rr: float, duration_s: float, sr: int = SAMPLING_RATE,
                          profile: Optional[PatientProfile] = None,
                          ie_ratio: float = 1.5,
                          t_start: float = 0.0) -> np.ndarray:
    """
    Generates a phase-continuous, breathing wave with configurable inspiration-to-expiration (I:E) ratio.
    """
    if profile is None:
        profile = PatientProfile()

    n_samples = int(duration_s * sr)
    t = t_start + np.arange(n_samples) / sr
    resp_freq = rr / 60.0
    
    # Initialize RSP phase if not present
    if not hasattr(profile, '_rsp_phase'):
        profile._rsp_phase = (t_start * resp_freq) % 1.0

    # Configurable I:E ratio phase warping
    insp_frac = 1.0 / (1.0 + ie_ratio)

    rsp = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        phase = profile._rsp_phase % 1.0
        
        if phase < insp_frac:
            # Inspiration: smooth rise
            w_phase = 0.5 * (phase / insp_frac)
        else:
            # Expiration: smooth decay
            w_phase = 0.5 + 0.5 * ((phase - insp_frac) / (1.0 - insp_frac))
            
        val = 0.5 - 0.5 * math.cos(2.0 * math.pi * w_phase)
        rsp[i] = 0.9 * val - 0.45
        
        # Increment phase
        profile._rsp_phase = (profile._rsp_phase + resp_freq / sr) % 1.0

    # Apply minor wander and clean noise
    rsp = _add_baseline_wander(rsp, sr, freq=0.03, amp=0.01, t_start=t_start)
    rsp = _add_gaussian_noise(rsp, sigma=profile.rsp_noise_amplitude * 0.25)
    
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
        self._temp_core = profile.temp_core_baseline
        self._temp_skin = profile.temp_skin_baseline

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
            self._temp_core = self._random_walk(
                self._temp_core, p.temp_core_baseline, p.temp_core_variability,
                34.0, 42.0, dt)
            self._temp_skin = self._random_walk(
                self._temp_skin, p.temp_skin_baseline, p.temp_skin_variability,
                30.0, 39.0, dt)

            # Apply state modifiers (with ramp scaling)
            hr = float(np.clip(self._hr + mods["hr_delta"], 30, 200))
            
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

            rr = float(np.clip(self._rr + mods["rr_delta"], 6, 40))

            # SpO2 is also coupled to RR (hypoxia link)
            rr_spo2_coupling = -max(0, (rr - p.rr_baseline) * 0.12)
            spo2 = float(np.clip(
                self._spo2 + mods["spo2_delta"] + rr_spo2_coupling,
                min_spo2, 100))

            # Apply temperature modifiers
            temp_core = float(np.clip(self._temp_core + mods.get("temp_core_delta", 0.0), 33.0, 43.0))
            temp_skin = float(np.clip(self._temp_skin + mods.get("temp_skin_delta", 0.0), 28.0, 41.0))

            # MAP (mean arterial pressure)
            map_val = dbp + (sbp - dbp) / 3.0

            return {
                "heart_rate": round(hr, 1),
                "systolic_bp": round(sbp, 1),
                "diastolic_bp": round(dbp, 1),
                "map": round(map_val, 1),
                "spo2": round(spo2, 1),
                "respiratory_rate": round(rr, 1),
                "core_temperature": round(temp_core, 2),
                "skin_temperature": round(temp_skin, 2),
                "patient_state": state.value,
                "episode_label": episode.label if episode else None,
                "episode_severity": episode.severity if episode else None,
                "episode_ramp": round(ramp, 3),
                "pvc_burden": round(mods.get("pvc_burden", 0.0), 3),
                "pac_burden": round(mods.get("pac_burden", 0.0), 3),
                "irregularity": round(mods.get("irregularity", 0.0), 3),
                "st_delta": round(mods.get("st_delta", 0.0), 3),
                "ppg_amplitude_factor": round(mods.get("ppg_amplitude_factor", 1.0), 3),
                "ie_ratio_factor": round(mods.get("ie_ratio_factor", 1.0), 3),
                "timestamp": time.time(),
            }
