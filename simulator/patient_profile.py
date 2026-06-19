"""
Patient Profile - defines the baseline vital parameters for the simulated ICU patient.
This profile is used across all modules to anchor the simulation to realistic clinical values.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class PatientProfile:
    """
    Represents a simulated ICU patient with baseline vital ranges.
    All simulation variability is computed relative to these baselines.
    """

    # Identity
    patient_id: str = "ICU-2026-001"
    name: str = "John Doe"
    age: int = 67
    ward: str = "NICU-3B"
    weight_kg: float = 78.0
    height_cm: float = 175.0

    # Cardiac baselines
    hr_baseline: float = 72.0          # bpm - resting heart rate
    hr_min: float = 52.0               # normal minimum (sinus)
    hr_max: float = 105.0              # normal maximum (sinus)
    hr_variability: float = 6.0        # ± bpm random walk SD per minute

    # Blood pressure baselines (mmHg)
    sbp_baseline: float = 122.0        # systolic
    dbp_baseline: float = 78.0         # diastolic
    sbp_variability: float = 4.0       # ± mmHg SD per minute
    dbp_variability: float = 2.5

    # Respiratory
    rr_baseline: float = 16.0          # breaths/min
    rr_min: float = 10.0
    rr_max: float = 22.0
    rr_variability: float = 1.5        # ± br/min SD per minute

    # SpO2
    spo2_baseline: float = 97.0        # %
    spo2_min: float = 92.0             # acceptable floor (before alert)
    spo2_variability: float = 0.5      # ± % SD per minute

    # Temperature baselines (°C)
    temp_core_baseline: float = 37.0
    temp_skin_baseline: float = 35.5
    temp_core_variability: float = 0.05
    temp_skin_variability: float = 0.07

    # Signal noise characteristics (realistic Holter-like)
    ecg_noise_amplitude: float = 0.02       # μV equivalent
    ecg_baseline_wander_hz: float = 0.1     # Hz of slow drift
    ecg_baseline_wander_amp: float = 0.08   # amplitude of drift
    ppg_noise_amplitude: float = 0.015
    rsp_noise_amplitude: float = 0.03

    # Comorbidities (affect simulation behaviour)
    has_hypertension: bool = True       # elevates BP baseline slightly
    has_afib_history: bool = False      # increases AFib episode probability
    is_on_beta_blocker: bool = True     # dampens HR response

    def __post_init__(self):
        if self.has_hypertension:
            self.sbp_baseline += 10
            self.dbp_baseline += 6
        if self.is_on_beta_blocker:
            self.hr_baseline = max(55, self.hr_baseline - 8)
            self.hr_variability *= 0.7


def load_patient_from_env() -> PatientProfile:
    """Load patient profile from environment variables."""
    return PatientProfile(
        patient_id=os.getenv("PATIENT_ID", "ICU-2026-001"),
        name=os.getenv("PATIENT_NAME", "John Doe"),
        age=int(os.getenv("PATIENT_AGE", "67")),
        ward=os.getenv("PATIENT_WARD", "NICU-3B"),
    )
