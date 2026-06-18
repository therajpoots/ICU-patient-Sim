"""
Main Simulation Engine
Runs the signal generation loop in a background thread.
Maintains rolling waveform buffers and a shared state object
accessible by the API and monitor layers.
"""

import time
import threading
import logging
import os
from collections import deque
from typing import Optional, Dict, Any, List

import numpy as np

from simulator.patient_profile import PatientProfile, load_patient_from_env
from simulator.arrhythmia_states import ArrhythmiaStateMachine, PatientState
from simulator.vitals import VitalsEngine, generate_ecg_segment, generate_ppg_segment, generate_rsp_segment

logger = logging.getLogger(__name__)

SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "250"))
BUFFER_SECONDS = int(os.getenv("WAVEFORM_BUFFER_SECONDS", "30"))
SIM_SPEED = float(os.getenv("SIM_SPEED", "1.0"))

# Segment length generated per tick (seconds of waveform)
SEGMENT_DURATION_S = 1.0


class SimulatorEngine:
    """
    Central simulation engine.
    Runs a background thread that every ~1 second:
      1. Updates vital signs via random walk + arrhythmia state
      2. Generates 1-second ECG, PPG, RSP waveform segments
      3. Appends to rolling waveform buffers
      4. Updates shared 'latest' vital snapshot (thread-safe)
    """

    def __init__(self, profile: Optional[PatientProfile] = None):
        self.profile = profile or load_patient_from_env()
        self.state_machine = ArrhythmiaStateMachine(
            afib_history=self.profile.has_afib_history
        )
        self.vitals_engine = VitalsEngine(self.profile, self.state_machine)

        max_buf = BUFFER_SECONDS * SAMPLING_RATE

        # Rolling waveform buffers
        self._ecg_buf: deque = deque(maxlen=max_buf)
        self._ppg_buf: deque = deque(maxlen=max_buf)
        self._rsp_buf: deque = deque(maxlen=max_buf)

        # Vital sign history (1 sample per second)
        self._vitals_history: deque = deque(maxlen=3600 * 13)  # 13 hours

        # Latest snapshot
        self._latest_vitals: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # Timing
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session_start = time.time()
        self._tick_count = 0

        # Callbacks (for monitor)
        self._vitals_callbacks: List = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the simulation loop in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sim-engine")
        self._thread.start()
        logger.info(f"Simulator started for patient: {self.profile.name} ({self.profile.patient_id})")

    def stop(self):
        """Stop the simulation loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Simulator stopped.")

    def register_vitals_callback(self, fn):
        """Register a callback called with each new vital snapshot."""
        self._vitals_callbacks.append(fn)

    def get_latest_vitals(self) -> Dict[str, Any]:
        """Thread-safe access to latest vital snapshot."""
        with self._lock:
            return dict(self._latest_vitals)

    def get_ecg_waveform(self, seconds: int = 10) -> List[float]:
        """Return last N seconds of ECG as a list."""
        n = min(seconds * SAMPLING_RATE, len(self._ecg_buf))
        with self._lock:
            return list(self._ecg_buf)[-n:]

    def get_ppg_waveform(self, seconds: int = 10) -> List[float]:
        """Return last N seconds of PPG as a list."""
        n = min(seconds * SAMPLING_RATE, len(self._ppg_buf))
        with self._lock:
            return list(self._ppg_buf)[-n:]

    def get_rsp_waveform(self, seconds: int = 10) -> List[float]:
        """Return last N seconds of RSP as a list."""
        n = min(seconds * SAMPLING_RATE, len(self._rsp_buf))
        with self._lock:
            return list(self._rsp_buf)[-n:]

    def get_vitals_history(self, seconds: int = 43200) -> List[Dict]:
        """Return vital sign history for the last N seconds (default 12 hours)."""
        cutoff = time.time() - seconds
        with self._lock:
            return [v for v in self._vitals_history if v["timestamp"] >= cutoff]

    def get_session_info(self) -> Dict[str, Any]:
        """Return metadata about the current simulation session."""
        return {
            "patient_id": self.profile.patient_id,
            "patient_name": self.profile.name,
            "patient_age": self.profile.age,
            "ward": self.profile.ward,
            "session_start": self._session_start,
            "session_duration_s": time.time() - self._session_start,
            "tick_count": self._tick_count,
            "sampling_rate": SAMPLING_RATE,
            "buffer_seconds": BUFFER_SECONDS,
            "next_episode_in_s": self.state_machine.seconds_until_next,
            "episode_history_count": len(self.state_machine.episode_history),
        }

    def trigger_anomaly(self, state_name: Optional[str] = None) -> dict:
        """Trigger an anomaly on the patient state machine."""
        with self._lock:
            episode = self.state_machine.trigger_anomaly(state_name)
            # Immediately update vitals snapshot so the API reflects it
            vitals = self.vitals_engine.update(0.0)
            self._latest_vitals = dict(vitals)
            return {
                "triggered": True,
                "state": episode.state.value,
                "label": episode.label,
                "duration_s": round(episode.duration_s, 1),
                "severity": episode.severity,
            }

    # ------------------------------------------------------------------
    # Internal Loop
    # ------------------------------------------------------------------

    def _loop(self):
        """Main simulation tick loop."""
        tick_duration = SEGMENT_DURATION_S / SIM_SPEED
        last_tick = time.time()

        while self._running:
            now = time.time()
            dt = now - last_tick
            last_tick = now

            try:
                self._tick(dt)
            except Exception as exc:
                logger.error(f"Simulation tick error: {exc}", exc_info=True)

            # Sleep remainder of tick duration
            elapsed = time.time() - now
            sleep_time = max(0, tick_duration - elapsed)
            time.sleep(sleep_time)

    def _tick(self, dt: float):
        """Single simulation step."""
        # 1. Update vitals
        vitals = self.vitals_engine.update(dt)

        hr = vitals["heart_rate"]
        rr = vitals["respiratory_rate"]
        pvc_burden = vitals["pvc_burden"]
        irregularity = vitals["irregularity"]
        state = PatientState(vitals["patient_state"])

        # 2. Generate waveform segments
        ecg = generate_ecg_segment(
            hr=hr,
            duration_s=SEGMENT_DURATION_S,
            sr=SAMPLING_RATE,
            profile=self.profile,
            state=state,
            pvc_burden=pvc_burden,
            irregularity=irregularity,
        )
        ppg = generate_ppg_segment(hr=hr, duration_s=SEGMENT_DURATION_S,
                                   sr=SAMPLING_RATE, profile=self.profile, rr=rr, state=state)
        rsp = generate_rsp_segment(rr=rr, duration_s=SEGMENT_DURATION_S,
                                   sr=SAMPLING_RATE, profile=self.profile)

        # 3. Update buffers
        with self._lock:
            self._ecg_buf.extend(ecg.tolist())
            self._ppg_buf.extend(ppg.tolist())
            self._rsp_buf.extend(rsp.tolist())
            self._latest_vitals = dict(vitals)
            self._vitals_history.append(dict(vitals))
            self._tick_count += 1

        # 4. Fire callbacks
        for cb in self._vitals_callbacks:
            try:
                cb(dict(vitals))
            except Exception as exc:
                logger.warning(f"Vitals callback error: {exc}")


# Global singleton instance (initialized at startup)
_engine_instance: Optional[SimulatorEngine] = None


def get_engine() -> SimulatorEngine:
    """Return the global simulator engine instance."""
    global _engine_instance
    if _engine_instance is None:
        raise RuntimeError("Simulator engine not initialized. Call init_engine() first.")
    return _engine_instance


def init_engine(profile: Optional[PatientProfile] = None) -> SimulatorEngine:
    """Initialize and start the global simulator engine."""
    global _engine_instance
    _engine_instance = SimulatorEngine(profile=profile)
    _engine_instance.start()
    return _engine_instance
