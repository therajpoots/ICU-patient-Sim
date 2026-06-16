"""
Decryption Client Helper
Provides functions to query the encrypted ICU API and decrypt responses.
Can be used as a library or run standalone.

Usage as library:
    from client_example.decrypt_client import ICUClient
    client = ICUClient()
    vitals = client.get_vitals()

Usage standalone:
    python -m client_example.decrypt_client
"""

import os
import sys
import json
import base64
import httpx
from pathlib import Path

# Add parent to path so we can import api.crypto
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from api.crypto import decrypt_payload

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "https://localhost:8443")
DEFAULT_API_KEY  = os.getenv("API_KEY", "icu-monitor-dev-key-change-in-production")


class ICUClient:
    """
    Client for the ICU Biosignal Pipeline REST API.
    Automatically decrypts AES-256-GCM encrypted responses.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        verify_ssl: bool = False,   # False for self-signed certs
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self._client = httpx.Client(
            verify=verify_ssl,
            headers=self.headers,
            timeout=15.0,
        )

    def _get(self, path: str, params: dict = None) -> dict:
        """Make a GET request and decrypt the response."""
        url = f"{self.base_url}{path}"
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("encrypted"):
            return decrypt_payload(data["payload"])
        return data

    def _post(self, path: str, params: dict = None) -> dict:
        """Make a POST request and decrypt the response."""
        url = f"{self.base_url}{path}"
        resp = self._client.post(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("encrypted"):
            return decrypt_payload(data["payload"])
        return data

    # ── Public methods ────────────────────────────────────────

    def health_check(self) -> dict:
        """Check server health (no auth required, no encryption)."""
        resp = httpx.get(f"{self.base_url}/api/v1/health", verify=False)
        return resp.json()

    def get_vitals(self) -> dict:
        """Get the latest vital signs snapshot."""
        return self._get("/api/v1/vitals/current")

    def get_ecg(self, seconds: int = 10) -> dict:
        """Get ECG waveform samples."""
        return self._get("/api/v1/signals/ecg", params={"seconds": seconds})

    def get_ppg(self, seconds: int = 10) -> dict:
        """Get PPG waveform samples."""
        return self._get("/api/v1/signals/ppg", params={"seconds": seconds})

    def get_rsp(self, seconds: int = 10) -> dict:
        """Get respiratory waveform samples."""
        return self._get("/api/v1/signals/rsp", params={"seconds": seconds})

    def get_episode_status(self) -> dict:
        """Get current arrhythmia / deterioration episode status."""
        return self._get("/api/v1/vitals/episode")

    def get_history(self, seconds: int = 3600) -> dict:
        """Get in-memory vital signs history for the last N seconds."""
        return self._get("/api/v1/vitals/history", params={"seconds": seconds})

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  ICU Biosignal Simulator — API Client Demo")
    print("=" * 55)

    with ICUClient() as client:
        # 1. Health check
        print("\n[1] Health Check")
        try:
            h = client.health_check()
            print(f"    Status   : {h.get('status')}")
            print(f"    Session  : {h.get('session_duration_s', 0):.0f}s running")
            print(f"    Security : {h.get('transport_security')} + {h.get('payload_encryption')}")
        except Exception as e:
            print(f"    ❌ {e}\n    → Is the server running? (python run_simulator.py --http)")
            sys.exit(1)

        # 2. Current vitals
        print("\n[2] Current Vitals (decrypted from AES-256-GCM payload)")
        v = client.get_vitals()
        print(f"    Heart Rate  : {v.get('heart_rate')} bpm")
        print(f"    BP          : {v.get('systolic_bp')}/{v.get('diastolic_bp')} mmHg  (MAP: {v.get('map')})")
        print(f"    SpO₂        : {v.get('spo2')}%")
        print(f"    Resp. Rate  : {v.get('respiratory_rate')} br/min")
        print(f"    Patient State: {v.get('patient_state')}")

        # 3. Episode status
        print("\n[3] Arrhythmia Episode Status")
        ep = client.get_episode_status()
        if ep.get("episode_active"):
            print(f"    ⚠️  ACTIVE: {ep['episode_label']} ({ep['severity']})")
            print(f"    Ramp factor: {ep['ramp_factor']} | Irregularity: {ep['irregularity']}")
        else:
            print(f"    ✅ No active episode (next in ~{ep.get('next_episode_in_s', '?'):.0f}s)")

        # 4. ECG waveform
        print("\n[4] ECG Waveform (5 seconds = 1250 samples @ 250 Hz)")
        ecg = client.get_ecg(seconds=5)
        samples = ecg.get("samples", [])
        print(f"    Received  : {len(samples)} samples")
        print(f"    Min/Max   : {min(samples):.4f} / {max(samples):.4f}")
        print(f"    First 5   : {[round(s, 4) for s in samples[:5]]}")

        # 5. PPG
        print("\n[5] PPG (3 seconds)")
        ppg = client.get_ppg(seconds=3)
        print(f"    Received  : {len(ppg.get('samples', []))} samples")

        # 6. Respiratory
        print("\n[6] Respiratory (3 seconds)")
        rsp = client.get_rsp(seconds=3)
        print(f"    Received  : {len(rsp.get('samples', []))} samples")

    print("\n✅ Client demo complete.")
