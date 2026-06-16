"""
Vitals API routes
Provides current vital sign snapshot, SSE live stream, and episode status.
All responses are AES-256-GCM encrypted.
"""

import time
import asyncio
import json
import logging
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from api.dependencies import verify_api_key
from api.crypto import encrypted_response
from simulator.engine import get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vitals", tags=["Vitals"])


@router.get(
    "/current",
    dependencies=[Depends(verify_api_key)],
    summary="Get latest vital signs snapshot",
)
async def get_current_vitals():
    """
    Returns the most recent 1-second vital sign snapshot.

    **Decrypted payload contains:**
    - `heart_rate` — bpm (float)
    - `systolic_bp` / `diastolic_bp` / `map` — mmHg (float)
    - `spo2` — % (float)
    - `respiratory_rate` — breaths/min (float)
    - `patient_state` — current clinical state string
    - `episode_label` — human-readable name if episode active, else null
    - `episode_severity` — "mild" | "moderate" | "severe" | null
    - `episode_ramp` — 0.0–1.0, smooth onset/offset factor
    - `pvc_burden` — fraction of beats that are PVCs (0.0–1.0)
    - `irregularity` — 0.0 = regular, 1.0 = fully irregular (AFib)
    - `timestamp` — Unix epoch seconds

    **Response is encrypted**: `{ "encrypted": true, "algorithm": "AES-256-GCM", "payload": "<base64>" }`
    """
    engine = get_engine()
    vitals = engine.get_latest_vitals()
    return encrypted_response(vitals)


@router.get(
    "/episode",
    dependencies=[Depends(verify_api_key)],
    summary="Get current arrhythmia episode status",
)
async def get_episode_status():
    """
    Returns the current arrhythmia/clinical episode status.
    Your monitoring system can poll this to know if a deterioration episode is active.

    **Decrypted payload:**
    - `episode_active` — bool
    - `state` — current patient state
    - `episode_label` — episode name or null
    - `severity` — "mild" | "moderate" | "severe" | null
    - `ramp_factor` — smooth 0→1→0 transition factor
    - `next_episode_in_s` — approx. seconds until next episode (0 if active)
    - `episodes_history_count` — total episodes since session started
    """
    engine = get_engine()
    v = engine.get_latest_vitals()
    info = engine.get_session_info()
    return encrypted_response({
        "episode_active":        v.get("episode_label") is not None,
        "state":                 v.get("patient_state", "stable"),
        "episode_label":         v.get("episode_label"),
        "severity":              v.get("episode_severity"),
        "ramp_factor":           v.get("episode_ramp", 0.0),
        "pvc_burden":            v.get("pvc_burden", 0.0),
        "irregularity":          v.get("irregularity", 0.0),
        "next_episode_in_s":     info["next_episode_in_s"],
        "episodes_history_count":info["episode_history_count"],
        "timestamp":             v.get("timestamp", time.time()),
    })


@router.get(
    "/history",
    dependencies=[Depends(verify_api_key)],
    summary="Get in-memory vital signs history",
)
async def get_vitals_history(seconds: int = 3600):
    """
    Returns up to `seconds` seconds of vital sign history from the in-memory buffer
    (max ~13 hours). Useful for your monitoring system to backfill on reconnect.

    **Query param:** `seconds` (default: 3600 = 1 hour, max: 46800 = 13 hours)
    """
    seconds = min(seconds, 46800)
    engine = get_engine()
    history = engine.get_vitals_history(seconds=seconds)
    return encrypted_response({
        "requested_seconds": seconds,
        "records_returned":  len(history),
        "oldest_timestamp":  history[0]["timestamp"] if history else None,
        "newest_timestamp":  history[-1]["timestamp"] if history else None,
        "vitals": history,
    })


@router.get(
    "/stream",
    summary="SSE live stream of vital signs (1 update/second)",
)
async def stream_vitals(x_api_key: str = Depends(verify_api_key)):
    """
    **Server-Sent Events (SSE)** — streams one encrypted vital snapshot per second.
    Keep this connection open in your monitoring system for continuous data.

    Each SSE event:
    - `event: vitals`
    - `data: { "encrypted": true, "algorithm": "AES-256-GCM", "payload": "<base64>" }`

    Decrypt `payload` with your AES-256-GCM key to get the vitals dict.

    **curl example:**
    ```
    curl -N -H "X-API-Key: <key>" https://localhost:8443/api/v1/vitals/stream
    ```
    """
    engine = get_engine()

    async def event_generator():
        while True:
            try:
                vitals = engine.get_latest_vitals()
                if vitals:
                    payload = encrypted_response(vitals)
                    yield {
                        "event": "vitals",
                        "data": json.dumps(payload),
                    }
            except Exception as exc:
                logger.error(f"SSE stream error: {exc}")
                yield {"event": "error", "data": json.dumps({"error": str(exc)})}
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())
