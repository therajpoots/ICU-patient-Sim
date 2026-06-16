"""
Waveform signal API routes — ECG, PPG, Respiratory.
"""

import logging
from fastapi import APIRouter, Depends, Query

from api.dependencies import verify_api_key
from api.crypto import encrypted_response
from simulator.engine import get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/signals", tags=["Waveforms"])


@router.get("/ecg", dependencies=[Depends(verify_api_key)])
async def get_ecg(seconds: int = Query(default=10, ge=1, le=30,
                                        description="Seconds of ECG to return")):
    """
    Returns the last N seconds of ECG waveform samples (250 Hz).
    Each sample is a float32 amplitude value.
    Response is AES-256-GCM encrypted.
    """
    engine = get_engine()
    waveform = engine.get_ecg_waveform(seconds=seconds)
    return encrypted_response({
        "signal": "ecg",
        "sampling_rate_hz": 250,
        "duration_s": seconds,
        "n_samples": len(waveform),
        "lead": "Lead-II equivalent",
        "samples": waveform,
    })


@router.get("/ppg", dependencies=[Depends(verify_api_key)])
async def get_ppg(seconds: int = Query(default=10, ge=1, le=30,
                                        description="Seconds of PPG to return")):
    """
    Returns the last N seconds of PPG (photoplethysmogram) waveform.
    Response is AES-256-GCM encrypted.
    """
    engine = get_engine()
    waveform = engine.get_ppg_waveform(seconds=seconds)
    return encrypted_response({
        "signal": "ppg",
        "sampling_rate_hz": 250,
        "duration_s": seconds,
        "n_samples": len(waveform),
        "samples": waveform,
    })


@router.get("/rsp", dependencies=[Depends(verify_api_key)])
async def get_rsp(seconds: int = Query(default=10, ge=1, le=30,
                                        description="Seconds of RSP to return")):
    """
    Returns the last N seconds of Respiratory waveform.
    Response is AES-256-GCM encrypted.
    """
    engine = get_engine()
    waveform = engine.get_rsp_waveform(seconds=seconds)
    return encrypted_response({
        "signal": "respiratory",
        "sampling_rate_hz": 250,
        "duration_s": seconds,
        "n_samples": len(waveform),
        "samples": waveform,
    })
