"""
FastAPI Application — Main Entry Point
Exposes the ICU biosignal simulator over HTTPS (TLS) with AES-256-GCM
payload encryption on all API endpoints.

Your monitoring system connects to these endpoints to get live data.
"""

import os
import time
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from api.routes import vitals, signals
from simulator.engine import init_engine, get_engine
from simulator.patient_profile import load_patient_from_env

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  ICU Biosignal Simulator — Starting")
    logger.info("=" * 60)

    profile = load_patient_from_env()
    engine = init_engine(profile)

    logger.info(f"Patient  : {profile.name}  |  ID: {profile.patient_id}")
    logger.info(f"Ward     : {profile.ward}  |  Age: {profile.age}")
    logger.info(f"Baseline : HR={profile.hr_baseline} bpm, "
                f"BP={profile.sbp_baseline}/{profile.dbp_baseline} mmHg, "
                f"SpO2={profile.spo2_baseline}%")
    logger.info("Simulator online — streaming biosignals.")
    logger.info("=" * 60)

    yield

    engine.stop()
    logger.info("Simulator stopped.")


# ─────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ICU Biosignal Simulator API",
    description=(
        "Secure REST API streaming realistic ICU patient biosignals.\n\n"
        "**Authentication**: All `/api/v1/*` endpoints require `X-API-Key` header.\n\n"
        "**Encryption**: All responses are AES-256-GCM encrypted on top of TLS transport.\n\n"
        "**Signals available**: ECG (250 Hz, single lead), PPG (250 Hz), Respiratory (250 Hz), "
        "SpO₂, Heart Rate, Blood Pressure (systolic/diastolic/MAP).\n\n"
        "**Arrhythmia simulation**: Patient undergoes random cardiac episodes every 2–4 hours "
        "(PVCs, AFib, Tachycardia, Bradycardia, SpO₂ desaturation, hypertensive spikes)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────
app.include_router(vitals.router,  prefix="/api/v1")
app.include_router(signals.router, prefix="/api/v1")


# ─────────────────────────────────────────────────────────
# System Endpoints (no auth)
# ─────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["System"], summary="Health check (no auth needed)")
async def health_check():
    """
    Public health check — no API key required.
    Returns server status, patient ID, session duration, and encryption info.
    """
    try:
        engine = get_engine()
        info = engine.get_session_info()
        return {
            "status": "ok",
            "server_time_utc": time.time(),
            "patient_id": info["patient_id"],
            "patient_name": info["patient_name"],
            "ward": info["ward"],
            "session_duration_s": round(info["session_duration_s"], 1),
            "ticks_generated": info["tick_count"],
            "sampling_rate_hz": info["sampling_rate"],
            "waveform_buffer_s": info["buffer_seconds"],
            "next_arrhythmia_episode_in_s": round(info["next_episode_in_s"], 0),
            "total_episodes_so_far": info["episode_history_count"],
            "transport_security": "TLS 1.3",
            "payload_encryption": "AES-256-GCM",
        }
    except RuntimeError:
        return JSONResponse(status_code=503, content={"status": "starting_up"})


@app.get("/api/v1/patient", tags=["System"], summary="Patient profile info (no auth needed)")
async def patient_info():
    """Returns the simulated patient's profile metadata (not encrypted — not PHI)."""
    try:
        engine = get_engine()
        info = engine.get_session_info()
        return {
            "patient_id":   info["patient_id"],
            "patient_name": info["patient_name"],
            "patient_age":  info["patient_age"],
            "ward":         info["ward"],
            "simulation_speed": os.getenv("SIM_SPEED", "1.0"),
            "signals": ["ECG (Lead-II equivalent)", "PPG", "Respiratory"],
            "derived_vitals": ["Heart Rate (bpm)", "SpO₂ (%)",
                               "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "MAP (mmHg)",
                               "Respiratory Rate (br/min)"],
            "arrhythmia_types": [
                "Premature Ventricular Contractions (PVC)",
                "Sinus Tachycardia",
                "Sinus Bradycardia",
                "Atrial Fibrillation",
                "SpO₂ Desaturation",
                "Hypertensive Spike",
                "Respiratory Distress",
            ],
        }
    except RuntimeError:
        return JSONResponse(status_code=503, content={"status": "starting_up"})


class AnomalyRequest(BaseModel):
    state: str

# ─────────────────────────────────────────────────────────
# Dashboard Endpoints (unencrypted for local UI)
# ─────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"], summary="Live bedside monitor dashboard")
async def dashboard():
    """Serves the unencrypted bedside clinical monitor HTML dashboard."""
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h2>Dashboard Template Not Found</h2>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/dashboard/api/stream", tags=["Dashboard"], summary="Unencrypted vital signs SSE stream")
async def dashboard_stream():
    """Streams live vital signs unencrypted for the browser dashboard."""
    engine = get_engine()
    async def event_generator():
        while True:
            try:
                vitals = engine.get_latest_vitals()
                if vitals:
                    yield {
                        "event": "vitals",
                        "data": json.dumps(vitals)
                    }
            except Exception as exc:
                yield {"event": "error", "data": json.dumps({"error": str(exc)})}
            await asyncio.sleep(1.0)
    return EventSourceResponse(event_generator())


@app.get("/dashboard/api/waveforms", tags=["Dashboard"], summary="Unencrypted waveform data")
async def dashboard_waveforms(seconds: int = 5):
    """Returns unencrypted waveform segments for charting in the browser."""
    seconds = min(seconds, 30)
    engine = get_engine()
    return {
        "sampling_rate_hz": 250,
        "ecg": engine.get_ecg_waveform(seconds=seconds),
        "ppg": engine.get_ppg_waveform(seconds=seconds),
        "rsp": engine.get_rsp_waveform(seconds=seconds),
    }


@app.post("/dashboard/api/anomaly", tags=["Dashboard"], summary="Unencrypted anomaly injector trigger")
async def dashboard_trigger_anomaly(req: AnomalyRequest):
    """Force-injects a patient anomaly/arrhythmia state from the dashboard."""
    engine = get_engine()
    state_val = req.state if req.state != "random" else None
    
    # Restores stable state if 'stable' or 'normal' is passed
    if state_val in ("stable", "normal"):
        with engine._lock:
            engine.state_machine.current_episode = None
            vitals = engine.vitals_engine.update(0.0)
            engine._latest_vitals = dict(vitals)
        return {"triggered": True, "state": "stable", "label": "Normal sinus rhythm restored"}
        
    result = engine.trigger_anomaly(state_name=state_val)
    return result


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "ICU Biosignal Simulator",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "health": "/api/v1/health",
        "patient": "/api/v1/patient",
    }
