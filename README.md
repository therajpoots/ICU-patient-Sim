# ICU Biosignal Simulator

A realistic, real-time ICU patient biosignal simulator exposed via a secure REST API (TLS + AES-256-GCM encryption). Designed as a **data source** for your own ICU monitoring system.

---

## What It Simulates

The simulator generates continuous, physiologically realistic signals for a single ICU patient:

| Signal | Type | Rate | Notes |
|--------|------|------|-------|
| **ECG** | Waveform | 250 Hz | Single lead (Lead-II equivalent), ECGSYN model |
| **PPG** | Waveform | 250 Hz | Photoplethysmogram |
| **Respiratory** | Waveform | 250 Hz | Chest/airflow signal |
| **Heart Rate** | Derived | 1 Hz | Extracted from ECG R-peaks |
| **SpO₂** | Derived | 1 Hz | Correlated with respiratory rate |
| **Blood Pressure** | Derived | 1 Hz | Systolic / Diastolic / MAP |
| **Respiratory Rate** | Derived | 1 Hz | Breaths per minute |

### Realistic Signal Characteristics

Every signal includes ICU-grade realism:

- ✅ **Baseline wander** — slow sinusoidal drift (0.05–0.3 Hz)
- ✅ **Gaussian white noise** — continuous low-level noise floor
- ✅ **50 Hz powerline interference** — subtle mains noise
- ✅ **Motion artifacts** — sporadic bursts (simulating patient movement)
- ✅ **Ornstein–Uhlenbeck random walk** — vitals drift naturally around baseline (not flat)
- ✅ **Physiological coupling** — SpO₂ drops when RR rises; BP follows HR changes

### Arrhythmia / Deterioration Episodes

Every **2–4 hours** (Poisson-distributed), the patient randomly enters one of these states:

| Episode | HR Effect | BP Effect | SpO₂ Effect | Duration |
|---------|-----------|-----------|-------------|---------|
| **PVCs** | −3 bpm | −6 mmHg | −1% | 30s–2min |
| **Sinus Tachycardia** | +32 bpm | +12 mmHg | −2% | 1–5min |
| **Sinus Bradycardia** | −25 bpm | −14 mmHg | −3% | 45s–4min |
| **Atrial Fibrillation** | +28 bpm, irregular | −8 mmHg | −3% | 1.5–6min |
| **SpO₂ Desaturation** | +8 bpm | +5 mmHg | −6% | 30s–3min |
| **Hypertensive Spike** | +12 bpm | +32 mmHg | −0.5% | 1–5min |
| **Respiratory Distress** | +10 bpm | +6 mmHg | −4% | 1–4min |

All transitions are **smooth** (12-second ramp in/out) — no abrupt jumps.

---

## Quick Start

### 1. Install Dependencies

```bash
cd "E:\ICU pipeline"
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` and set:
```env
AES_SHARED_SECRET=your-very-long-random-secret-here
API_KEY=your-api-key-here
PATIENT_NAME=John Doe
PATIENT_ID=ICU-2026-001
PATIENT_WARD=NICU-3B
```

### 3. Start the Simulator

**Option A: HTTPS (recommended for production)**
```bash
python run_simulator.py
```
> TLS certificates are auto-generated on first run in `certs/`

**Option B: HTTP (for local dev/testing)**
```bash
python run_simulator.py --http
```

### 4. Verify It's Working

```bash
# Health check (no auth needed)
curl -k https://localhost:8443/api/v1/health

# Or with HTTP mode:
curl http://localhost:8080/api/v1/health
```

---

## API Reference

### Authentication

All `/api/v1/vitals/*` and `/api/v1/signals/*` endpoints require:
```
X-API-Key: <your-api-key-from-.env>
```

Public endpoints (no auth): `/api/v1/health`, `/api/v1/patient`, `/docs`

### Encryption

All authenticated endpoint responses are **double-encrypted**:
1. **TLS** — transport layer (HTTPS)
2. **AES-256-GCM** — application layer (payload)

Every response looks like:
```json
{
  "encrypted": true,
  "algorithm": "AES-256-GCM",
  "payload": "<base64-encoded: nonce(12) + ciphertext + tag(16)>"
}
```

**Decrypting in Python:**
```python
import base64, json, os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# Key derivation (same as server)
def derive_key(secret: str) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32,
        salt=b"icu-biosignal-pipeline-v1-salt-2026",
        iterations=260_000, backend=default_backend()
    )
    return kdf.derive(secret.encode())

key = derive_key(os.environ["AES_SHARED_SECRET"])

def decrypt(payload_b64: str):
    combined = base64.b64decode(payload_b64)
    nonce, ciphertext = combined[:12], combined[12:]
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext)
```

**Or use the included client helper:**
```python
from client_example.decrypt_client import ICUClient

with ICUClient() as client:
    vitals = client.get_vitals()
    print(vitals["heart_rate"])  # e.g. 74.3
```

---

### Endpoints

#### `GET /api/v1/health`
**No auth required.** Server status check.

```json
{
  "status": "ok",
  "patient_id": "ICU-2026-001",
  "session_duration_s": 3600.0,
  "next_arrhythmia_episode_in_s": 5423,
  "transport_security": "TLS 1.3",
  "payload_encryption": "AES-256-GCM"
}
```

---

#### `GET /api/v1/patient`
**No auth required.** Patient profile metadata.

```json
{
  "patient_id": "ICU-2026-001",
  "patient_name": "John Doe",
  "ward": "NICU-3B",
  "signals": ["ECG (Lead-II equivalent)", "PPG", "Respiratory"],
  "derived_vitals": ["Heart Rate (bpm)", "SpO₂ (%)", "Systolic BP (mmHg)", ...]
}
```

---

#### `GET /api/v1/vitals/current`
**Auth required.** Latest 1-second vital snapshot. **Encrypted.**

*Decrypted payload:*
```json
{
  "heart_rate": 74.3,
  "systolic_bp": 126.1,
  "diastolic_bp": 81.4,
  "map": 96.3,
  "spo2": 97.2,
  "respiratory_rate": 15.8,
  "patient_state": "stable",
  "episode_label": null,
  "episode_severity": null,
  "episode_ramp": 0.0,
  "pvc_burden": 0.0,
  "irregularity": 0.0,
  "timestamp": 1750000000.0
}
```

**Patient states:** `stable`, `pvc`, `sinus_tachycardia`, `sinus_bradycardia`, `atrial_fibrillation`, `spo2_desaturation`, `hypertensive_spike`, `respiratory_distress`

---

#### `GET /api/v1/vitals/episode`
**Auth required.** Current arrhythmia episode status. **Encrypted.**

*Decrypted payload:*
```json
{
  "episode_active": true,
  "state": "atrial_fibrillation",
  "episode_label": "Atrial Fibrillation",
  "severity": "moderate",
  "ramp_factor": 0.87,
  "pvc_burden": 0.0,
  "irregularity": 0.87,
  "next_episode_in_s": 0,
  "episodes_history_count": 3,
  "timestamp": 1750000000.0
}
```

> Poll this endpoint in your monitoring system to detect deterioration events.

---

#### `GET /api/v1/vitals/stream`
**Auth required.** Server-Sent Events — 1 update per second. **Each event encrypted.**

```bash
# With curl (self-signed cert — skip verification):
curl -N -k -H "X-API-Key: your-key" https://localhost:8443/api/v1/vitals/stream

# With HTTP mode:
curl -N -H "X-API-Key: your-key" http://localhost:8080/api/v1/vitals/stream
```

SSE event format:
```
event: vitals
data: {"encrypted":true,"algorithm":"AES-256-GCM","payload":"<base64>"}
```

---

#### `GET /api/v1/vitals/history?seconds=3600`
**Auth required.** In-memory vital history (up to 13 hours). **Encrypted.**

Useful to backfill your monitoring system on reconnect.

| Parameter | Default | Max | Description |
|-----------|---------|-----|-------------|
| `seconds` | 3600 | 46800 | How many seconds of history to return |

---

#### `GET /api/v1/signals/ecg?seconds=10`
**Auth required.** ECG waveform samples. **Encrypted.**

*Decrypted payload:*
```json
{
  "signal": "ecg",
  "sampling_rate_hz": 250,
  "duration_s": 10,
  "n_samples": 2500,
  "lead": "Lead-II equivalent",
  "samples": [0.0123, -0.0045, 0.0891, ...]
}
```

| Parameter | Default | Max |
|-----------|---------|-----|
| `seconds` | 10 | 30 |

---

#### `GET /api/v1/signals/ppg?seconds=10`
**Auth required.** PPG waveform. **Encrypted.** Same structure as ECG.

---

#### `GET /api/v1/signals/rsp?seconds=10`
**Auth required.** Respiratory waveform. **Encrypted.** Same structure as ECG.

---

### Interactive API Docs

When the server is running, visit:
- **Swagger UI**: https://localhost:8443/docs
- **ReDoc**: https://localhost:8443/redoc

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `AES_SHARED_SECRET` | *(required)* | Shared secret for AES-256 key derivation |
| `API_KEY` | `icu-monitor-dev-key-...` | API key for X-API-Key header |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8443` | HTTPS port |
| `USE_TLS` | `true` | Enable/disable TLS |
| `PATIENT_ID` | `ICU-2026-001` | Patient identifier |
| `PATIENT_NAME` | `John Doe` | Patient name |
| `PATIENT_AGE` | `67` | Patient age |
| `PATIENT_WARD` | `NICU-3B` | Ward identifier |
| `SAMPLING_RATE` | `250` | Hz for all waveforms |
| `WAVEFORM_BUFFER_SECONDS` | `30` | Rolling waveform buffer size |
| `SIM_SPEED` | `1.0` | 1.0 = real-time, 2.0 = 2× speed |
| `ARRHYTHMIA_MEAN_INTERVAL_S` | `9000` | Mean seconds between episodes (~2.5 hrs) |

---

## Using the Client Helper

The `client_example/decrypt_client.py` provides a ready-made client:

```python
from client_example.decrypt_client import ICUClient

with ICUClient(
    base_url="https://localhost:8443",
    api_key="your-api-key",
    verify_ssl=False,   # False for self-signed certs
) as client:

    # Health check
    h = client.health_check()
    print(h["status"])

    # Current vitals (auto-decrypted)
    v = client.get_vitals()
    print(f"HR: {v['heart_rate']} bpm | SpO2: {v['spo2']}%")

    # Episode status
    ep = client.get_episode_status()
    if ep["episode_active"]:
        print(f"ALERT: {ep['episode_label']} ({ep['severity']})")

    # ECG waveform (10 seconds = 2500 samples)
    ecg = client.get_ecg(seconds=10)
    samples = ecg["samples"]

    # PPG
    ppg = client.get_ppg(seconds=5)

    # Respiratory
    rsp = client.get_rsp(seconds=5)

    # History (last 1 hour of vitals)
    hist = client.get_history(seconds=3600)
    for record in hist["vitals"]:
        print(record["heart_rate"], record["spo2"])
```

---

## Testing the System

```bash
python -m client_example.test_system
```

Runs connectivity, auth, encryption validation, waveform, and data quality tests.

For a **live terminal monitor** (shows real-time vitals while running):
```bash
python -m client_example.test_system --monitor --monitor-time 60
```

---

## Project Structure

```
E:\ICU pipeline\
├── run_simulator.py          # ← Start here
├── setup_tls.py              # TLS cert generator (auto-called by runner)
├── requirements.txt
├── .env.example              # Copy to .env and configure
│
├── simulator/                # Core signal generation
│   ├── engine.py             # Main loop, waveform buffers, threading
│   ├── vitals.py             # ECG/PPG/RSP generation + noise layers
│   ├── arrhythmia_states.py  # Episode state machine (Poisson-timed)
│   └── patient_profile.py    # Patient baseline configuration
│
├── api/                      # REST API layer
│   ├── main.py               # FastAPI app + startup
│   ├── crypto.py             # AES-256-GCM encrypt/decrypt
│   ├── dependencies.py       # API key auth
│   └── routes/
│       ├── vitals.py         # /vitals/current, /stream, /history, /episode
│       └── signals.py        # /signals/ecg, /ppg, /rsp
│
├── client_example/
│   ├── decrypt_client.py     # Python API client with auto-decryption
│   └── test_system.py        # Comprehensive test suite
│
└── certs/                    # Auto-generated TLS certs (gitignore this)
```

---

## Integration Guide for Your Monitoring System

### Connecting to the Simulator

Your monitoring system should:

1. **Read config** — load `AES_SHARED_SECRET` and `API_KEY` from your secure config store
2. **Derive AES key** — use PBKDF2-HMAC-SHA256 with the fixed salt (see crypto.py)
3. **Connect via SSE** (`/api/v1/vitals/stream`) for real-time 1-second vital updates
4. **Fetch waveforms** on demand from `/api/v1/signals/ecg|ppg|rsp`
5. **Poll episode status** from `/api/v1/vitals/episode` to detect deteriorations
6. **Backfill history** on reconnect using `/api/v1/vitals/history`

### Recommended Polling Strategy

| Data | Endpoint | Frequency |
|------|----------|-----------|
| Live vitals | `/vitals/stream` (SSE) | Keep-alive connection |
| Episode status | `/vitals/episode` | Every 5 seconds |
| ECG waveform | `/signals/ecg?seconds=10` | Every 10 seconds |
| PPG waveform | `/signals/ppg?seconds=10` | Every 10 seconds |
| RSP waveform | `/signals/rsp?seconds=10` | Every 10 seconds |
| History backfill | `/vitals/history` | On connect/reconnect |

### Simulating for Testing

To make episodes fire more frequently (for testing your monitoring logic):

```env
# .env — fire episodes every ~10 minutes instead of ~2.5 hours
ARRHYTHMIA_MEAN_INTERVAL_S=600
```

To run at 2× real-time speed:
```env
SIM_SPEED=2.0
```

---

## Security Notes

- **TLS certificates** in `certs/` are **self-signed** (valid 10 years). Clients must either trust the cert or use `verify=False` (only for testing).
- The **AES shared secret** should be a random 32+ character string stored securely (e.g., in a secrets manager, not plain text in production).
- **Rotate the API key** before any deployment by changing `API_KEY` in `.env`.
- In production, put the simulator behind a proper reverse proxy (Nginx) with a CA-signed certificate.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

```bash
pip install -r requirements.txt
```

Key libraries: `neurokit2`, `fastapi`, `uvicorn`, `cryptography`, `sse-starlette`, `numpy`, `scipy`
