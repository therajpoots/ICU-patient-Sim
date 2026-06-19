# HealthFi: Explainable AI Bedside Telemetry Monitor & Patient Simulator

HealthFi is an end-to-end explainable AI clinical early warning system and bedside telemetry monitor. It simulates a **physiological twin** of an ICU patient (generating continuous ECG, PPG, and respiration waveforms), processes signals in real time to extract **39+ tabular features**, runs a pre-trained machine learning model to evaluate anomaly probability (early warning score), computes **SHAP explanations** to identify clinical drivers, and logs events securely in an AES-encrypted local database.

The system features a premium, glassmorphic dark-neon **PyQt6 Bedside Monitor Dashboard** for real-time visualization, threshold setups, emergency nurse calls, and PDF shift report compilation.

---

## System Architecture

```mermaid
graph TD
    subgraph Patient Simulator (BioSim Engine)
        A["arrhythmia_states.py: ArrhythmiaStateMachine"] -->|"Update State & Mods"| B["vitals.py: VitalsEngine"]
        B -->|"Compute Random Walks & State Deltas"| C["Vitals: SBP, DBP, MAP, SpO2, HR, RR, Core/Skin Temp"]
        B -->|"Generate Continuous Waveforms"| D["Waveforms: ECG, PPG, RSP"]
    end

    subgraph Feature Engineering Layer (features.py)
        C & D -->|"Window Analysis"| E["Extract 39+ Tabular Features"]
        E -->|"Output Feature Vector"| F["Tabular Feature Dataset"]
    end

    subgraph Anomaly Detection & Explainability AI (ml/)
        F -->|"Input Vector"| G["CatBoost / RF Classifier"]
        G -->|"Calculate Probability"| H["Anomaly Score (0-100)"]
        G -->|"Evaluate Decision Path"| I["Anomaly Severity (Normal/Mild/Mod/Severe)"]
        F & G -->|"Explain Predictions"| J["SHAP Tree Explainer"]
        J -->|"Identify Top Drivers"| K["Contributing Physiological Features"]
    end

    subgraph Clinical Surveillance Engine (gui/worker.py)
        H & I & K -->|"Real-time Telemetry"| L["PyQt6 Monitor Client Dashboard"]
        H & I & K -->|"Transition Event Log"| M["Encrypted Database: SQLite + AES-GCM"]
    end

    subgraph Report Generator (gui/pdf_generator.py)
        M -->|"Retrieve Logs & Waveforms"| N["Compile Shift Report PDF"]
        N -->|"Output clinical report"| O["Physician Case Report"]
    end
```

---

## Step-by-Step Installation Guide

Follow these steps to set up the program after downloading or cloning from GitHub:

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed on your system. You can check your version by running:
```bash
python --version
```

### 2. Download and Extract the Repository
If you downloaded the code as a ZIP file from GitHub, extract it to a folder of your choice (e.g., `E:\ICU pipeline`).

If you are cloning via Git:
```bash
git clone https://github.com/your-organization/icu-pipeline.git
cd icu-pipeline
```

### 3. Set Up a Virtual Environment (Recommended)
Creating a virtual environment ensures Python packages don't interfere with your global Python installation:

* **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
Install all required clinical libraries, GUI packages, and ML libraries:
```bash
pip install -r requirements.txt
```

### 5. Configure the Environment Settings (`.env`)
The simulator and encrypted database require a set of secrets and patient metrics. 

1. Duplicate the `.env.example` file and rename it to `.env`:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS / Linux
   cp .env.example .env
   ```
2. Open `.env` in any text editor and fill out the values:
   ```env
   # Shared secret used to derive the AES-256 key for payload encryption
   AES_SHARED_SECRET=your-32+character-random-key-here
   
   # API authentication token
   API_KEY=your-secure-telemetry-api-key
   
   # Default Patient profile loaded on start
   PATIENT_NAME=John Doe
   PATIENT_ID=PT-2026-9041
   PATIENT_AGE=65
   PATIENT_WARD=ICU Ward A
   
   # Model parameters
   ARRHYTHMIA_MEAN_INTERVAL_S=9000
   SIM_SPEED=1.0
   ```

---

## Step-by-Step Guide to Run the Program

HealthFi supports two primary topologies: **Local Simulation** (all-in-one PyQt app) and **Remote Surveillance** (FastAPI backend + separate PyQt clients).

### Option A: Running the All-in-One Application (easiest)
This launches the PyQt6 desktop monitor which directly simulates patient telemetry in the background.

1. **Launch the Desktop Client**:
   ```bash
   python run_desktop.py
   ```
2. **Configure the Connection Page (Connection Setup)**:
   - **Telemetry Source Mode**: Select `Simulate Patient (Local)` or `Demo Mode (10-Min / 6 Anomalies)`.
   - **Clinician Security Access Key**: Enter a password (e.g., `clinical123`). This password is used via **PBKDF2-HMAC-SHA256** to derive an AES key that secures/encrypts local patient telemetry databases.
   - **Enable Local MCP Server**: Keep checked to enable Model Context Protocol interactions.
   - Click **Start Clinical Monitoring**.
3. **Telemetry UI Controls**:
   - Switch between **📡 Live Monitor** (showing rolling ECG, PPG, RSP plots, vitals cards, and live anomaly logs) and **📊 Tabular Features** (showing 39+ features with baseline ranges, live drifts, and SHAP weight gauges).
   - **Setup Dialog (⚙️ Setup)**: Click this on the sidebar to adjust upper/lower clinical alarm thresholds and customize the patient name, ID, ward, unit, bed, and age.
   - **Emergency Call Button (⚠ Emergency Call)**: Click to trigger an immediate nurse call distress signal, log the incident into the local database, and flash the alarm banner for 15 seconds.
   - **Reports Dialog (📋 Reports)**: Generate a shift report. Select the time window and compile a PDF report containing 3-channel waveform charts, vitals progression, and XAI diagnosis explanations.

---

### Option B: Running in Client-Server REST Mode (Distributed)
In this mode, the patient twin simulator runs on a central server, exposing encrypted telemetry endpoints, while monitor clients query it over the network.

#### Step 1: Start the REST API Simulator Server
Launch the backend server using the FastAPI configuration:

* **Production (HTTPS - Auto-generates self-signed certificates)**:
  ```bash
  python run_simulator.py
  ```
* **Development (HTTP - local testing)**:
  ```bash
  python run_simulator.py --http
  ```
Verify the server is running by opening the FastAPI documentation in your browser:
- Swagger Docs: http://localhost:8080/docs (HTTP) or https://localhost:8443/docs (HTTPS)

#### Step 2: Launch the GUI Monitor Client
1. Launch the desktop GUI on any terminal in the network:
   ```bash
   python run_desktop.py
   ```
2. On the Connection Page:
   - **Telemetry Source Mode**: Select `Remote Patient Monitor (REST SSE)`.
   - **Remote Host IP & Port**: Enter the IP address and port (e.g., `127.0.0.1` and `8080`).
   - **DeepSeek API Key**: Provide an API key for clinical summary compilation.
   - **Clinician Security Access Key**: Enter the database encryption password.
   - Click **Test Telemetry Stream** to verify connectivity, then click **Start Clinical Monitoring**.

---

## Machine Learning & AI Explainability Pipeline

To retrain the clinical surveillance models or test signal feature analysis offline:

1. **Train Model Cohort**:
   ```bash
   python ml/train_model.py
   ```
   This simulates physiological twin cohorts, extracts tabular features, trains five classifiers (Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression), plots comparative metrics (ROC-AUC curves, Feature Importance, Decision Paths), and saves the model objects (`trained_gbm.pkl` and `shap_explainer.pkl`).
2. **Verify Anomaly Detection**:
   ```bash
   python -u ml/verify_anomaly.py
   ```
   Validates model predictions and prints the top 3 physiological drivers identified by SHAP tree explainers.

---

## Project Structure

```
├── run_desktop.py            # PyQt6 Bedside Telemetry Monitor Entry point
├── run_simulator.py          # FastAPI Backend Twin Simulator Entry point
├── requirements.txt          # Python dependency list
├── .env.example              # Configuration template
│
├── gui/                      # Client UI Layer
│   ├── app.py                # Main window layout and route coordination
│   ├── connection_page.py    # Glassmorphic setup login dialog
│   ├── monitor_page.py       # Live telemetry, Setup thresholds config, emergency calling
│   ├── features_page.py      # Tabular features, drift status, SHAP influence bars
│   ├── report_dialog.py      # PDF Report compiler setup modal
│   ├── pdf_generator.py      # Generates PDF report with charts & XAI summaries
│   ├── worker.py             # Background thread polling REST SSE & running ML
│   └── database.py           # Local encrypted SQLite event logger
│
├── simulator/                # Patient Physiological Twin Engine
│   ├── engine.py             # Main loop coordinating waveform buffers
│   ├── vitals.py             # Pre-filtered ECG, PPG, RSP waveform synthesis
│   ├── arrhythmia_states.py  # 15-state physiological twin state machine
│   └── patient_profile.py    # Default baselines and environments
│
├── ml/                       # Machine Learning & AI Explainability
│   ├── train_model.py        # Model cohort trainer & performance comparison
│   ├── explainability.py     # SHAP tree explanation utilities
│   └── verify_anomaly.py     # Offline ML performance evaluation
│
└── client_example/           # Dev telemetry utilities
    ├── decrypt_client.py     # AES-256-GCM decrypting REST client helper
    └── test_system.py        # End-to-end integration test runner
```

---

## Troubleshooting

- **Database Decryption Failure**: If you get warnings like `Skipping log ID due to decryption failure`, it means the database was written using a different Clinician Security Access Key password. Use the password that was active when those logs were created, or delete `data/anomaly_logs.db` to start a fresh archive.
- **TLS Certificate Warnings**: In production/HTTPS mode, the server auto-generates self-signed certificates. If your custom clients fail to connect, ensure you disable SSL verification (e.g., `verify_ssl=False` in REST calls or python test scripts).
- **PDF Export Freeze**: Ensure you have a valid DeepSeek API key configured in settings, or use the offline fallback mode by keeping the default `sk-...` dummy key in settings.
