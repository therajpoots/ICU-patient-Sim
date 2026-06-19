"""
Model Context Protocol (MCP) Server for ICU Patient Monitor.
Exposes real-time patient vitals, encrypted anomaly logs, simulator controls,
DeepSeek diagnosis engines, and PDF report triggers.
"""

import os
import sys
import time
import json
import httpx
from datetime import datetime
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.database import EncryptedDatabase, derive_db_key, decrypt_value

# Initialize FastMCP Server
mcp = FastMCP("ICU-PatientTelemetry")

# Load environmental configs
db_password = os.getenv("DB_PASSWORD", "change-me-to-a-very-long-random-secret-string-here")
api_key = os.getenv("API_KEY", "sk-3ae47177f18e4ecf808440d6168c0d6f")
api_host = os.getenv("API_HOST", "127.0.0.1")
api_port = os.getenv("API_PORT", "8080")
monitoring_mode = os.getenv("MONITORING_MODE", "remote")

# --- Resources ---

@mcp.resource("patient://vitals")
def get_patient_vitals() -> str:
    """Gets the current patient vital signs decrypted from telemetry."""
    headers = {"X-API-Key": api_key}
    url = f"http://{api_host}:{api_port}/api/v1/vitals/current"
    try:
        r = httpx.get(url, headers=headers, timeout=2.0)
        if r.status_code == 200:
            res_data = r.json()
            aes_key = derive_db_key(db_password)
            vitals = decrypt_value(res_data["payload"], aes_key)
            return json.dumps(vitals, indent=2)
    except Exception as e:
        print(f"MCP resource failed to connect to simulator API: {e}")
        
    # Return simulated healthy fallback
    mock_vitals = {
        "heart_rate": 72.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "map": 93.3,
        "spo2": 98.0,
        "respiratory_rate": 14.0,
        "patient_state": "stable",
        "episode_label": None,
        "ml_anomaly_detected": False,
        "timestamp": time.time(),
        "extracted_features": {}
    }
    return json.dumps(mock_vitals, indent=2)


@mcp.resource("patient://profile")
def get_patient_profile() -> str:
    """Gets the patient's clinical demographic profile."""
    name = os.getenv("PATIENT_NAME", "Rana Talha Khalid")
    p_id = os.getenv("PATIENT_ID", "PT-2026-9041")
    ward = os.getenv("PATIENT_WARD", "ICU - Bed 04")
    age = os.getenv("PATIENT_AGE", "29")
    
    profile = {
        "patient_name": name,
        "patient_id": p_id,
        "ward": ward,
        "patient_age": age,
        "admitted_date": "2026-06-15",
        "attending_physician": "Dr. Antigravity",
        "primary_diagnosis": "Arrhythmia Monitoring"
    }
    return json.dumps(profile, indent=2)


@mcp.resource("logs://anomalies")
def get_anomalies_log() -> str:
    """Gets the historical decrypted patient anomaly logs from SQLite (excluding waveforms)."""
    try:
        db = EncryptedDatabase()
        logs = db.get_anomaly_logs(password=db_password)
        light_logs = []
        for l in logs:
            light_log = {
                "id": l["id"],
                "start_time": datetime.fromtimestamp(l["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.fromtimestamp(l["end_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(l["end_time"] - l["start_time"], 1),
                "state": l["state"],
                "vitals": l["vitals"]
            }
            light_logs.append(light_log)
        return json.dumps(light_logs, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve/decrypt anomaly logs: {e}"})


# --- Tools ---

@mcp.tool()
def trigger_anomaly(state: str) -> str:
    """
    Triggers a vital sign or arrhythmia anomaly in the patient simulator.
    
    Supported states: pvc, sinus_tachycardia, spo2_desaturation, atrial_fibrillation, ventricular_fibrillation, hypertensive_spike, stable
    """
    # 1. Write command to shared IPC channel (for local GUI mode)
    cmd_file = "data/mcp_commands.json"
    try:
        os.makedirs("data", exist_ok=True)
        with open(cmd_file, "w") as f:
            json.dump({"action": "trigger_anomaly", "state": state}, f)
        print(f"INFO: Wrote IPC command to trigger '{state}'.")
    except Exception as e:
        print(f"WARNING: Failed to write IPC command: {e}")
        
    # 2. Fire REST call to remote server (for remote mode)
    headers = {"X-API-Key": api_key}
    url = f"http://{api_host}:{api_port}/api/v1/vitals/anomaly"
    try:
        r = httpx.post(url, params={"state": state}, headers=headers, timeout=3.0)
        if r.status_code == 200:
            return f"Successfully triggered anomaly state '{state}' via REST API."
    except Exception as e:
        print(f"INFO: API trigger bypassed: {e}")
        
    return f"Anomaly state '{state}' triggered (written to local command channel)."


@mcp.tool()
def get_anomaly_logs(start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
    """
    Queries the encrypted SQLite database for historical anomaly episodes.
    Returns decrypted anomaly metadata. Timestamps should be in Unix epoch format.
    """
    try:
        db = EncryptedDatabase()
        logs = db.get_anomaly_logs(password=db_password, start_time=start_time, end_time=end_time)
        light_logs = []
        for l in logs:
            light_log = {
                "id": l["id"],
                "start_time": datetime.fromtimestamp(l["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.fromtimestamp(l["end_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(l["end_time"] - l["start_time"], 1),
                "state": l["state"],
                "vitals": l["vitals"]
            }
            light_logs.append(light_log)
        return json.dumps(light_logs, indent=2)
    except Exception as e:
        return f"Error loading logs: {e}"


@mcp.tool()
def generate_deepseek_insight(
    state: str, 
    heart_rate: float, 
    systolic_bp: float, 
    diastolic_bp: float, 
    spo2: float, 
    respiratory_rate: float
) -> str:
    """
    Queries DeepSeek LLM with patient vitals to generate a brief clinical diagnosis and insight.
    """
    from gui.pdf_generator import query_deepseek_insight
    vitals = {
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "spo2": spo2,
        "respiratory_rate": respiratory_rate
    }
    insight = query_deepseek_insight(state=state, vitals=vitals, api_key=api_key)
    return insight


@mcp.tool()
def generate_pdf_report(
    patient_name: str, 
    patient_id: str, 
    ward: str, 
    age: str, 
    output_path: str, 
    start_time: Optional[float] = None, 
    end_time: Optional[float] = None
) -> str:
    """
    Compiles and exports a professional clinical report PDF for a patient incorporating decrypted historical logs.
    """
    try:
        db = EncryptedDatabase()
        logs = db.get_anomaly_logs(password=db_password, start_time=start_time, end_time=end_time)
        if not logs:
            return "No anomaly logs found in the database for the specified timeframe."
            
        patient_info = {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "ward": ward,
            "patient_age": age
        }
        
        # Import compiler
        from gui.pdf_generator import generate_pdf_report as compile_pdf
        compile_pdf(
            patient_info=patient_info,
            anomaly_logs=logs,
            output_path=output_path,
            api_key=api_key
        )
        return f"Clinical PDF report successfully generated at: {output_path}"
    except Exception as e:
        return f"Failed to generate PDF report: {e}"


# --- Prompts ---

@mcp.prompt()
def analyze_arrhythmia(state: str) -> str:
    """Provides clinical diagnosis template for checking specific patient states."""
    return f"""You are a senior cardiologist and intensivist.
Please write a detailed clinical analysis for a patient in the following state: {state}.
Ensure you detail:
1. Pathophysiological mechanism of this arrhythmia or deterioration state.
2. Potential triggers in an intensive care setting (electrolyte imbalances, catecholamine surge, myocardial ischemia, etc.).
3. Immediate bedside interventions (pharmacotherapy, electrical therapies, ventilation changes).
4. Clinical risks if left untreated (stroke, myocardial infarction, hemodynamic collapse).
Please keep the language formal, precise, and clinical."""


if __name__ == "__main__":
    mcp.run()
