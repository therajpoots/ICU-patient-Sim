"""
PyQt6 Background Worker Thread
Handles real-time data collection from local or remote simulator,
QRS peak detection, feature extraction, real-time ML anomaly classification,
demo countdown scheduler, and database logging on anomaly transitions.
"""

import os
import time
import json
import httpx
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from PyQt6.QtCore import QThread, pyqtSignal

# Set up logging for error capture
os.makedirs("data", exist_ok=True)
logging.basicConfig(
    filename="data/app_error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("MonitoringWorker")

from gui.database import EncryptedDatabase, derive_db_key, decrypt_value
from simulator.preprocessing import clean_ecg, clean_ppg, clean_rsp, pan_tompkins_qrs_detector
from simulator.features import extract_features_from_window
from simulator.engine import SimulatorEngine, SAMPLING_RATE

def classify_patient_state(vitals: Dict[str, Any], features: Dict[str, Any]) -> str:
    """
    Classifies the patient state name from physiological vitals and features,
    without leaking the ground-truth injected patient_state.
    """
    hr = vitals.get("heart_rate", 72.0)
    sbp = vitals.get("systolic_bp", 120.0)
    dbp = vitals.get("diastolic_bp", 80.0)
    spo2 = vitals.get("spo2", 98.0)
    rr = vitals.get("respiratory_rate", 16.0)
    temp_core = vitals.get("core_temperature", 37.0)
    temp_skin = vitals.get("skin_temperature", 35.5)
    
    st_elevation = features.get("ecg_st_elevation", 0.0)
    qrs_duration = features.get("ecg_qrs_duration", 0.08)
    sdnn = features.get("hrv_sdnn", 0.03)
    rmssd = features.get("hrv_rmssd", 0.03)

    # 1. Ventricular Fibrillation (vfib): extreme HR + extreme drop in BP
    if hr > 165 and sbp < 50:
        return "ventricular_fibrillation"
        
    # 2. Ventricular Tachycardia (vtach): extreme HR (> 135) + low BP
    if hr > 135 and sbp < 95:
        return "ventricular_tachycardia"
        
    # 3. STEMI Heart Attack: high ST elevation
    if st_elevation > 0.15:
        return "stemi"
        
    # 4. Myocardial Ischemia: low ST depression
    if st_elevation < -0.10:
        return "myocardial_ischemia"
        
    # 5. Septic Shock: fever + tachycardia + hypotension
    if temp_core > 38.5 and hr > 95 and sbp < 105:
        return "sepsis"
        
    # 6. Respiratory Distress: very high breathing rate + low SpO2
    if rr > 24 and spo2 < 95:
        return "respiratory_distress"
        
    # 7. SpO2 Desaturation: low SpO2
    if spo2 < 93:
        return "spo2_desaturation"
        
    # 8. Hypertensive Crisis: very high BP
    if sbp > 145 or dbp > 95:
        return "hypertensive_spike"
        
    # 9. Hypotensive Crisis: very low BP
    if sbp < 95 or dbp < 60:
        return "hypotension"
        
    # 10. Acute Heart Failure: low BP + low SpO2
    if sbp < 110 and spo2 < 95:
        return "heart_failure"
        
    # 11. Atrial Fibrillation: highly irregular rhythm (high HRV SDNN/RMSSD and high HR)
    if sdnn > 0.05 and rmssd > 0.05 and hr > 95:
        return "atrial_fibrillation"
        
    # 12. Atrial Flutter: moderately irregular rhythm + high HR
    if sdnn > 0.04 and rmssd > 0.04 and hr > 90:
        return "atrial_flutter"
        
    # 13. Premature Ventricular Contractions (PVC): high HRV SDNN/RMSSD + normal/slightly depressed HR
    if sdnn > 0.06 and rmssd > 0.06:
        return "pvc"
        
    # 14. Premature Atrial Contractions (PAC): moderate HRV SDNN/RMSSD
    if sdnn > 0.04 and rmssd > 0.04:
        return "pac"
        
    # 15. Sinus Tachycardia: high HR, regular rhythm
    if hr > 100:
        return "sinus_tachycardia"
        
    # 16. Sinus Bradycardia: low HR, regular rhythm
    if hr < 55:
        return "sinus_bradycardia"
        
    return "stable"

def get_rule_based_shap(vitals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule-based fallback for SHAP contributors based on PatientState."""
    state = vitals.get("patient_state", "stable")
    if state == "stable":
        return []
    
    mapping = {
        "sepsis": [("temp_core", 2.2), ("temp_skin", -1.5), ("hr_mean", 25.0)],
        "stemi": [("ecg_st_elevation", 0.25), ("temp_core", 0.6), ("hr_mean", 15.0)],
        "myocardial_ischemia": [("ecg_st_elevation", -0.15), ("dbp_mean", -10.0), ("hr_mean", 10.0)],
        "heart_failure": [("ppg_pulse_amplitude", -0.4), ("sbp_mean", -15.0), ("spo2_mean", -5.0)],
        "respiratory_distress": [("rsp_rate", 12.0), ("spo2_mean", -6.0), ("rsp_ie_ratio", -0.3)],
        "sinus_tachycardia": [("hr_mean", 40.0), ("ecg_hr", 40.0), ("hrv_sdnn", -0.02)],
        "sinus_bradycardia": [("hr_mean", -25.0), ("ecg_hr", -25.0), ("hrv_sdnn", -0.01)],
        "atrial_fibrillation": [("hr_mean", 20.0), ("hr_var", 8.0), ("ppg_pulse_variability", 0.15)],
        "atrial_flutter": [("hr_mean", 15.0), ("hr_var", 5.0), ("ecg_hr", 15.0)],
        "spo2_desaturation": [("spo2_mean", -8.0), ("spo2_desat_events", 1.0), ("rsp_rate", 4.0)],
        "hypertensive_spike": [("sbp_mean", 40.0), ("dbp_mean", 25.0), ("bp_pulse_pressure", 15.0)],
        "hypotension": [("sbp_mean", -30.0), ("dbp_mean", -15.0), ("bp_map", -20.0)],
        "ventricular_tachycardia": [("hr_mean", 80.0), ("ecg_hr", 80.0), ("ppg_pulse_amplitude", -0.5)],
        "ventricular_fibrillation": [("hr_mean", 110.0), ("ppg_pulse_amplitude", -0.8), ("bp_map", -50.0)],
        "pvc": [("hrv_sdnn", 0.08), ("hrv_rmssd", 0.09), ("hr_mean", 5.0)],
        "pac": [("hrv_sdnn", 0.05), ("hrv_rmssd", 0.06), ("hr_mean", 3.0)]
    }
    
    contribs = mapping.get(state, [("hr_mean", 1.0), ("sbp_mean", 1.0), ("spo2_mean", 1.0)])
    return [{"feature": name, "influence": val} for name, val in contribs]


class MonitoringWorker(QThread):
    # Signals
    vitals_updated = pyqtSignal(dict) # latest vitals snapshot + current features
    waveforms_updated = pyqtSignal(list, list, list) # ecg, ppg, rsp lists for plotting
    anomaly_logged = pyqtSignal(int, str, float) # log_id, state, start_time
    demo_timer_updated = pyqtSignal(int) # remaining demo seconds
    error_occurred = pyqtSignal(str) # connection / decryption errors
    finished_monitoring = pyqtSignal() # loop terminated

    def __init__(self, mode: str, host: str = "127.0.0.1", port: int = 8080, api_key: str = "", password: str = ""):
        super().__init__()
        self.mode = mode # 'remote' | 'local' | 'demo'
        self.host = host
        self.port = port
        self.api_key = api_key
        self.password = password
        self.aes_key = derive_db_key(password) if password else None
        
        self.running = False
        self.db = EncryptedDatabase()
        
        # Load ML models if available
        self.gbm = None
        self.minmax = None
        self.zscore = None
        self.explainer = None
        self._load_ml_models()
        
        # Anomaly tracking state
        self.active_anomaly = False
        self.anomaly_start_time = None
        self.anomaly_state = None
        self.anomaly_vitals = []
        self.anomaly_ecg = []
        self.anomaly_ppg = []
        self.anomaly_rsp = []

    def _load_ml_models(self):
        """Loads the pre-trained Random Forest model and scaling pipelines, plus the SHAP explainer."""
        try:
            with open("trained_gbm.pkl", "rb") as f:
                self.gbm = pickle.load(f)
            with open("minmax_scaler.pkl", "rb") as f:
                self.minmax = pickle.load(f)
            with open("zscore_scaler.pkl", "rb") as f:
                self.zscore = pickle.load(f)
            print("Worker loaded Random Forest ML model and scalers successfully.")
        except Exception as e:
            print(f"Warning: Primary ML model files not found ({e}). Falling back to rule-based classification.")
            
        try:
            with open("shap_explainer.pkl", "rb") as f:
                self.explainer = pickle.load(f)
            print("Worker loaded SHAP explainer successfully.")
        except Exception as e:
            print(f"Warning: SHAP explainer not loaded ({e}). Will use rule-based fallback explanations.")

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        
        # 1. Initialize data source
        self.local_engine = None
        if self.mode in ('local', 'demo'):
            # Start local simulator
            self.local_engine = SimulatorEngine()
            self.local_engine.start()
            print("Local simulation engine started.")
            
        demo_elapsed = 0
        demo_duration = 600 # 10 minutes
        
        # Scheduled demo anomalies: (timestamp_seconds, state_name)
        demo_schedule = [
            (30, "pvc"),
            (120, "sinus_tachycardia"),
            (240, "spo2_desaturation"),
            (360, "atrial_fibrillation"),
            (480, "ventricular_fibrillation"),
            (540, "hypertensive_spike"),
            (590, "stable") # restore to stable before completion
        ]
        
        headers = {"X-API-Key": self.api_key}
        
        # Slide buffers for remote mode (to smooth charting)
        ecg_history = []
        ppg_history = []
        rsp_history = []
        
        # Secondary vital histories (for feature trend calculation)
        sbp_3s = []
        dbp_3s = []
        spo2_3s = []
        hr_3s = []
        temp_core_3s = []
        temp_skin_3s = []
        
        tick_count = 0
        
        while self.running:
            start_tick_time = time.time()
            tick_count += 1
            
            # --- Check for IPC commands from MCP Server ---
            cmd_file = "data/mcp_commands.json"
            if os.path.exists(cmd_file):
                try:
                    with open(cmd_file, "r") as f:
                        cmd_data = json.load(f)
                    os.remove(cmd_file)
                    if cmd_data.get("action") == "trigger_anomaly":
                        state = cmd_data.get("state")
                        print(f"IPC Trigger Command Received: {state}")
                        self.trigger_anomaly(state)
                except Exception as e:
                    print(f"Error handling IPC command: {e}")

            try:
                vitals = {}
                ecg_seg = []
                ppg_seg = []
                rsp_seg = []
                
                # --- A. Fetch Data ---
                if self.mode in ('local', 'demo'):
                    # Handle Demo mode scheduling
                    if self.mode == 'demo':
                        demo_elapsed += 1
                        self.demo_timer_updated.emit(max(0, demo_duration - demo_elapsed))
                        
                        # Trigger scheduled anomalies
                        for trigger_time, state_name in demo_schedule:
                            if demo_elapsed == trigger_time:
                                print(f"Demo Mode: Triggering scheduled anomaly '{state_name}' at {demo_elapsed}s.")
                                self.local_engine.trigger_anomaly(state_name)
                                
                        if demo_elapsed >= demo_duration:
                            self.running = False
                            print("Demo Mode completed.")
                            break
                            
                    # Get local vitals & waveforms (3s segments)
                    vitals = self.local_engine.get_latest_vitals()
                    ecg_seg = self.local_engine.get_ecg_waveform(3)
                    ppg_seg = self.local_engine.get_ppg_waveform(3)
                    rsp_seg = self.local_engine.get_rsp_waveform(3)
                    
                else: # 'remote' mode
                    # 1. Fetch decrypted vitals SSE equivalent via REST polling
                    v_url = f"http://{self.host}:{self.port}/api/v1/vitals/current"
                    r = httpx.get(v_url, headers=headers, timeout=3.0)
                    if r.status_code != 200:
                        raise ConnectionError(f"API returned status {r.status_code}")
                    v_data = r.json()
                    vitals = decrypt_value(v_data["payload"], self.aes_key)
                    
                    # 2. Fetch decrypted waveforms (last 3 seconds)
                    ecg_url = f"http://{self.host}:{self.port}/api/v1/signals/ecg?seconds=3"
                    ppg_url = f"http://{self.host}:{self.port}/api/v1/signals/ppg?seconds=3"
                    rsp_url = f"http://{self.host}:{self.port}/api/v1/signals/rsp?seconds=3"
                    
                    e_data = httpx.get(ecg_url, headers=headers, timeout=3.0).json()
                    p_data = httpx.get(ppg_url, headers=headers, timeout=3.0).json()
                    r_data = httpx.get(rsp_url, headers=headers, timeout=3.0).json()
                    
                    ecg_seg = decrypt_value(e_data["payload"], self.aes_key)["samples"]
                    ppg_seg = decrypt_value(p_data["payload"], self.aes_key)["samples"]
                    rsp_seg = decrypt_value(r_data["payload"], self.aes_key)["samples"]
                
                # --- B. Preprocessing & Feature Extraction ---
                # Ensure we have a complete 3-second buffer (750 samples) before running DSP & ML
                if len(ecg_seg) < 750 or len(ppg_seg) < 750 or len(rsp_seg) < 750:
                    self.vitals_updated.emit(vitals)
                    self.waveforms_updated.emit(
                        list(ecg_seg[-250:]),
                        list(ppg_seg[-250:]),
                        list(rsp_seg[-250:])
                    )
                    # Sleep to match 1.0 second simulation tick before continuing to prevent CPU spin
                    elapsed = time.time() - start_tick_time
                    sleep_time = max(0.01, 1.0 - elapsed)
                    time.sleep(sleep_time)
                    continue

                # Accumulate history for trend slope/variance (last 3 seconds)
                # Use .get() with safe defaults to prevent KeyError if vitals dict is incomplete
                sbp_3s.append(vitals.get("systolic_bp", 120.0))
                dbp_3s.append(vitals.get("diastolic_bp", 80.0))
                spo2_3s.append(vitals.get("spo2", 98.0))
                hr_3s.append(vitals.get("heart_rate", 72.0))
                temp_core_3s.append(vitals.get("core_temperature", 37.0))
                temp_skin_3s.append(vitals.get("skin_temperature", 35.5))
                
                if len(sbp_3s) > 3:
                    sbp_3s.pop(0)
                    dbp_3s.pop(0)
                    spo2_3s.pop(0)
                    hr_3s.pop(0)
                    temp_core_3s.pop(0)
                    temp_skin_3s.pop(0)
                
                # Extract features from current 3-second segments
                ecg_raw = np.array(ecg_seg[-750:])
                ppg_raw = np.array(ppg_seg[-750:])
                rsp_raw = np.array(rsp_seg[-750:])
                ecg_clean = clean_ecg(ecg_raw, SAMPLING_RATE)
                ppg_clean = clean_ppg(ppg_raw, SAMPLING_RATE)
                rsp_clean = clean_rsp(rsp_raw, SAMPLING_RATE)
                
                features = extract_features_from_window(
                    ecg_raw, ppg_raw, rsp_raw,
                    sbp_3s, dbp_3s, spo2_3s, hr_3s,
                    SAMPLING_RATE,
                    temp_core_3s, temp_skin_3s
                )
                
                # --- C. Real-Time Anomaly Inference & SHAP ---
                predicted_state = classify_patient_state(vitals, features)
                
                anomaly_score = 5.0
                is_anomaly = False
                
                if self.gbm is not None and self.minmax is not None and self.zscore is not None:
                    try:
                        feat_vector = np.array([list(features.values())])
                        scaled_vector = self.zscore.transform(self.minmax.transform(feat_vector))
                        
                        # Anomaly Score is probability of anomaly class (Abnormal) scaled to 0-100
                        prob = self.gbm.predict_proba(scaled_vector)[0, 1]
                        anomaly_score = float(prob * 100.0)
                        
                        if predicted_state == "stable":
                            is_anomaly = False
                            anomaly_score = min(anomaly_score, 44.9) # override stable to normal/stable
                        else:
                            is_anomaly = bool(anomaly_score >= 45.0)
                    except Exception as e:
                        print(f"ML Prediction Error: {e}")
                        is_anomaly = bool(predicted_state != "stable")
                        anomaly_score = 90.0 if is_anomaly else 5.0
                else:
                    is_anomaly = bool(predicted_state != "stable")
                    anomaly_score = 90.0 if is_anomaly else 5.0
                
                # Classify Severity Level based on Alarm Threshold Bands
                if anomaly_score < 20.0:
                    severity = "Normal"
                elif anomaly_score < 50.0:
                    severity = "Mild"
                elif anomaly_score < 80.0:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                    
                vitals["ml_anomaly_detected"] = is_anomaly
                vitals["anomaly_score"] = round(anomaly_score, 1)
                vitals["anomaly_severity"] = severity
                vitals["extracted_features"] = features
                
                # Overwrite patient_state to prevent ground-truth state leakage in UI and logs
                vitals["patient_state"] = predicted_state
                
                # Run SHAP TreeExplainer on the active anomaly
                if is_anomaly:
                    shap_contribs = []
                    if self.explainer is not None and self.minmax is not None and self.zscore is not None:
                        try:
                            # Evaluate SHAP values
                            shap_vals = self.explainer.shap_values(scaled_vector)
                            
                            # Handle different SHAP output formats (multiclass / binary list)
                            if isinstance(shap_vals, list):
                                class_shap = shap_vals[1][0]
                            elif len(shap_vals.shape) == 3:
                                class_shap = shap_vals[0, :, 1]
                            elif len(shap_vals.shape) == 2:
                                if shap_vals.shape[0] == 1:
                                    class_shap = shap_vals[0]
                                else:
                                    class_shap = shap_vals[:, 1]
                            else:
                                class_shap = shap_vals[0]
                                
                            feat_names = list(features.keys())
                            shap_pairs = [(feat_names[i], float(class_shap[i])) for i in range(len(feat_names))]
                            # Sort by absolute SHAP value descending
                            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Filter to positive contributors (or just top 3 absolute influence)
                            top_shap = shap_pairs[:3]
                            shap_contribs = [{"feature": f, "influence": val} for f, val in top_shap]
                        except Exception as e:
                            print(f"SHAP explainer error: {e}. Falling back to rule-based explanations.")
                            shap_contribs = get_rule_based_shap(vitals)
                    else:
                        shap_contribs = get_rule_based_shap(vitals)
                    
                    vitals["shap_contributors"] = shap_contribs
                else:
                    vitals["shap_contributors"] = []
                
                # --- D. Encrypted Database Logging on Anomaly Transitions ---
                current_state = vitals.get("patient_state", "stable")
                
                if is_anomaly and current_state != "stable":
                    if not self.active_anomaly:
                        # Anomaly Event Starts!
                        self.active_anomaly = True
                        self.anomaly_start_time = time.time()
                        self.anomaly_state = current_state
                        self.anomaly_vitals = []
                        self.anomaly_ecg = []
                        self.anomaly_ppg = []
                        self.anomaly_rsp = []
                        print(f"Anomaly logging started: {current_state}")
                    
                    # Log vital snapshots and waveform frames during the active anomaly
                    self.anomaly_vitals.append(vitals)
                    # Add current second's waveforms (last 250 samples)
                    self.anomaly_ecg.extend(ecg_seg[-250:])
                    self.anomaly_ppg.extend(ppg_seg[-250:])
                    self.anomaly_rsp.extend(rsp_seg[-250:])
                    
                else:
                    # If patient returns to stable, close and save active log
                    if self.active_anomaly:
                        self.active_anomaly = False
                        end_time = time.time()
                        
                        # Find the vitals snapshot representing the peak
                        mid_idx = len(self.anomaly_vitals) // 2
                        peak_vitals = self.anomaly_vitals[mid_idx] if len(self.anomaly_vitals) > 0 else vitals
                        
                        # Waveform dictionaries
                        waveforms_log = {
                            "ecg": self.anomaly_ecg[-2500:], # clamp to last 10 seconds max for DB size
                            "ppg": self.anomaly_ppg[-2500:],
                            "rsp": self.anomaly_rsp[-2500:]
                        }
                        
                        try:
                            log_id = self.db.log_anomaly(
                                password=self.password,
                                start_time=self.anomaly_start_time,
                                end_time=end_time,
                                state=self.anomaly_state,
                                vitals=peak_vitals,
                                waveforms=waveforms_log
                            )
                            self.anomaly_logged.emit(log_id, self.anomaly_state, self.anomaly_start_time)
                            print(f"Anomaly logged successfully in encrypted database: Log ID {log_id}")
                        except Exception as e:
                            print(f"Failed to save anomaly to DB: {e}")
                
                # --- E. Emit Update Signals for UI ---
                self.vitals_updated.emit(vitals)
                self.waveforms_updated.emit(
                    list(ecg_seg[-250:]),
                    list(ppg_seg[-250:]),
                    list(rsp_seg[-250:])
                )
                
            except Exception as e:
                logger.exception("Exception inside monitoring worker tick loop:")
                self.error_occurred.emit(str(e))
                time.sleep(1.0)
                
            # Sleep to match 1.0 second simulation tick
            elapsed = time.time() - start_tick_time
            sleep_time = max(0.01, 1.0 - elapsed)
            time.sleep(sleep_time)

        # Stop simulator if running locally
        if self.local_engine:
            self.local_engine.stop()
            print("Local simulation engine stopped.")
            
        self.finished_monitoring.emit()

    def trigger_anomaly(self, state: str):
        """Allows triggering anomalies dynamically from the UI or MCP."""
        if self.mode in ('local', 'demo') and self.local_engine:
            self.local_engine.trigger_anomaly(state)
        elif self.mode == 'remote':
            try:
                headers = {"X-API-Key": self.api_key}
                url = f"http://{self.host}:{self.port}/api/v1/vitals/anomaly"
                httpx.post(url, params={"state": state}, headers=headers, timeout=3.0)
                print(f"Triggered anomaly {state} remotely.")
            except Exception as e:
                print(f"Failed to trigger remote anomaly: {e}")
