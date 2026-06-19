"""
ML Model Training Pipeline - HealthFi Telemetry Anomaly Detection.
Generates balanced physiological dataset (1:1 Normal to Abnormal ratio),
runs the actual DSP and features extraction pipeline, and trains/evaluates
Random Forest, XGBoost, LightGBM, CatBoost, and Logistic Regression.
Displays live progress bars.
"""

import os
import sys
import time
import pickle
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import shap

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.preprocessing import clean_ecg, clean_ppg, clean_rsp
from simulator.features import extract_features_from_window
from simulator.arrhythmia_states import PatientState
from simulator.patient_profile import PatientProfile
from simulator.vitals import (
    VitalsEngine, ArrhythmiaStateMachine, 
    generate_ecg_segment, generate_ppg_segment, generate_rsp_segment
)

# Set up logging
os.makedirs("data", exist_ok=True)
logging.basicConfig(
    filename="data/app_error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("TrainModel")

# Constants
SAMPLING_RATE = 250
WINDOW_DURATION_S = 3.0
WINDOW_SAMPLES = int(WINDOW_DURATION_S * SAMPLING_RATE) # 750 samples

STATES_TO_SIMULATE = [
    PatientState.STABLE,
    PatientState.PVC,
    PatientState.PAC,
    PatientState.TACHYCARDIA,
    PatientState.BRADYCARDIA,
    PatientState.AFIB,
    PatientState.AFLUTTER,
    PatientState.DESATURATION,
    PatientState.HYPERTENSIVE,
    PatientState.HYPOTENSION,
    PatientState.RESP_DISTRESS,
    PatientState.VTACH,
    PatientState.ISCHEMIA,
    PatientState.STEMI,
    PatientState.HEART_FAILURE,
    PatientState.SEPSIS,
    PatientState.VFIB
]

def draw_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", bar_length: int = 30):
    """Prints a standard console-based text progress bar."""
    percent = float(current) / total
    arrow = '-' * int(percent * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f"\r{prefix} [{arrow}{spaces}] {percent*100:.1f}% {suffix}")
    sys.stdout.flush()

def generate_data_for_state(state: PatientState, duration_seconds: int, progress_counter=None, counter_lock=None) -> Tuple[List[Dict[str, float]], List[int]]:
    """Simulates a patient in a state and extracts features with progress reporting."""
    profile = PatientProfile()
    sm = ArrhythmiaStateMachine(mean_interval_s=999999) # prevent random swaps
    ve = VitalsEngine(profile, sm)
    
    if state != PatientState.STABLE:
        sm.trigger_anomaly(state.value)
        
    features_list = []
    labels = []
    
    from collections import deque
    ecg_buf = deque(maxlen=750)
    ppg_buf = deque(maxlen=750)
    rsp_buf = deque(maxlen=750)
    
    sbp_buf = deque(maxlen=3)
    dbp_buf = deque(maxlen=3)
    spo2_buf = deque(maxlen=3)
    hr_buf = deque(maxlen=3)
    temp_core_buf = deque(maxlen=3)
    temp_skin_buf = deque(maxlen=3)
    
    for t in range(duration_seconds):
        vitals = ve.update(1.0)
        
        hr = vitals["heart_rate"]
        rr = vitals["respiratory_rate"]
        pvc_burden = vitals["pvc_burden"]
        pac_burden = vitals["pac_burden"]
        irregularity = vitals["irregularity"]
        st_delta = vitals["st_delta"]
        ppg_amplitude_factor = vitals["ppg_amplitude_factor"]
        ie_ratio_factor = vitals["ie_ratio_factor"]
        current_state = PatientState(vitals["patient_state"])
        
        # Generate 1 second of waveforms
        ecg_sec = generate_ecg_segment(
            hr=hr,
            duration_s=1.0,
            sr=SAMPLING_RATE,
            profile=profile,
            state=current_state,
            pvc_burden=pvc_burden,
            pac_burden=pac_burden,
            irregularity=irregularity,
            st_delta=st_delta,
            t_start=float(t)
        )
        ppg_sec = generate_ppg_segment(
            hr=hr,
            duration_s=1.0,
            sr=SAMPLING_RATE,
            profile=profile,
            rr=rr,
            state=current_state,
            ppg_amplitude_factor=ppg_amplitude_factor,
            t_start=float(t)
        )
        rsp_sec = generate_rsp_segment(
            rr=rr,
            duration_s=1.0,
            sr=SAMPLING_RATE,
            profile=profile,
            ie_ratio=ie_ratio_factor,
            t_start=float(t)
        )
        
        ecg_buf.extend(ecg_sec.tolist())
        ppg_buf.extend(ppg_sec.tolist())
        rsp_buf.extend(rsp_sec.tolist())
        
        sbp_buf.append(vitals["systolic_bp"])
        dbp_buf.append(vitals["diastolic_bp"])
        spo2_buf.append(vitals["spo2"])
        hr_buf.append(vitals["heart_rate"])
        temp_core_buf.append(vitals["core_temperature"])
        temp_skin_buf.append(vitals["skin_temperature"])
        
        if len(ecg_buf) >= 750:
            feat = extract_features_from_window(
                np.array(ecg_buf),
                np.array(ppg_buf),
                np.array(rsp_buf),
                list(sbp_buf),
                list(dbp_buf),
                list(spo2_buf),
                list(hr_buf),
                SAMPLING_RATE,
                list(temp_core_buf),
                list(temp_skin_buf)
            )
            features_list.append(feat)
            labels.append(0 if state == PatientState.STABLE else 1)
            
        # Draw progress bar or update shared counter
        if progress_counter is not None and counter_lock is not None:
            with counter_lock:
                progress_counter.value += 1
        else:
            if (t + 1) % 10 == 0 or t == duration_seconds - 1:
                draw_progress_bar(t + 1, duration_seconds, prefix=f"Generating {state.value:<18}", suffix=f"({t+1}/{duration_seconds}s)")
            
    return features_list, labels


def add_shadow_and_border(input_path: str, output_path: str):
    """Applies drop-shadow styling wrapper using Pillow."""
    from PIL import Image as PILImage, ImageFilter
    img = PILImage.open(input_path)
    w, h = img.size
    pad = 40
    canvas = PILImage.new("RGBA", (w + pad * 2, h + pad * 2), (255, 255, 255, 255))
    shadow = PILImage.new("RGBA", (w, h), (0, 0, 0, 45))
    radius = 16
    from PIL import ImageDraw
    mask = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    canvas.paste(shadow, (pad + 15, pad + 15), mask)
    canvas = canvas.filter(ImageFilter.GaussianBlur(12))
    rounded_img = PILImage.new("RGBA", (w, h))
    rounded_img.paste(img, (0, 0), mask)
    canvas.paste(rounded_img, (pad, pad), mask)
    canvas.convert("RGB").save(output_path, "PNG")

def compute_metrics(y_true, y_pred, y_prob):
    """Calculate core classifier evaluation statistics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

def run_ml_pipeline(duration_per_state: int = 600, output_dir: str = "."):
    print("==================================================================")
    print("          HEALTHFI MACHINE LEARNING TRAINING CENTER               ")
    print("==================================================================")
    print("  Generating balanced patient telemetry data (1:1 Normal:Abnormal)")
    print("  Running actual features extraction and training RF, XGB, LGB, Cat, LR...")
    print("==================================================================")
    
    # 1. Compile Dataset (balance normal stable state with anomaly states)
    anomaly_count = len(STATES_TO_SIMULATE) - 1
    state_durations = {
        state: duration_per_state * anomaly_count if state == PatientState.STABLE else duration_per_state
        for state in STATES_TO_SIMULATE
    }
    
    all_features = []
    all_labels = []
    
    total_seconds = sum(state_durations.values())
    print(f"Starting parallel generation of {len(STATES_TO_SIMULATE)} states ({total_seconds} seconds total)...")
    
    manager = multiprocessing.Manager()
    progress_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()
    
    max_workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generate_data_for_state, 
                state, 
                state_durations[state], 
                progress_counter, 
                counter_lock
            ): state for state in STATES_TO_SIMULATE
        }
        
        while any(not f.done() for f in futures):
            val = progress_counter.value
            draw_progress_bar(val, total_seconds, prefix="Generating Telemetry Data", suffix=f"({val}/{total_seconds}s)")
            time.sleep(0.5)
            
        draw_progress_bar(total_seconds, total_seconds, prefix="Generating Telemetry Data", suffix=f"({total_seconds}/{total_seconds}s)")
        print()
        
        # Gather results
        for future in futures:
            state = futures[future]
            try:
                feats, labels = future.result()
                all_features.extend(feats)
                all_labels.extend(labels)
            except Exception as e:
                print(f"\n[ERROR] State {state.value} simulation failed: {e}")
                logger.exception(f"Error generating data for state {state.value}:")
        
    df_features = pd.DataFrame(all_features)
    labels = np.array(all_labels)
    
    print(f"[INFO] Dataset generation complete: {df_features.shape[0]} samples, {df_features.shape[1]} features.")
    
    # 2. Chronological Split (80% Train, 20% Test per state)
    train_idx = []
    test_idx = []
    
    current_idx = 0
    for state in STATES_TO_SIMULATE:
        n_samples = state_durations[state] - 2
        start = current_idx
        end = start + n_samples
        split_point = int(start + 0.8 * n_samples)
        
        train_idx.extend(range(start, split_point))
        test_idx.extend(range(split_point, end))
        current_idx = end
        
    X_train_raw = df_features.iloc[train_idx].values
    y_train = labels[train_idx]
    X_test_raw = df_features.iloc[test_idx].values
    y_test = labels[test_idx]
    
    # 3. Fit Normalization pipelines
    minmax = MinMaxScaler()
    X_train_mm = minmax.fit_transform(X_train_raw)
    X_test_mm = minmax.transform(X_test_raw)
    
    zscore = StandardScaler()
    X_train = zscore.fit_transform(X_train_mm)
    X_test = zscore.transform(X_test_mm)
    
    # 4. Train 5 Classifiers with balanced class weights
    print("\n[INFO] Training Classifiers...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, class_weight='balanced', verbose=-1),
        "CatBoost": CatBoostClassifier(n_estimators=100, depth=4, learning_rate=0.1, random_state=42, auto_class_weights='Balanced', verbose=0),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
        logger.info(f"{name} CV F1 Scores: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        
        # Fit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        # Compute metrics
        results[name] = {
            "metrics": compute_metrics(y_test, y_pred, y_prob),
            "model": model,
            "y_prob": y_prob
        }
    
    # 5. Save primary Random Forest and scalers
    print("\n[INFO] Saving primary Random Forest model and scaling pipelines...")
    primary_rf = results["Random Forest"]["model"]
    with open(os.path.join(output_dir, "trained_gbm.pkl"), "wb") as f: # Save RF to trained_gbm.pkl for backward compatibility in worker load
        pickle.dump(primary_rf, f)
    with open(os.path.join(output_dir, "minmax_scaler.pkl"), "wb") as f:
        pickle.dump(minmax, f)
    with open(os.path.join(output_dir, "zscore_scaler.pkl"), "wb") as f:
        pickle.dump(zscore, f)
        
    # Create and save SHAP TreeExplainer
    print("[INFO] Serializing SHAP Tree Explainer for clinical reasoning...")
    explainer = shap.TreeExplainer(primary_rf)
    with open(os.path.join(output_dir, "shap_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)
    print("[SUCCESS] Models, scalers, and SHAP explainer serialized.")

    # 6. xAI Decision Tree Surrogate
    print("\n[INFO] Generating xAI Decision Tree Surrogate...")
    surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate.fit(X_train, primary_rf.predict(X_train))
    
    fig, ax = plt.subplots(figsize=(15, 9))
    plot_tree(
        surrogate,
        feature_names=df_features.columns.tolist(),
        class_names=["Normal", "Abnormal"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    plt.title("HealthFi xAI Global Surrogate Decision Tree Split Paths (Binary)", fontsize=14, fontweight='bold', pad=15)
    temp_tree_path = os.path.join(output_dir, "temp_tree.png")
    plt.savefig(temp_tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    add_shadow_and_border(temp_tree_path, os.path.join(output_dir, "xai_tree_plot.png"))
    if os.path.exists(temp_tree_path):
        os.remove(temp_tree_path)
        
    # 7. Donut chart
    print("\n[INFO] Plotting State & Anomaly Distribution Donut chart...")
    fig, ax = plt.subplots(figsize=(8, 8))
    outer_vals = [state_durations[s] for s in STATES_TO_SIMULATE]
    outer_colors = ["#1e293b", "#e11d48", "#f59e0b", "#d97706", "#b45309", "#06b6d4", "#2563eb", "#ffb4ab", "#93000a", "#10b981", "#a7f3d0", "#8b5cf6", "#a855f7", "#ec4899", "#f43f5e", "#ea580c", "#e11d48"]
    outer_labels = ["Stable", "PVC", "PAC", "Tachycardia", "Bradycardia", "AFib", "AFlutter", "SpO2 Desat", "Hypertensive", "Hypotension", "Resp Distress", "VTach", "Ischemia", "STEMI", "Heart Failure", "Sepsis", "VFib"]
    
    inner_vals = [state_durations[PatientState.STABLE], sum(state_durations[s] for s in STATES_TO_SIMULATE if s != PatientState.STABLE)]
    inner_colors = ["#334155", "#991b1b"]
    inner_labels = ["Normal", "Abnormal"]
    
    wedges1, texts1, autotexts1 = ax.pie(outer_vals, radius=1.0, colors=outer_colors[:len(STATES_TO_SIMULATE)], labels=outer_labels,
           wedgeprops=dict(width=0.2, edgecolor='w'), autopct='%1.1f%%', pctdistance=0.88, labeldistance=1.08)
    wedges2, texts2, autotexts2 = ax.pie(inner_vals, radius=0.8, colors=inner_colors, labels=inner_labels,
           wedgeprops=dict(width=0.2, edgecolor='w'), autopct='%1.1f%%', pctdistance=0.68, labeldistance=0.42)
           
    for t in texts1 + texts2:
        t.set_fontsize(7)
        t.set_fontweight('bold')
    for at in autotexts1 + autotexts2:
        at.set_fontsize(7)
        at.set_fontweight('bold')
        at.set_color('white')
    
    centre_circle = plt.Circle((0,0), 0.6, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.text(0, 0.1, "Balanced\nCohort", ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')
    ax.text(0, -0.15, f"{df_features.shape[0]} Segments", ha='center', va='center', fontsize=11, fontweight='medium', color='#64748b')
    
    plt.title("HealthFi Pretrain Cohort Modality & Anomaly Distribution", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    temp_donut_path = os.path.join(output_dir, "temp_donut.png")
    plt.savefig(temp_donut_path, dpi=300, bbox_inches='tight')
    plt.close()
    add_shadow_and_border(temp_donut_path, os.path.join(output_dir, "state_distribution_donut.png"))
    if os.path.exists(temp_donut_path):
        os.remove(temp_donut_path)
 
    # 8. Grouped Bar Chart of F1 & Accuracy for all 5 models
    print("[INFO] Plotting model comparison bar chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(models.keys())
    accuracy_scores = [results[m]["metrics"]["Accuracy"] for m in model_names]
    f1_scores = [results[m]["metrics"]["F1 Score"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='#06b6d4')
    ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#8b5cf6')
    
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title('Classifier Comparison on Telemetry Test Set', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    temp_subgroups_path = os.path.join(output_dir, "temp_subgroups.png")
    plt.savefig(temp_subgroups_path, dpi=300, bbox_inches='tight')
    plt.close()
    add_shadow_and_border(temp_subgroups_path, os.path.join(output_dir, "across_tasks_performance.png"))
    if os.path.exists(temp_subgroups_path):
        os.remove(temp_subgroups_path)
 
    # 9. Multiclass ROC Curves
    print("[INFO] Plotting ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ["#06b6d4", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899"]
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc_score = res["metrics"]["ROC-AUC"]
        ax.plot(fpr, tpr, color=colors[i], lw=2.5, label=f'{name} (AUC = {auc_score:.3f})')
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', label='Random Guess')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="lower right", frameon=True, shadow=True)
    
    plt.tight_layout()
    temp_roc_path = os.path.join(output_dir, "temp_roc.png")
    plt.savefig(temp_roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    add_shadow_and_border(temp_roc_path, os.path.join(output_dir, "roc_curves.png"))
    if os.path.exists(temp_roc_path):
        os.remove(temp_roc_path)
 
    # 10. Performance Radar / Polar Chart for anomaly states
    print("[INFO] Plotting polar radar chart comparison...")
    anomaly_states = [s for s in STATES_TO_SIMULATE if s != PatientState.STABLE]
    state_names = [s.value for s in anomaly_states]
    
    # Calculate slice sizes for specific test anomalies
    test_idx_array = np.array(test_idx)
    
    # Find normal test samples (where y_test == 0)
    normal_test_mask = (y_test == 0)
    X_test_normal = X_test[normal_test_mask]
    y_test_normal = y_test[normal_test_mask]
    
    aurocs_rf = []
    aurocs_xgb = []
    aurocs_lgb = []
    
    # Map index to anomaly states in y_test
    # The anomaly states in test_idx start after the stable test samples
    normal_test_count = len(X_test_normal)
    anomaly_test_count = (duration_per_state - 2) - int(0.8 * (duration_per_state - 2))
    
    for idx, state in enumerate(anomaly_states):
        # Slice the specific anomaly test samples
        start_a = normal_test_count + idx * anomaly_test_count
        end_a = start_a + anomaly_test_count
        X_test_anomaly = X_test[start_a:end_a]
        y_test_anomaly = np.ones(len(X_test_anomaly))
        
        X_sub = np.vstack([X_test_normal, X_test_anomaly])
        y_sub = np.concatenate([np.zeros(len(X_test_normal)), y_test_anomaly])
        
        # RF
        probs_rf = results["Random Forest"]["model"].predict_proba(X_sub)[:, 1]
        aurocs_rf.append(roc_auc_score(y_sub, probs_rf))
        
        # XGB
        probs_xgb = results["XGBoost"]["model"].predict_proba(X_sub)[:, 1]
        aurocs_xgb.append(roc_auc_score(y_sub, probs_xgb))
        
        # LGB
        probs_lgb = results["LightGBM"]["model"].predict_proba(X_sub)[:, 1]
        aurocs_lgb.append(roc_auc_score(y_sub, probs_lgb))
        
    num_vars = len(state_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], state_names, color='#1e293b', size=8, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    stats_rf = aurocs_rf + aurocs_rf[:1]
    ax.plot(angles, stats_rf, color='#8b5cf6', linewidth=2, label='Random Forest')
    ax.fill(angles, stats_rf, color='#8b5cf6', alpha=0.15)
    
    stats_xgb = aurocs_xgb + aurocs_xgb[:1]
    ax.plot(angles, stats_xgb, color='#06b6d4', linewidth=2, label='XGBoost')
    ax.fill(angles, stats_xgb, color='#06b6d4', alpha=0.15)
    
    stats_lgb = aurocs_lgb + aurocs_lgb[:1]
    ax.plot(angles, stats_lgb, color='#10b981', linewidth=2, label='LightGBM')
    ax.fill(angles, stats_lgb, color='#10b981', alpha=0.15)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, shadow=True)
    plt.title("HealthFi Multiclass Diagnostic Coverage (AUROC per Anomaly)", fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    temp_radar_path = os.path.join(output_dir, "temp_radar.png")
    plt.savefig(temp_radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    add_shadow_and_border(temp_radar_path, os.path.join(output_dir, "radar_performance_comparison.png"))
    if os.path.exists(temp_radar_path):
        os.remove(temp_radar_path)

    print("\n==========================================================================")
    print("                       MODEL PERFORMANCE COMPARISON                       ")
    print("==========================================================================")
    print(f" {'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'Sens':<6} | {'Spec':<6}")
    print("--------------------------------------------------------------------------")
    for name, res in results.items():
        m = res["metrics"]
        print(f" {name:<20} | {m['Accuracy']:<6.3f} | {m['Precision']:<6.3f} | {m['Recall']:<6.3f} | {m['F1 Score']:<6.3f} | {m['ROC-AUC']:<7.3f} | {m['Sensitivity']:<6.3f} | {m['Specificity']:<6.3f}")
    print("==========================================================================")
    print("[INFO] Model training process complete.")
    
    # Generate structured metrics table
    comparison_data = {
        "Model": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }
    for name, res in results.items():
        comparison_data["Model"].append(name)
        comparison_data["Precision"].append(res["metrics"]["Precision"])
        comparison_data["Recall"].append(res["metrics"]["Recall"])
        comparison_data["F1 Score"].append(res["metrics"]["F1 Score"])
    results_df = pd.DataFrame(comparison_data)
    return results_df



def main():
    run_ml_pipeline(duration_per_state=600, output_dir=".")

if __name__ == "__main__":
    main()
