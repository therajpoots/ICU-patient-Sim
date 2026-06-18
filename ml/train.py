"""
ML Training and Evaluation Pipeline
1. Generates training data using the Patient Simulator in various states.
2. Extracts features in 3-second windows.
3. Fits Min-Max and Standard Scalers.
4. Trains Gradient Boosting, Isolation Forest, and PyTorch LSTM Autoencoder.
5. Computes Precision, Recall, and F1 scores.
6. Plots the xAI global surrogate decision tree and saves it.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from scipy.interpolate import make_interp_spline
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from simulator.patient_profile import PatientProfile
from simulator.arrhythmia_states import PatientState, ArrhythmiaStateMachine
from simulator.vitals import VitalsEngine, generate_ecg_segment, generate_ppg_segment, generate_rsp_segment
from simulator.features import extract_features_from_window
from ml.models import create_supervised_model, create_isolation_forest, LSTMAutoencoder

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLING_RATE = 250
WINDOW_DURATION_S = 3.0
WINDOW_SAMPLES = int(WINDOW_DURATION_S * SAMPLING_RATE) # 750 samples

# List of anomaly states for generating data
STATES_TO_SIMULATE = [
    PatientState.STABLE,
    PatientState.PVC,
    PatientState.TACHYCARDIA,
    PatientState.BRADYCARDIA,
    PatientState.AFIB,
    PatientState.DESATURATION,
    PatientState.HYPERTENSIVE,
    PatientState.RESP_DISTRESS,
    PatientState.VFIB
]

def generate_patient_data_for_state(state: PatientState, duration_seconds: int = 180) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Simulates a patient in a specific state for a duration and extracts features in 3-second windows.
    Returns:
        List of feature dicts, List of binary labels (0=Stable, 1=Anomaly)
    """
    logger.info(f"Simulating patient state: {state.value} for {duration_seconds}s...")
    
    # Initialize engines locally
    profile = PatientProfile()
    # Force state machine to stay in the chosen state
    sm = ArrhythmiaStateMachine(mean_interval_s=999999)
    ve = VitalsEngine(profile, sm)
    
    # Set the state explicitly
    if state != PatientState.STABLE:
        sm.trigger_anomaly(state.value)
    
    features_list = []
    labels = []
    
    # Waveform buffers (3 seconds)
    ecg_buffer = []
    ppg_buffer = []
    rsp_buffer = []
    
    # Vitals history buffers (3 seconds)
    sbp_history = []
    dbp_history = []
    spo2_history = []
    hr_history = []
    
    # Run second-by-second simulation
    for t in range(duration_seconds):
        vitals = ve.update(1.0)
        
        hr = vitals["heart_rate"]
        rr = vitals["respiratory_rate"]
        pvc_burden = vitals["pvc_burden"]
        irregularity = vitals["irregularity"]
        current_state = PatientState(vitals["patient_state"])
        
        # Generate 1 second of waveforms
        ecg_sec = generate_ecg_segment(hr, 1.0, SAMPLING_RATE, profile, current_state, pvc_burden, irregularity)
        ppg_sec = generate_ppg_segment(hr, 1.0, SAMPLING_RATE, profile, rr, current_state)
        rsp_sec = generate_rsp_segment(rr, 1.0, SAMPLING_RATE, profile)
        
        # Add to buffers
        ecg_buffer.extend(ecg_sec.tolist())
        ppg_buffer.extend(ppg_sec.tolist())
        rsp_buffer.extend(rsp_sec.tolist())
        
        sbp_history.append(vitals["systolic_bp"])
        dbp_history.append(vitals["diastolic_bp"])
        spo2_history.append(vitals["spo2"])
        hr_history.append(vitals["heart_rate"])
        
        # Every 3 seconds, extract features and reset buffers
        if (t + 1) % 3 == 0:
            feat = extract_features_from_window(
                np.array(ecg_buffer),
                np.array(ppg_buffer),
                np.array(rsp_buffer),
                sbp_history,
                dbp_history,
                spo2_history,
                hr_history,
                SAMPLING_RATE
            )
            features_list.append(feat)
            # Label: 0 if stable, 1 if any anomaly state
            labels.append(0 if state == PatientState.STABLE else 1)
            
            # Reset buffers
            ecg_buffer = []
            ppg_buffer = []
            rsp_buffer = []
            sbp_history = []
            dbp_history = []
            spo2_history = []
            hr_history = []
            
    return features_list, labels

def compile_dataset(duration_per_state: int = 180) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generates features for all states and compiles them into a DataFrame and label array."""
    all_features = []
    all_labels = []
    
    for state in STATES_TO_SIMULATE:
        feats, labels = generate_patient_data_for_state(state, duration_seconds=duration_per_state)
        all_features.extend(feats)
        all_labels.extend(labels)
        
    df = pd.DataFrame(all_features)
    return df, np.array(all_labels)

def train_lstm_autoencoder(model: nn.Module, train_seqs: np.ndarray, val_seqs: np.ndarray, epochs: int = 30, lr: float = 0.005) -> Tuple[nn.Module, float]:
    """Trains the LSTM Autoencoder on training sequences (stable baseline data)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_tensor = torch.FloatTensor(train_seqs)
    val_tensor = torch.FloatTensor(val_seqs)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_tensor)
        loss = criterion(outputs, train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_tensor)
                val_loss = criterion(val_outputs, val_tensor).item()
            logger.info(f"LSTM Autoencoder Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss:.5f}")
            model.train()
            
    # Compute threshold on validation set (95th percentile of reconstruction errors)
    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor).numpy()
    errors = np.mean((val_seqs - val_recon) ** 2, axis=(1, 2))
    threshold = float(np.percentile(errors, 95))
    logger.info(f"LSTM Autoencoder Threshold set to: {threshold:.5f} (95th percentile of baseline reconstruction error)")
    return model, threshold

def create_sequence_matrices(data: np.ndarray, seq_len: int = 5) -> np.ndarray:
    """Slices a 2D features array into 3D sequence matrices of shape (samples, seq_len, features)."""
    if len(data) == 0:
        return np.empty((0, seq_len, data.shape[1] if data.ndim > 1 else 0))
    if len(data) < seq_len:
        # Pad by repeating the last row to meet seq_len
        pad_width = seq_len - len(data)
        last_row = data[-1:]
        padding = np.repeat(last_row, pad_width, axis=0)
        data = np.concatenate([data, padding], axis=0)
        
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i : i + seq_len])
    return np.array(seqs)

def add_shadow_and_border(image_path: str, output_path: str):
    """Adds a soft drop shadow and a sleek border to an image using Pillow for publication-quality xAI and performance plots."""
    img = Image.open(image_path).convert("RGBA")
    
    offset = (15, 15)
    border = 30
    
    # Create background canvas with border space
    bg_size = (img.size[0] + border * 2, img.size[1] + border * 2)
    bg = Image.new("RGBA", bg_size, (255, 255, 255, 0)) # transparent background
    
    # Create shadow mask
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    # Draw rounded rectangle shadow base
    shadow_draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius=16, fill=(0, 0, 0, 75))
    # Blur the shadow
    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(12))
    
    # Paste shadow onto background
    bg.paste(shadow_blur, (border + offset[0], border + offset[1]), shadow_blur)
    
    # Create rounded corner mask for main image
    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius=16, fill=255)
    
    # Crop / paste image with mask
    img_with_corners = img.copy()
    img_with_corners.putalpha(mask)
    
    # Paste main image
    bg.paste(img_with_corners, (border, border), img_with_corners)
    
    # Save as RGB PNG
    final_img = Image.new("RGB", bg_size, (255, 255, 255))
    final_img.paste(bg, (0, 0), bg)
    final_img.save(output_path, "PNG")

def run_ml_pipeline(duration_per_state: int = 180, output_dir: str = "."):
    """Runs data generation, training, evaluation, and plotting."""
    logger.info("Starting ML Anomaly Detection Pipeline...")
    
    # 1. Compile dataset
    df_features, labels = compile_dataset(duration_per_state=duration_per_state)
    logger.info(f"Dataset compiled: {df_features.shape[0]} samples, {df_features.shape[1]} features.")
    
    # 2. Chronological Split (preserving state sequences)
    # We do a train/test split per state, then combine them.
    # This prevents leaking future temporal data to training.
    train_idx = []
    test_idx = []
    
    samples_per_state = duration_per_state // 3
    num_states = len(STATES_TO_SIMULATE)
    
    for i in range(num_states):
        start = i * samples_per_state
        end = start + samples_per_state
        split_point = int(start + 0.8 * samples_per_state)
        
        train_idx.extend(range(start, split_point))
        test_idx.extend(range(split_point, end))
        
    X_train_raw = df_features.iloc[train_idx].values
    y_train = labels[train_idx]
    X_test_raw = df_features.iloc[test_idx].values
    y_test = labels[test_idx]
    
    # 3. Fit Normalization (Stage 3b)
    # Min-Max first to [0, 1]
    minmax = MinMaxScaler()
    X_train_mm = minmax.fit_transform(X_train_raw)
    X_test_mm = minmax.transform(X_test_raw)
    
    # Z-Score scaling to mean=0, std=1
    zscore = StandardScaler()
    X_train = zscore.fit_transform(X_train_mm)
    X_test = zscore.transform(X_test_mm)
    
    # 4. Train Supervised Gradient Boosting Classifier (Stage 4)
    logger.info("Training Supervised Gradient Boosting Classifier...")
    gbm = create_supervised_model(n_estimators=100, max_depth=3)
    
    # Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gbm, X_train, y_train, cv=kfold, scoring='f1')
    logger.info(f"Supervised CV F1 Scores: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Fit model
    gbm.fit(X_train, y_train)
    y_pred_gbm = gbm.predict(X_test)
    
    # Evaluate
    prec_gbm, rec_gbm, f1_gbm, _ = precision_recall_fscore_support(y_test, y_pred_gbm, average='binary')
    logger.info(f"GBM Test Metrics - Precision: {prec_gbm:.4f} | Recall: {rec_gbm:.4f} | F1: {f1_gbm:.4f}")
    
    # 5. Train Unsupervised Isolation Forest (Stage 5)
    # Train ONLY on Stable (Normal) train data
    logger.info("Training Isolation Forest...")
    stable_train_idx = [i for i, label in enumerate(y_train) if label == 0]
    X_train_stable = X_train[stable_train_idx]
    
    if len(X_train_stable) == 0:
        # Fallback if no stable data
        X_train_stable = X_train
        
    iforest = create_isolation_forest(contamination=0.05)
    iforest.fit(X_train_stable)
    
    # Predict (-1: anomaly, 1: normal)
    iforest_preds = iforest.predict(X_test)
    # Map to binary (0=stable, 1=anomaly)
    y_pred_if = np.where(iforest_preds == -1, 1, 0)
    
    prec_if, rec_if, f1_if, _ = precision_recall_fscore_support(y_test, y_pred_if, average='binary')
    logger.info(f"Isolation Forest Test Metrics - Precision: {prec_if:.4f} | Recall: {rec_if:.4f} | F1: {f1_if:.4f}")
    
    # 6. Train PyTorch LSTM Autoencoder (Stage 5)
    logger.info("Training PyTorch LSTM Autoencoder...")
    seq_len = 5
    
    # Create sequence matrices for stable train data
    train_stable_seqs = create_sequence_matrices(X_train_stable, seq_len=seq_len)
    
    # Split train stable sequences into train and validation
    if len(train_stable_seqs) > 1:
        val_split = int(0.8 * len(train_stable_seqs))
        if val_split == 0:
            val_split = 1
        train_seqs = train_stable_seqs[:val_split]
        val_seqs = train_stable_seqs[val_split:]
    else:
        train_seqs = train_stable_seqs
        val_seqs = train_stable_seqs
        
    # Create sequence matrices for the test set
    X_test_seqs = create_sequence_matrices(X_test, seq_len=seq_len)
    y_test_seqs = y_test[seq_len - 1 :] # Ground truth labels align with the end of each sequence
    
    # Initialize and train LSTM Autoencoder
    input_dim = X_train.shape[1]
    ae = LSTMAutoencoder(input_dim=input_dim, hidden_dim=8, seq_len=seq_len)
    ae, threshold = train_lstm_autoencoder(ae, train_seqs, val_seqs, epochs=40)
    
    # Evaluate LSTM Autoencoder
    ae.eval()
    with torch.no_grad():
        test_recon = ae(torch.FloatTensor(X_test_seqs)).numpy()
    test_recon_errors = np.mean((X_test_seqs - test_recon) ** 2, axis=(1, 2))
    
    y_pred_ae = np.where(test_recon_errors > threshold, 1, 0)
    prec_ae, rec_ae, f1_ae, _ = precision_recall_fscore_support(y_test_seqs, y_pred_ae, average='binary')
    logger.info(f"LSTM Autoencoder Test Metrics - Precision: {prec_ae:.4f} | Recall: {rec_ae:.4f} | F1: {f1_ae:.4f}")
    
    # 7. xAI Global Surrogate Decision Tree Plotting
    logger.info("Training and Plotting xAI Global Surrogate Decision Tree...")
    surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate.fit(X_train, gbm.predict(X_train))
    
    fig, ax = plt.subplots(figsize=(15, 9))
    feature_names = df_features.columns.tolist()
    plot_tree(
        surrogate,
        feature_names=feature_names,
        class_names=["Stable", "Anomaly"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    plt.title("xAI Global Surrogate Decision Tree (Mimicking Gradient Boosting)", fontsize=14, fontweight='bold', pad=15)
    
    temp_tree_path = os.path.join(output_dir, "temp_tree.png")
    plt.savefig(temp_tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    tree_plot_path = os.path.join(output_dir, "xai_tree_plot.png")
    add_shadow_and_border(temp_tree_path, tree_plot_path)
    if os.path.exists(temp_tree_path):
        os.remove(temp_tree_path)
    logger.info(f"xAI Decision Tree plot saved to: {tree_plot_path}")
    
    # 8. Beautiful Grouped Performance Comparison Bar Chart
    logger.info("Plotting beautiful performance comparison chart...")
    
    models = ["GBM Forest", "Isolation Forest", "LSTM Autoencoder"]
    
    # Values
    precisions = [prec_gbm, prec_if, prec_ae]
    recalls = [rec_gbm, rec_if, rec_ae]
    f1s = [f1_gbm, f1_if, f1_ae]
    
    x = np.arange(len(models))
    width = 0.22
    
    colors = ["#00b4d8", "#7209b7", "#ffb703"] # Modern cyan, purple, amber
    
    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width, precisions, width, label='Precision', color=colors[0], edgecolor='none')
    rects2 = ax.bar(x, recalls, width, label='Recall', color=colors[1], edgecolor='none')
    rects3 = ax.bar(x + width, f1s, width, label='F1 Score', color=colors[2], edgecolor='none')
    
    # Styling labels and layout
    ax.set_ylabel('Score', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('ICU Pipeline Anomaly Detection - Model Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
    
    # Add values on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    temp_comp_path = os.path.join(output_dir, "temp_comp.png")
    plt.savefig(temp_comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    comp_plot_path = os.path.join(output_dir, "performance_comparison.png")
    add_shadow_and_border(temp_comp_path, comp_plot_path)
    if os.path.exists(temp_comp_path):
        os.remove(temp_comp_path)
    logger.info(f"Performance comparison plot saved to: {comp_plot_path}")

    # 9. Plotting smooth ROC curves
    logger.info("Plotting smooth ROC curves...")
    
    # Calculate scores
    gbm_scores = gbm.predict_proba(X_test)[:, 1]
    iforest_scores = -iforest.decision_function(X_test)
    
    # Helper to smooth ROC curve using make_interp_spline
    def get_smooth_roc(y_true, scores):
        fpr, tpr, _ = roc_curve(y_true, scores)
        # Sort and keep unique FPR to avoid interpolation crashes
        unique_fpr, unique_indices = np.unique(fpr, return_index=True)
        unique_tpr = tpr[unique_indices]
        
        # Interpolate for smooth curve
        x_new = np.linspace(0, 1, 200)
        try:
            spl = make_interp_spline(unique_fpr, unique_tpr, k=min(3, len(unique_fpr) - 1))
            y_new = spl(x_new)
            # Clip between 0 and 1
            y_new = np.clip(y_new, 0.0, 1.0)
            y_new[0] = 0.0
            y_new[-1] = 1.0
            return x_new, y_new, auc(unique_fpr, unique_tpr)
        except Exception:
            # Fallback to standard curve
            return fpr, tpr, auc(fpr, tpr)

    fpr_gbm_s, tpr_gbm_s, auc_gbm = get_smooth_roc(y_test, gbm_scores)
    fpr_if_s, tpr_if_s, auc_if = get_smooth_roc(y_test, iforest_scores)
    fpr_ae_s, tpr_ae_s, auc_ae = get_smooth_roc(y_test_seqs, test_recon_errors)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw ROC curves with beautiful thick smooth curves
    ax.plot(fpr_gbm_s, tpr_gbm_s, color='#06b6d4', lw=2.5, label=f'GBM Forest (AUC = {auc_gbm:.3f})')
    ax.plot(fpr_if_s, tpr_if_s, color='#8b5cf6', lw=2.5, label=f'Isolation Forest (AUC = {auc_if:.3f})')
    ax.plot(fpr_ae_s, tpr_ae_s, color='#f59e0b', lw=2.5, label=f'LSTM Autoencoder (AUC = {auc_ae:.3f})')
    
    # Random guess diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', label='Random Guess')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none', shadow=True)
    
    plt.tight_layout()
    temp_roc_path = os.path.join(output_dir, "temp_roc.png")
    plt.savefig(temp_roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    roc_plot_path = os.path.join(output_dir, "roc_curves.png")
    add_shadow_and_border(temp_roc_path, roc_plot_path)
    if os.path.exists(temp_roc_path):
        os.remove(temp_roc_path)
    logger.info(f"ROC curves plot saved to: {roc_plot_path}")
    
    # Generate structured metrics table
    results = {
        "Model": ["Gradient Boosting Forest", "Isolation Forest", "LSTM Autoencoder"],
        "Precision": [prec_gbm, prec_if, prec_ae],
        "Recall": [rec_gbm, rec_if, rec_ae],
        "F1 Score": [f1_gbm, f1_if, f1_ae]
    }
    results_df = pd.DataFrame(results)
    
    # Print beautiful table
    print("\n" + "="*60)
    print("                 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60 + "\n")
    
    return results_df
