"""
Automated Smoke Test for the ML and DSP Pipeline
Verifies filtering, peak detection, feature extraction, model creation, and dummy training runs.
"""

import os
import torch
import numpy as np
from simulator.preprocessing import clean_ecg, clean_ppg, clean_rsp, pan_tompkins_qrs_detector
from simulator.features import extract_features_from_window
from ml.models import create_supervised_model, create_isolation_forest, LSTMAutoencoder
from ml.train import run_ml_pipeline

def test_dsp():
    print("Testing DSP module...")
    fs = 250.0
    t = np.arange(750) / fs # 3 seconds
    # Generate ECG-like sinewave + 50Hz noise + baseline drift
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 50.0 * t) + 0.2 * np.sin(2 * np.pi * 0.1 * t)
    
    cleaned = clean_ecg(ecg, fs)
    assert len(cleaned) == len(ecg), "ECG cleaning output length mismatch"
    print("  ECG filtering OK")
    
    ppg = np.sin(2 * np.pi * 1.2 * t) ** 2 + 0.1 * np.sin(2 * np.pi * 50.0 * t)
    cleaned_ppg = clean_ppg(ppg, fs)
    assert len(cleaned_ppg) == len(ppg), "PPG cleaning output length mismatch"
    print("  PPG filtering OK")
    
    rsp = np.sin(2 * np.pi * 0.2 * t)
    cleaned_rsp = clean_rsp(rsp, fs)
    assert len(cleaned_rsp) == len(rsp), "RSP cleaning output length mismatch"
    print("  RSP filtering OK")
    
    # Test Pan-Tompkins on QRS impulses
    ecg_impulses = np.zeros(750)
    ecg_impulses[100] = 1.0
    ecg_impulses[350] = 1.0
    ecg_impulses[600] = 1.0
    peaks = pan_tompkins_qrs_detector(ecg_impulses, fs)
    assert len(peaks) > 0, "Pan-Tompkins failed to detect any peaks"
    print(f"  Pan-Tompkins QRS Detection OK (detected peaks: {peaks.tolist()})")

def test_feature_extraction():
    print("\nTesting Feature Extraction...")
    fs = 250.0
    ecg_raw = np.sin(2 * np.pi * 1.2 * np.arange(750) / fs)
    ppg_raw = np.sin(2 * np.pi * 1.2 * np.arange(750) / fs)
    rsp_raw = np.sin(2 * np.pi * 0.2 * np.arange(750) / fs)
    
    sbp_hist = [120.0, 122.0, 121.0]
    dbp_hist = [80.0, 79.0, 81.0]
    spo2_hist = [98.0, 98.0, 97.0]
    hr_hist = [72.0, 73.0, 71.0]
    
    features = extract_features_from_window(
        ecg_raw, ppg_raw, rsp_raw,
        sbp_hist, dbp_hist, spo2_hist, hr_hist, fs
    )
    
    assert isinstance(features, dict), "extract_features_from_window did not return a dict"
    assert len(features) >= 35, f"Expected at least 35 features, got {len(features)}"
    expected_keys = [
        "ecg_qrs_duration", "ecg_st_elevation",
        "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
        "rsp_rate", "rsp_tidal_volume", "rsp_ie_ratio",
        "sbp_mean", "sbp_var", "sbp_slope",
        "dbp_mean", "dbp_var", "dbp_slope",
        "spo2_mean", "spo2_var", "spo2_slope",
        "hr_mean", "hr_var", "hr_slope",
        "temp_core", "temp_skin", "ppg_ptt"
    ]
    for key in expected_keys:
        assert key in features, f"Feature key '{key}' missing"
        assert isinstance(features[key], (float, int)), f"Feature '{key}' is not a float or int"
    print("  Feature extraction keys and types OK")

def test_models():
    print("\nTesting ML Models creation...")
    gbm = create_supervised_model()
    assert gbm is not None, "Failed to create GBM"
    print("  GBM model creation OK")
    
    iforest = create_isolation_forest()
    assert iforest is not None, "Failed to create Isolation Forest"
    print("  Isolation Forest creation OK")
    
    # Test LSTM Autoencoder forward pass
    seq_len = 5
    input_dim = 20
    hidden_dim = 8
    batch_size = 4
    
    ae = LSTMAutoencoder(input_dim, hidden_dim, seq_len)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = ae(dummy_input)
    assert output.shape == (batch_size, seq_len, input_dim), f"Expected shape {(batch_size, seq_len, input_dim)}, got {output.shape}"
    print("  LSTM Autoencoder forward pass OK")

def test_train_pipeline_end_to_end():
    print("\nTesting Training Pipeline end-to-end (fast dummy run)...")
    temp_dir = "data/smoke_test_tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_plot = os.path.join(temp_dir, "xai_tree_plot.png")
    
    df_metrics = run_ml_pipeline(duration_per_state=15, output_dir=temp_dir)
    
    assert os.path.exists(temp_plot), "xAI decision tree plot was not created in temp dir"
    assert df_metrics is not None, "Metrics dataframe was not returned"
    
    # Cleanup temp directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("  End-to-end training pipeline OK")

def main():
    print("=" * 60)
    print("           ML AND DSP PIPELINE SMOKE TESTS")
    print("=" * 60)
    try:
        test_dsp()
        test_feature_extraction()
        test_models()
        test_train_pipeline_end_to_end()
        print("\n" + "=" * 60)
        print("           ALL ML SMOKE TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failure: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
