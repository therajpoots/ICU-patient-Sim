"""
Feature Extraction Module
Extracts mathematical features from preprocessed 3-second windows of ECG, PPG, RSP, SpO2, and BP,
and provides min-max and z-score normalization utilities.
"""

import numpy as np
from scipy import signal
from typing import List, Dict, Any, Tuple, Optional
from simulator.preprocessing import clean_ecg, clean_ppg, clean_rsp, pan_tompkins_qrs_detector

def calculate_qrs_duration(ecg: np.ndarray, r_peaks: np.ndarray, fs: float = 250.0) -> float:
    """
    Estimates the average QRS duration in seconds.
    Measures peak width at 20% height in the bandpass-filtered ECG around R-peaks.
    """
    if len(r_peaks) == 0:
        return 0.100  # Default normal QRS duration (100 ms)

    widths = []
    search_win = int(0.08 * fs)  # search window of 80ms around peak
    for r in r_peaks:
        start = max(0, r - search_win)
        end = min(len(ecg), r + search_win)
        segment = ecg[start:end]
        if len(segment) == 0:
            continue
        
        # Height of peak relative to segment minimum
        peak_val = ecg[r]
        base_val = np.min(segment)
        height = peak_val - base_val
        threshold = base_val + 0.2 * height
        
        # Find width where signal is above threshold
        above_threshold = np.where(segment > threshold)[0]
        if len(above_threshold) > 1:
            duration = (above_threshold[-1] - above_threshold[0]) / fs
            # Constraint QRS duration to physiological bounds (50ms - 250ms)
            if 0.04 <= duration <= 0.25:
                widths.append(duration)
                
    if len(widths) == 0:
        return 0.100
    return float(np.mean(widths))

def calculate_st_elevation(ecg: np.ndarray, r_peaks: np.ndarray, fs: float = 250.0) -> float:
    """
    Estimates the average ST elevation in millivolts (assuming raw ECG scale).
    Compares the TP/PR baseline (80-120ms before R-peak) to the ST segment (60-100ms after R-peak).
    """
    if len(r_peaks) == 0:
        return 0.0

    elevations = []
    baseline_offset_start = int(0.120 * fs)
    baseline_offset_end = int(0.080 * fs)
    st_offset_start = int(0.060 * fs)
    st_offset_end = int(0.100 * fs)

    for r in r_peaks:
        # Baseline window before QRS
        b_start = r - baseline_offset_start
        b_end = r - baseline_offset_end
        # ST segment window after QRS
        st_start = r + st_offset_start
        st_end = r + st_offset_end

        if b_start >= 0 and st_end < len(ecg):
            baseline = np.mean(ecg[b_start:b_end])
            st_val = np.mean(ecg[st_start:st_end])
            elevations.append(st_val - baseline)

    if len(elevations) == 0:
        return 0.0
    return float(np.mean(elevations))

def calculate_hrv_features(r_peaks: np.ndarray, fs: float = 250.0) -> Tuple[float, float, float]:
    """
    Calculates HRV metrics (SDNN, RMSSD, pNN50) from R-peak indices.
    If R-peaks < 2, returns 0.0 for all.
    """
    if len(r_peaks) < 2:
        return 0.0, 0.0, 0.0

    # R-R intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs

    # 1. SDNN (Standard Deviation of R-R intervals)
    sdnn = float(np.std(rr_intervals))

    # 2. RMSSD (Root Mean Square of Successive Differences)
    rr_diffs = np.diff(rr_intervals)
    if len(rr_diffs) > 0:
        rmssd = float(np.sqrt(np.mean(rr_diffs ** 2)))
        # 3. pNN50 (Percentage of differences > 50ms)
        nn50 = np.sum(np.abs(rr_diffs) > 0.050)
        pnn50 = float((nn50 / len(rr_diffs)) * 100.0)
    else:
        rmssd = 0.0
        pnn50 = 0.0

    return sdnn, rmssd, pnn50

def calculate_respiratory_features(rsp: np.ndarray, fs: float = 250.0) -> Tuple[float, float, float]:
    """
    Extracts breathing rate, tidal volume proxy, and I:E ratio from cleaned RSP waveform.
    """
    # 1. Tidal volume proxy: Peak-to-trough amplitude
    tidal_volume = float(np.max(rsp) - np.min(rsp)) if len(rsp) > 0 else 0.0

    # 2. I:E ratio: Insp/Exp duration proxy based on derivative slope signs
    rsp_diff = np.diff(rsp)
    insp_samples = np.sum(rsp_diff > 0)
    exp_samples = np.sum(rsp_diff < 0)
    ie_ratio = float(insp_samples / exp_samples) if exp_samples > 0 else 1.0

    # 3. Breathing rate (breaths per minute)
    # Since window is 3s, zero crossings or peak counts are very coarse.
    # We find peaks with a minimum distance of 1.0 seconds (60 bpm ceiling)
    peaks, _ = signal.find_peaks(rsp, distance=int(1.0 * fs), prominence=0.15)
    if len(peaks) >= 2:
        # Average peak-to-peak interval in seconds
        p2p_intervals = np.diff(peaks) / fs
        mean_p2p = np.mean(p2p_intervals)
        breathing_rate = float(60.0 / mean_p2p)
    else:
        # Fallback based on counting peaks/crossings in 3-second window
        duration_s = len(rsp) / fs
        # Estimate from peaks
        num_peaks = len(peaks)
        if num_peaks == 1:
            # Assume 1 breath in 3s -> ~20 bpm
            breathing_rate = 20.0
        elif num_peaks == 0:
            # Let's count zero crossings as fallback
            zero_crossings = np.where(np.diff(np.sign(rsp)))[0]
            # 2 zero crossings per breath cycles
            breathing_rate = float((len(zero_crossings) / 2.0) / duration_s * 60.0)
        else:
            breathing_rate = float(num_peaks / duration_s * 60.0)
            
    # Clamp to reasonable physiologic bounds [4, 45]
    breathing_rate = float(np.clip(breathing_rate, 4.0, 45.0))

    return breathing_rate, tidal_volume, ie_ratio

def calculate_vital_features(vitals_history: List[float]) -> Tuple[float, float, float]:
    """
    Computes mean, variance, and rolling trend slope for a vital sign.
    vitals_history represents vital values over the window.
    """
    if len(vitals_history) == 0:
        return 0.0, 0.0, 0.0

    vals = np.array(vitals_history, dtype=float)
    mean_val = float(np.mean(vals))
    var_val = float(np.var(vals)) if len(vals) > 1 else 0.0
    
    # Linear trend slope (change per second)
    if len(vals) > 1:
        x = np.arange(len(vals))
        slope = float(np.polyfit(x, vals, 1)[0])
    else:
        slope = 0.0

    return mean_val, var_val, slope

def extract_features_from_window(
    ecg_raw: np.ndarray,
    ppg_raw: np.ndarray,
    rsp_raw: np.ndarray,
    sbp_history: List[float],
    dbp_history: List[float],
    spo2_history: List[float],
    hr_history: List[float],
    fs: float = 250.0
) -> Dict[str, float]:
    """
    Intakes raw waveforms and vital history, cleans the waveforms,
    extracts all mathematical features, and returns them as a flat dictionary.
    """
    # 1. Clean raw waveforms
    ecg_clean = clean_ecg(ecg_raw, fs)
    ppg_clean = clean_ppg(ppg_raw, fs)
    rsp_clean = clean_rsp(rsp_raw, fs)

    # 2. QRS Peak Detection
    r_peaks = pan_tompkins_qrs_detector(ecg_clean, fs)

    # 3. Compute features
    qrs_dur = calculate_qrs_duration(ecg_clean, r_peaks, fs)
    st_elev = calculate_st_elevation(ecg_clean, r_peaks, fs)
    sdnn, rmssd, pnn50 = calculate_hrv_features(r_peaks, fs)

    resp_rate, tidal_vol, ie_ratio = calculate_respiratory_features(rsp_clean, fs)

    sbp_mean, sbp_var, sbp_slope = calculate_vital_features(sbp_history)
    dbp_mean, dbp_var, dbp_slope = calculate_vital_features(dbp_history)
    spo2_mean, spo2_var, spo2_slope = calculate_vital_features(spo2_history)
    hr_mean, hr_var, hr_slope = calculate_vital_features(hr_history)

    return {
        "ecg_qrs_duration": qrs_dur,
        "ecg_st_elevation": st_elev,
        "hrv_sdnn": sdnn,
        "hrv_rmssd": rmssd,
        "hrv_pnn50": pnn50,
        "rsp_rate": resp_rate,
        "rsp_tidal_volume": tidal_vol,
        "rsp_ie_ratio": ie_ratio,
        "sbp_mean": sbp_mean,
        "sbp_var": sbp_var,
        "sbp_slope": sbp_slope,
        "dbp_mean": dbp_mean,
        "dbp_var": dbp_var,
        "dbp_slope": dbp_slope,
        "spo2_mean": spo2_mean,
        "spo2_var": spo2_var,
        "spo2_slope": spo2_slope,
        "hr_mean": hr_mean,
        "hr_var": hr_var,
        "hr_slope": hr_slope
    }
