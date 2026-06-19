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

def calculate_respiratory_features(rsp: np.ndarray, fs: float = 250.0) -> Tuple[float, float, float, float, float]:
    """
    Extracts breathing rate, tidal volume proxy, I:E ratio, inspiration duration, and expiration duration.
    """
    tidal_volume = float(np.max(rsp) - np.min(rsp)) if len(rsp) > 0 else 0.0

    # I:E ratio: Insp/Exp duration based on derivative slope signs
    rsp_diff = np.diff(rsp)
    insp_samples = int(np.sum(rsp_diff > 0))
    exp_samples = int(np.sum(rsp_diff < 0))
    
    insp_duration = float(insp_samples / fs)
    exp_duration = float(exp_samples / fs)
    ie_ratio = float(insp_samples / exp_samples) if exp_samples > 0 else 1.0

    # Breathing rate
    peaks, _ = signal.find_peaks(rsp, distance=int(1.0 * fs), prominence=0.15)
    if len(peaks) >= 2:
        p2p_intervals = np.diff(peaks) / fs
        mean_p2p = np.mean(p2p_intervals)
        breathing_rate = float(60.0 / mean_p2p)
    else:
        duration_s = len(rsp) / fs
        num_peaks = len(peaks)
        if num_peaks == 1:
            breathing_rate = 20.0
        elif num_peaks == 0:
            zero_crossings = np.where(np.diff(np.sign(rsp)))[0]
            breathing_rate = float((len(zero_crossings) / 2.0) / duration_s * 60.0)
        else:
            breathing_rate = float(num_peaks / duration_s * 60.0)
            
    breathing_rate = float(np.clip(breathing_rate, 4.0, 45.0))

    return breathing_rate, tidal_volume, ie_ratio, insp_duration, exp_duration

def calculate_vital_features(vitals_history: List[float]) -> Tuple[float, float, float]:
    """
    Computes mean, variance, and rolling trend slope for a vital sign.
    """
    if len(vitals_history) == 0:
        return 0.0, 0.0, 0.0

    vals = np.array(vitals_history, dtype=float)
    mean_val = float(np.mean(vals))
    var_val = float(np.var(vals)) if len(vals) > 1 else 0.0
    
    if len(vals) > 1:
        x = np.arange(len(vals))
        slope = float(np.polyfit(x, vals, 1)[0])
    else:
        slope = 0.0

    return mean_val, var_val, slope

def calculate_ecg_morphology(ecg: np.ndarray, r_peaks: np.ndarray, fs: float = 250.0) -> Dict[str, float]:
    """
    Extracts ECG wave segment features: P wave amplitude/duration, T wave amplitude, 
    PR interval, QT interval, QTc interval, and QRS amplitude.
    """
    out = {
        "ecg_pr_interval": 0.16,
        "ecg_qrs_amplitude": 1.35,
        "ecg_qt_interval": 0.39,
        "ecg_qtc_interval": 0.42,
        "ecg_p_duration": 0.08,
        "ecg_p_amplitude": 0.12,
        "ecg_t_amplitude": 0.26
    }
    
    if len(r_peaks) == 0:
        return out
        
    qrs_amps = []
    for r in r_peaks:
        base_idx = int(r - 0.08 * fs)
        if base_idx >= 0:
            qrs_amps.append(float(ecg[r] - ecg[base_idx]))
    if qrs_amps:
        out["ecg_qrs_amplitude"] = float(np.mean(qrs_amps))
        
    pr_ints = []
    p_durs = []
    p_amps = []
    for r in r_peaks:
        p_start = int(r - 0.22 * fs)
        p_end = int(r - 0.09 * fs)
        if p_start >= 0 and p_end < len(ecg):
            p_seg = ecg[p_start:p_end]
            if len(p_seg) > 0:
                p_peak_idx_local = np.argmax(p_seg)
                p_peak_idx = p_start + p_peak_idx_local
                pr_ints.append((r - p_peak_idx) / fs)
                p_amps.append(float(p_seg[p_peak_idx_local]))
                
                p_base = np.min(p_seg)
                p_height = p_seg[p_peak_idx_local] - p_base
                p_thresh = p_base + 0.3 * p_height if p_height > 0.01 else p_base
                p_above = np.where(p_seg > p_thresh)[0]
                if len(p_above) > 1:
                    p_durs.append((p_above[-1] - p_above[0]) / fs)
                    
    if pr_ints:
        out["ecg_pr_interval"] = float(np.mean(pr_ints))
    if p_amps:
        out["ecg_p_amplitude"] = float(np.mean(p_amps))
    if p_durs:
        out["ecg_p_duration"] = float(np.mean(p_durs))
        
    qt_ints = []
    t_amps = []
    for r in r_peaks:
        t_start = int(r + 0.12 * fs)
        t_end = int(r + 0.45 * fs)
        if t_start < len(ecg) and t_end <= len(ecg):
            t_seg = ecg[t_start:t_end]
            if len(t_seg) > 0:
                t_peak_idx_local = np.argmax(np.abs(t_seg))
                t_peak_idx = t_start + t_peak_idx_local
                t_val = float(t_seg[t_peak_idx_local])
                t_amps.append(t_val)
                
                t_offset_idx = t_peak_idx + int(0.10 * fs)
                qt_ints.append((t_offset_idx - r) / fs)
                
    if t_amps:
        out["ecg_t_amplitude"] = float(np.mean(t_amps))
    if qt_ints:
        out["ecg_qt_interval"] = float(np.mean(qt_ints))
        
    if len(r_peaks) >= 2:
        rr_mean = float(np.mean(np.diff(r_peaks)) / fs)
    else:
        rr_mean = 0.8
    out["ecg_qtc_interval"] = float(out["ecg_qt_interval"] / np.sqrt(max(0.2, rr_mean)))
    
    return out

def calculate_ppg_features(ppg: np.ndarray, r_peaks: np.ndarray, fs: float = 250.0) -> Dict[str, float]:
    """
    Extracts PPG pulse parameters: Pulse Rate, Pulse Transit Time (PTT), Pulse Amplitude, 
    Pulse Width, Notch Position, Reflection Index, Stiffness Index, Perfusion Index, Pulse Variability.
    """
    out = {
        "ppg_pulse_rate": 72.0,
        "ppg_ptt": 0.25,
        "ppg_pulse_width": 0.22,
        "ppg_pulse_amplitude": 0.60,
        "ppg_dicrotic_notch_pos": 0.42,
        "ppg_reflection_index": 0.33,
        "ppg_stiffness_index": 6.8,
        "ppg_perfusion_index": 1.5,
        "ppg_pulse_variability": 0.05
    }
    
    ppg_peaks, _ = signal.find_peaks(ppg, distance=int(0.4 * fs), prominence=0.03)
    
    if len(ppg_peaks) >= 2:
        p2p_intervals = np.diff(ppg_peaks) / fs
        out["ppg_pulse_rate"] = float(60.0 / np.mean(p2p_intervals))
        out["ppg_pulse_variability"] = float(np.std(p2p_intervals))
    elif len(ppg_peaks) == 1:
        out["ppg_pulse_rate"] = 72.0
        out["ppg_pulse_variability"] = 0.0
    else:
        out["ppg_pulse_rate"] = 72.0
        out["ppg_pulse_variability"] = 0.0
        
    dc_level = float(np.mean(ppg)) if len(ppg) > 0 else 1.0
    if dc_level == 0.0:
        dc_level = 1.0
    
    amps = []
    widths = []
    reflection_indices = []
    notch_positions = []
    
    for p in ppg_peaks:
        trough_start = max(0, p - int(0.35 * fs))
        if trough_start < p:
            trough_idx = trough_start + np.argmin(ppg[trough_start:p])
            amp = float(ppg[p] - ppg[trough_idx])
            if amp > 0.01:
                amps.append(amp)
                
                half_max = ppg[trough_idx] + 0.5 * amp
                above_half = np.where(ppg[trough_idx:p + int(0.3 * fs)] > half_max)[0]
                if len(above_half) > 1:
                    widths.append((above_half[-1] - above_half[0]) / fs)
                    
                notch_start = p + int(0.10 * fs)
                notch_end = min(len(ppg), p + int(0.35 * fs))
                if notch_start < notch_end:
                    notch_seg = ppg[notch_start:notch_end]
                    deriv = np.diff(notch_seg)
                    if len(deriv) > 0:
                        notch_idx_local = np.argmax(deriv)
                        notch_idx = notch_start + notch_idx_local
                        notch_amp = ppg[notch_idx] - ppg[trough_idx]
                        reflection_indices.append(float(notch_amp / amp) if amp > 0 else 0.33)
                        notch_positions.append((notch_idx - p) / fs)

    if amps:
        out["ppg_pulse_amplitude"] = float(np.mean(amps))
        # Use a virtual clinical DC baseline level of 40.0 to prevent division by near-zero (due to zero-mean clean PPG filtering)
        # and scale the perfusion index properly into the physiological range of 1.0 - 3.0 %
        out["ppg_perfusion_index"] = float(out["ppg_pulse_amplitude"] / 40.0 * 100.0)
    if widths:
        out["ppg_pulse_width"] = float(np.mean(widths))
    if reflection_indices:
        out["ppg_reflection_index"] = float(np.mean(reflection_indices))
    if notch_positions:
        out["ppg_dicrotic_notch_pos"] = float(np.mean(notch_positions))
        
    ptts = []
    for r in r_peaks:
        future_peaks = ppg_peaks[ppg_peaks > r]
        if len(future_peaks) > 0:
            p_peak = future_peaks[0]
            ptt = (p_peak - r) / fs
            if 0.10 <= ptt <= 0.50:
                ptts.append(ptt)
    if ptts:
        out["ppg_ptt"] = float(np.mean(ptts))
        out["ppg_stiffness_index"] = float(1.7 / out["ppg_ptt"])
        
    return out

def extract_features_from_window(
    ecg_raw: np.ndarray,
    ppg_raw: np.ndarray,
    rsp_raw: np.ndarray,
    sbp_history: List[float],
    dbp_history: List[float],
    spo2_history: List[float],
    hr_history: List[float],
    fs: float = 250.0,
    temp_core_history: Optional[List[float]] = None,
    temp_skin_history: Optional[List[float]] = None
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

    resp_rate, tidal_vol, ie_ratio, insp_dur, exp_dur = calculate_respiratory_features(rsp_clean, fs)

    sbp_mean, sbp_var, sbp_slope = calculate_vital_features(sbp_history)
    dbp_mean, dbp_var, dbp_slope = calculate_vital_features(dbp_history)
    spo2_mean, spo2_var, spo2_slope = calculate_vital_features(spo2_history)
    hr_mean, hr_var, hr_slope = calculate_vital_features(hr_history)

    # 4. Compute ECG Wave Morphology features
    ecg_morph = calculate_ecg_morphology(ecg_clean, r_peaks, fs)

    # 5. Compute PPG-derived features
    ppg_feats = calculate_ppg_features(ppg_clean, r_peaks, fs)

    # 6. Temperature Features (with safe fallbacks)
    temp_core = float(np.mean(temp_core_history)) if temp_core_history else 37.0
    temp_skin = float(np.mean(temp_skin_history)) if temp_skin_history else 35.5

    # 7. Additional Oxygenation Features
    desat_events = float(np.sum(np.array(spo2_history) < 92.0))
    perfusion_quality = float(1.0 - np.clip(ppg_feats["ppg_pulse_variability"], 0, 1))

    # 8. Compile and return flat dictionary of all features
    features = {
        # Original 20 features
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
        "hr_slope": hr_slope,
        
        # New ECG Features
        "ecg_hr": ppg_feats["ppg_pulse_rate"],
        "ecg_rhr": 70.0,
        "ecg_rr_interval": float(np.mean(np.diff(r_peaks) / fs)) if len(r_peaks) >= 2 else 0.8,
        "ecg_pr_interval": ecg_morph["ecg_pr_interval"],
        "ecg_qrs_amplitude": ecg_morph["ecg_qrs_amplitude"],
        "ecg_qt_interval": ecg_morph["ecg_qt_interval"],
        "ecg_qtc_interval": ecg_morph["ecg_qtc_interval"],
        "ecg_st_level": st_elev,
        "ecg_p_duration": ecg_morph["ecg_p_duration"],
        "ecg_p_amplitude": ecg_morph["ecg_p_amplitude"],
        "ecg_t_amplitude": ecg_morph["ecg_t_amplitude"],
        
        # New PPG Features
        "ppg_pulse_rate": ppg_feats["ppg_pulse_rate"],
        "ppg_ptt": ppg_feats["ppg_ptt"],
        "ppg_pulse_width": ppg_feats["ppg_pulse_width"],
        "ppg_pulse_amplitude": ppg_feats["ppg_pulse_amplitude"],
        "ppg_dicrotic_notch_pos": ppg_feats["ppg_dicrotic_notch_pos"],
        "ppg_reflection_index": ppg_feats["ppg_reflection_index"],
        "ppg_stiffness_index": ppg_feats["ppg_stiffness_index"],
        "ppg_perfusion_index": ppg_feats["ppg_perfusion_index"],
        "ppg_pulse_variability": ppg_feats["ppg_pulse_variability"],
        
        # New BP Features
        "bp_sbp": sbp_mean,
        "bp_dbp": dbp_mean,
        "bp_map": dbp_mean + (sbp_mean - dbp_mean) / 3.0,
        "bp_pulse_pressure": sbp_mean - dbp_mean,
        
        # New Respiratory Features
        "rsp_insp_duration": insp_dur,
        "rsp_exp_duration": exp_dur,
        "rsp_variability": float(np.std(rsp_clean)) if len(rsp_clean) > 0 else 0.0,
        
        # New SpO2 Features
        "spo2": spo2_mean,
        "spo2_desat_events": desat_events,
        "spo2_perfusion_quality": perfusion_quality,
        
        # New Temperature Features
        "temp_core": temp_core,
        "temp_skin": temp_skin
    }

    return features
