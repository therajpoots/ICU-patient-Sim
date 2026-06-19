"""
DSP Preprocessing Module
Provides bandpass/notch filtering, Pan-Tompkins QRS detection,
detrending, and windowing functions for ICU patient signals.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter."""
    if len(data) < order * 3: # Handle short segments safely
        return data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.filtfilt(b, a, data)

def notch_filter(data: np.ndarray, notch_freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """Applies a zero-phase IIR notch filter to remove powerline noise."""
    if len(data) < 15: # Handle short segments
        return data
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, data)

def clean_ecg(ecg: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Filters raw ECG using a bandpass filter (0.5-35 Hz) and 50Hz notch filter."""
    # 1. Detrend to remove general DC offset
    detrended = signal.detrend(ecg)
    # 2. Apply notch filter to remove 50Hz powerline interference
    notched = notch_filter(detrended, 50.0, fs)
    # 3. Apply bandpass filter to capture QRS (0.5-35 Hz) with order=3 for sharper cutoff
    cleaned = butter_bandpass_filter(notched, 0.5, 35.0, fs, order=3)
    return cleaned

def clean_ppg(ppg: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Filters raw PPG using a bandpass filter (0.5-8 Hz) and 50Hz notch filter."""
    detrended = signal.detrend(ppg)
    notched = notch_filter(detrended, 50.0, fs)
    cleaned = butter_bandpass_filter(notched, 0.5, 8.0, fs)
    return cleaned

def clean_rsp(rsp: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Filters raw Respiratory signal using a bandpass filter (0.05-1.0 Hz)."""
    detrended = signal.detrend(rsp)
    cleaned = butter_bandpass_filter(detrended, 0.05, 1.0, fs)
    return cleaned

def pan_tompkins_qrs_detector(ecg: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Implements the Pan-Tompkins algorithm for QRS detection.
    Returns indices of detected R-peaks.
    """
    # 1. Bandpass filter (5-15 Hz) to isolate QRS complexes
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = signal.butter(1, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, ecg)

    # 2. Derivative filter (highlight QRS slope)
    # Filter transfer function approximation: h = [2, 1, 0, -1, -2] / 8.0
    kernel = np.array([-2, -1, 0, 1, 2]) / 8.0
    derived = np.convolve(filtered, kernel, mode='same')

    # 3. Squaring (intensifies QRS, makes all values positive)
    squared = derived ** 2

    # 4. Moving Window Integration (MWI)
    # Width of integration window should be approximately 150 ms (37.5 samples @ 250 Hz)
    window_len = int(0.150 * fs)
    mwi = np.convolve(squared, np.ones(window_len) / window_len, mode='same')

    # 5. Adaptive peak finding
    # Minimum refractory period of 200 ms between R-peaks (50 samples @ 250 Hz)
    min_dist = int(0.200 * fs)
    
    # We find peaks in the moving integration window
    # Prominence threshold prevents triggering on low amplitude noise
    prominence = 0.01 if np.max(mwi) < 1e-5 else 0.35 * np.max(mwi)
    mwi_peaks, _ = signal.find_peaks(mwi, distance=min_dist, prominence=prominence)

    # Map MWI peaks back to local maximum in bandpass-filtered ECG (within a 100 ms search window)
    r_peaks = []
    search_win = int(0.100 * fs)
    for p in mwi_peaks:
        start = max(0, p - search_win)
        end = min(len(ecg), p + search_win)
        if start >= end:
            continue
        local_max_idx = start + np.argmax(np.abs(ecg[start:end]))
        r_peaks.append(local_max_idx)

    return np.array(r_peaks, dtype=int)
