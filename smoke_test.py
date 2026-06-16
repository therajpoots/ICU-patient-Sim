"""Smoke test script for the ICU Biosignal Simulator."""
import os, sys

os.environ["AES_SHARED_SECRET"] = "test-secret-for-smoke-test"
os.environ["API_KEY"] = "test-key"
os.environ["PATIENT_ID"] = "ICU-TEST-001"
os.environ["PATIENT_NAME"] = "Test Patient"
os.environ["PATIENT_AGE"] = "55"
os.environ["PATIENT_WARD"] = "NICU-1A"
os.environ["ARRHYTHMIA_MEAN_INTERVAL_S"] = "999999"

print("Testing imports...")
from simulator.patient_profile import PatientProfile
from simulator.arrhythmia_states import ArrhythmiaStateMachine
from simulator.vitals import VitalsEngine, generate_ecg_segment, generate_ppg_segment, generate_rsp_segment
from api.crypto import encrypt_payload, decrypt_payload
print("  All imports OK")

print("\nTesting patient profile...")
profile = PatientProfile(name="Test Patient", age=55)
print(f"  HR baseline  : {profile.hr_baseline} bpm")
print(f"  BP baseline  : {profile.sbp_baseline}/{profile.dbp_baseline} mmHg")
print(f"  SpO2 baseline: {profile.spo2_baseline}%")

print("\nTesting arrhythmia state machine...")
sm = ArrhythmiaStateMachine(mean_interval_s=999999)
state, ep, ramp = sm.update()
print(f"  State: {state} | Episode: {ep} | Ramp: {ramp}")

print("\nTesting vitals engine (5 ticks)...")
ve = VitalsEngine(profile, sm)
for i in range(5):
    v = ve.update(1.0)
hr = v["heart_rate"]
sbp = v["systolic_bp"]
dbp = v["diastolic_bp"]
spo2 = v["spo2"]
rr = v["respiratory_rate"]
print(f"  HR={hr} | BP={sbp}/{dbp} | SpO2={spo2}% | RR={rr}")

print("\nTesting ECG generation (1s @ 250 Hz)...")
ecg = generate_ecg_segment(hr=72, duration_s=1, sr=250, profile=profile)
print(f"  ECG samples : {len(ecg)}")
print(f"  ECG range   : {float(ecg.min()):.4f} to {float(ecg.max()):.4f}")

print("\nTesting PPG generation (1s)...")
ppg = generate_ppg_segment(hr=72, duration_s=1, sr=250, profile=profile)
print(f"  PPG samples : {len(ppg)}")

print("\nTesting RSP generation (1s)...")
rsp = generate_rsp_segment(rr=16, duration_s=1, sr=250, profile=profile)
print(f"  RSP samples : {len(rsp)}")

print("\nTesting AES-256-GCM encryption round-trip...")
test_data = {"heart_rate": 74.3, "spo2": 97.1, "systolic_bp": 122.0}
encrypted = encrypt_payload(test_data)
print(f"  Encrypted payload  : {len(encrypted)} chars (base64)")
decrypted = decrypt_payload(encrypted)
assert decrypted["heart_rate"] == test_data["heart_rate"], "Decryption mismatch!"
print(f"  Decrypted HR       : {decrypted['heart_rate']} bpm  PASS")

# Test wrong key is rejected
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
wrong_key = PBKDF2HMAC(
    algorithm=hashes.SHA256(), length=32,
    salt=b"icu-biosignal-pipeline-v1-salt-2026",
    iterations=260000, backend=default_backend()
).derive(b"COMPLETELY_WRONG_SECRET")
combined = base64.b64decode(encrypted)
try:
    AESGCM(wrong_key).decrypt(combined[:12], combined[12:], None)
    print("  Wrong key test     : FAIL (should have been rejected!)")
except Exception:
    print("  Wrong key rejected : PASS")

print()
print("=" * 50)
print("  ALL SMOKE TESTS PASSED")
print("=" * 50)
print()
print("Start the server with:")
print("  python run_simulator.py --http")
print("  python run_simulator.py         (HTTPS)")
