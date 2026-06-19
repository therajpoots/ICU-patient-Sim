"""
Encrypted Local Database Module
Provides SQLite database integration with column/row-level AES-256-GCM encryption
for anomaly logs, vitals, and waveforms.
"""

import os
import sqlite3
import base64
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# Constants for KDF (matching API crypto for consistency)
_KDF_SALT = b"icu-biosignal-pipeline-v1-salt-2026"
_KEY_ITERATIONS = 260_000

def derive_db_key(password: str) -> bytes:
    """Derive 256-bit AES key from user-provided password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_KDF_SALT,
        iterations=_KEY_ITERATIONS,
        backend=default_backend(),
    )
    return kdf.derive(password.encode("utf-8"))

def encrypt_value(data: Any, key: bytes) -> str:
    """Serialize data to JSON and encrypt using AES-256-GCM, returning base64 string."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        if type(obj).__name__ in ('bool_', 'bool'):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    plaintext = json.dumps(data, default=convert_numpy).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    combined = nonce + ciphertext
    return base64.b64encode(combined).decode("ascii")

def decrypt_value(encrypted_b64: str, key: bytes) -> Any:
    """Decrypt a base64-encoded AES-256-GCM string, returning Python object."""
    aesgcm = AESGCM(key)
    try:
        combined = base64.b64decode(encrypted_b64)
        nonce = combined[:12]
        ciphertext = combined[12:]
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Decryption failed: {exc}")

class EncryptedDatabase:
    """
    Manages local SQLite database for anomaly logging.
    All sensitive information is encrypted in SQLite via AES-GCM at the row level.
    """
    def __init__(self, db_path: str = "data/anomaly_logs.db"):
        self.db_path = db_path
        # Create data directory if not exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                state_enc TEXT,
                vitals_enc TEXT,
                waveforms_enc TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_anomaly(
        self,
        password: str,
        start_time: float,
        end_time: float,
        state: str,
        vitals: Dict[str, Any],
        waveforms: Dict[str, List[float]]
    ) -> int:
        """
        Encrypts state, vitals, and waveforms, and saves them to the SQLite database.
        Returns the inserted log ID.
        """
        key = derive_db_key(password)
        state_enc = encrypt_value(state, key)
        vitals_enc = encrypt_value(vitals, key)
        waveforms_enc = encrypt_value(waveforms, key)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO anomaly_logs (start_time, end_time, state_enc, vitals_enc, waveforms_enc)
            VALUES (?, ?, ?, ?, ?)
        """, (start_time, end_time, state_enc, vitals_enc, waveforms_enc))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def get_anomaly_logs(
        self,
        password: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries and decrypts anomaly records from the database.
        Raises ValueError if the password fails to decrypt the records.
        """
        key = derive_db_key(password)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id, start_time, end_time, state_enc, vitals_enc, waveforms_enc FROM anomaly_logs"
        params = []
        
        conditions = []
        if start_time is not None:
            conditions.append("start_time >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("end_time <= ?")
            params.append(end_time)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY start_time DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        decrypted_logs = []
        for r in rows:
            row_id, s_time, e_time, s_enc, v_enc, w_enc = r
            try:
                state = decrypt_value(s_enc, key)
                if state == "stable":
                    continue
                vitals = decrypt_value(v_enc, key)
                waveforms = decrypt_value(w_enc, key)
                decrypted_logs.append({
                    "id": row_id,
                    "start_time": s_time,
                    "end_time": e_time,
                    "state": state,
                    "vitals": vitals,
                    "waveforms": waveforms
                })
            except Exception as exc:
                print(f"WARNING: Skipping log ID {row_id} due to decryption failure (possibly incorrect password or corrupted data): {exc}")
                
        return decrypted_logs

