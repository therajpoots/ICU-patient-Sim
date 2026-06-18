"""
ML Models Module
Defines the models used for anomaly detection:
1. Supervised: Gradient Boosting Classifier
2. Unsupervised: Isolation Forest
3. Unsupervised: PyTorch LSTM Autoencoder
"""

import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from typing import Tuple

# --- Supervised Model ---
def create_supervised_model(n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1) -> GradientBoostingClassifier:
    """Creates a Gradient Boosting Classifier ensemble."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )

# --- Unsupervised Models ---
def create_isolation_forest(contamination: float = 0.05) -> IsolationForest:
    """Creates an Isolation Forest model."""
    return IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multi-channel anomaly detection.
    Compresses a sequence of feature vectors and reconstructs them.
    Anomalies are detected via high reconstruction error.
    """
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder: compresses seq_len x input_dim -> hidden_dim
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Decoder: reconstructs hidden_dim -> seq_len x input_dim
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. Encode
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n shape: (1, batch_size, hidden_dim)
        latent = h_n.squeeze(0)  # shape: (batch_size, hidden_dim)

        # 2. Repeat latent state seq_len times
        # Shape: (batch_size, seq_len, hidden_dim)
        decoder_in = latent.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 3. Decode
        decoder_out, _ = self.decoder_lstm(decoder_in)
        # Project to input_dim
        reconstructed = self.decoder_fc(decoder_out)
        
        return reconstructed
