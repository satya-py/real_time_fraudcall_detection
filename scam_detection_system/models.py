import torch
import torch.nn as nn
import torch.nn.functional as F

class PhonemePatternModel(nn.Module):
    """
    Submodel A: Coarse Phoneme Pattern Modeling.
    Input: MFCCs (Batch, Channels=n_mfcc, Time)
    Output: Scam Probability based on phonetic sequences.
    """
    def __init__(self, n_mfcc=13, hidden_dim=64, n_classes=1):
        super(PhonemePatternModel, self).__init__()
        # 1D Conv for local texture (phoneme-level features)
        self.conv1 = nn.Conv1d(n_mfcc, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # GRU for sequence modeling (phrase-level)
        self.gru = nn.GRU(64, hidden_dim, batch_first=True, bidirectional=False)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x shape: (Batch, n_mfcc, Time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Transpose for GRU: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)
        
        _, h_n = self.gru(x) 
        # h_n shape: (1, Batch, Hidden)
        
        out = torch.sigmoid(self.fc(h_n[-1]))
        return out

class ProsodyModel(nn.Module):
    """
    Submodel B: Urgency & Emotion Detection.
    Input: Vector of prosody features (Speech Rate, Pitch Var, Energy Var, etc.)
    """
    def __init__(self, input_dim=5, hidden_dim=32):
        super(ProsodyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class RepetitionDetector:
    """
    Submodel C: Algorithmic Repetition Detection using Self-Similarity.
    Not a learnable PyTorch model in this version, but a signal processing block.
    """
    def compute_score(self, mfccs):
        # mfccs: (n_mfcc, Time) numpy array or tensor
        if isinstance(mfccs, torch.Tensor):
            mfccs = mfccs.detach().cpu().numpy()
            
        # Compute Cosine Similarity Matrix
        # Normalize columns
        norm = np.linalg.norm(mfccs, axis=0, keepdims=True)
        mfccs_norm = mfccs / (norm + 1e-8)
        
        sim_matrix = np.dot(mfccs_norm.T, mfccs_norm)
        
        # Look for high values off-diagonal
        # Exclude main diagonal area 
        n = sim_matrix.shape[0]
        mask = np.ones((n, n)) - np.eye(n)
        # Also exclude near-diagonal (immediate temporal correlation)
        for i in range(1, 10): 
            if i < n:
                mask -= np.diag(np.ones(n-i), k=i)
                mask -= np.diag(np.ones(n-i), k=-i)
        
        masked_sim = sim_matrix * mask
        
        # Simple metric: Mean of top 5% similarity values
        # logic: if there are repeated phrases, we will see high off-diagonal blocks.
        if n == 0: return 0.0
        
        top_k = int(n * n * 0.05)
        if top_k == 0: top_k = 1
        
        flat = np.abs(masked_sim.flatten())
        top_values = np.partition(flat, -top_k)[-top_k:]
        
        repetition_score = np.mean(top_values)
        return float(repetition_score)

import numpy as np

class RiskFusionEngine:
    """
    Aggregates scores from submodels with temporal smoothing.
    """
    def __init__(self, w_phoneme=0.4, w_prosody=0.3, w_repetition=0.3):
        self.weights = np.array([w_phoneme, w_prosody, w_repetition])
        self.history = []
        self.alpha = 0.2 # Smoothing factor

    def fuse(self, scores):
        # scores: [s_phoneme, s_prosody, s_repetition]
        raw_risk = np.dot(self.weights, scores)
        
        # Temporal Smoothing (EMA)
        if not self.history:
            smooth_risk = raw_risk
        else:
            smooth_risk = self.alpha * raw_risk + (1 - self.alpha) * self.history[-1]
        
        self.history.append(smooth_risk)
        return smooth_risk

