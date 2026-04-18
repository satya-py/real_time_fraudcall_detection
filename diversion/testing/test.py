# Cell 2: Load models with direct paths — no config file needed

import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from features import FeatureExtractor

# ── Change working directory to diversion ────────────────────
os.chdir(r"D:\d_drive_project\Realtime_fraud_call_detection\diversion")
print("Working directory:", os.getcwd())

# ── Hardcode all settings ─────────────────────────────────────
cfg = {
    "sequence_length":      301,
    "feature_dim":          39,
    "sample_rate":          16000,
    "chunk_duration":       3.0,
    "n_mfcc":               13,
    "n_fft":                512,
    "hop_length":           160,
    "n_mels":               26,
    "scam_threshold":       0.60,
    "suspicious_threshold": 0.40,
}

# ── Load models using full absolute paths ─────────────────────
seq_model  = keras.models.load_model(
    r"D:\d_drive_project\Realtime_fraud_call_detection\saved_models\sequence_model_best.keras"
)
pros_model = keras.models.load_model(
    r"D:\d_drive_project\Realtime_fraud_call_detection\saved_models\prosody_model_best.keras"
)

print(f"sequence_model : {seq_model.count_params():,} parameters")
print(f"prosody_model  : {pros_model.count_params():,} parameters")

# ── Build feature extractor ───────────────────────────────────
extractor = FeatureExtractor(
    sample_rate = cfg["sample_rate"],
    n_mfcc      = cfg["n_mfcc"],
    n_fft       = cfg["n_fft"],
    hop_length  = cfg["hop_length"],
    n_mels      = cfg["n_mels"],
)

print("Feature extractor ready")
print("Cell 2 complete ✓")