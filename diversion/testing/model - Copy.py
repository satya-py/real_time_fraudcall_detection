# model.py
# ============================================================
# CLEAN REBUILD — Step 2 of 4
#
# PURPOSE:
#   Define both neural network models, the fusion engine,
#   and all save/load/tflite export utilities.
#
# MODELS:
#   SequenceModel  — Conv1D + GlobalAveragePooling
#                    Input:  (batch, 301, 39)
#                    Output: (batch, 1) scam probability
#
#   ProsodyModel   — 3-layer MLP
#                    Input:  (batch, 5)
#                    Output: (batch, 1) scam probability
#
#   FusionEngine   — weighted average + EMA smoothing
#                    Input:  two floats
#                    Output: one float + alert level
#
# KEY DESIGN CHOICES vs previous version:
#   GRU(64) replaced by GlobalAveragePooling  -> 25,000 fewer parameters
#   dropout 0.3 -> 0.4 on sequence model      -> less overfitting
#   .keras save format                        -> works with TF 2.20
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ================================================================
# MODEL A: SEQUENCE MODEL (Conv1D + GlobalAveragePooling)
# ================================================================

def build_sequence_model(
    sequence_length: int,
    feature_dim:     int   = 39,
    filters_1:       int   = 64,
    filters_2:       int   = 64,
    kernel_size:     int   = 3,
    dense_units:     int   = 64,
    dropout_rate:    float = 0.4,
) -> keras.Model:
    """
    Builds the sequence model for phoneme/acoustic pattern detection.

    ARCHITECTURE:
        Input(301, 39)
            |
        Conv1D(64, kernel=3, padding=same) -> shape (301, 64)
        BatchNorm -> ReLU -> Dropout(0.4)
            |
        Conv1D(64, kernel=3, padding=same) -> shape (301, 64)
        BatchNorm -> ReLU -> Dropout(0.4)
            |
        GlobalAveragePooling1D             -> shape (64,)
            Averages across all 301 time frames.
            Zero parameters.
            |
        Dense(64) -> ReLU -> Dropout(0.4)
        Dense(32) -> ReLU -> Dropout(0.2)
        Dense(1)  -> Sigmoid
            |
        Output: scam probability in [0, 1]

    INPUT:  (batch_size, sequence_length, feature_dim)
    OUTPUT: (batch_size, 1)
    """

    inputs = keras.Input(
        shape=(sequence_length, feature_dim),
        name="sequence_input"
    )

    # Conv Block 1: detects local 30ms patterns
    x = layers.Conv1D(
        filters=filters_1,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        name="conv1"
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # Conv Block 2: combines local patterns into 90ms patterns
    x = layers.Conv1D(
        filters=filters_2,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # GlobalAveragePooling: averages across all time frames
    # (None, 301, 64) -> (None, 64)
    # Zero parameters — just takes the mean over axis=1
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Dense classification head
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    x = layers.Dense(dense_units // 2, activation="relu", name="dense2")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop4")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SequenceModel")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# ================================================================
# MODEL B: PROSODY MODEL (MLP)
# ================================================================

def build_prosody_model(
    input_dim:    int   = 5,
    hidden_1:     int   = 32,
    hidden_2:     int   = 16,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    3-layer MLP for prosody-based scam detection.

    INPUT:  (batch_size, 5)
    OUTPUT: (batch_size, 1)
    """

    inputs = keras.Input(shape=(input_dim,), name="prosody_input")

    x = layers.Dense(hidden_1, use_bias=False, name="dense1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(hidden_2, activation="relu", name="dense2")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ProsodyModel")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


# ================================================================
# FUSION ENGINE
# ================================================================

class FusionEngine:
    """
    Combines sequence and prosody scores with EMA smoothing.

    WEIGHTED AVERAGE:
        raw = 0.6 * sequence_score + 0.4 * prosody_score

    EMA SMOOTHING:
        smooth[t] = 0.35 * raw[t] + 0.65 * smooth[t-1]
    """

    def __init__(
        self,
        w_sequence:           float = 0.6,
        w_prosody:            float = 0.4,
        alpha:                float = 0.35,
        scam_threshold:       float = 0.65,
        suspicious_threshold: float = 0.40,
    ):
        assert abs(w_sequence + w_prosody - 1.0) < 1e-6, \
            f"Weights must sum to 1.0"

        self.w_sequence           = w_sequence
        self.w_prosody            = w_prosody
        self.alpha                = alpha
        self.scam_threshold       = scam_threshold
        self.suspicious_threshold = suspicious_threshold
        self._ema                 = None
        self.history              = []

    def fuse(self, sequence_score: float, prosody_score: float) -> dict:
        """
        INPUT:  two floats in [0, 1]
        OUTPUT: dict with smooth_score and alert_level
        """
        raw = self.w_sequence * sequence_score + self.w_prosody * prosody_score

        if self._ema is None:
            smooth = raw
        else:
            smooth = self.alpha * raw + (1.0 - self.alpha) * self._ema

        self._ema = smooth
        self.history.append(smooth)

        if smooth >= self.scam_threshold:
            alert = "SCAM"
        elif smooth >= self.suspicious_threshold:
            alert = "SUSPICIOUS"
        else:
            alert = "LOW"

        return {
            "raw_score":    float(raw),
            "smooth_score": float(smooth),
            "alert_level":  alert,
            "seq_score":    float(sequence_score),
            "pros_score":   float(prosody_score),
        }

    def reset(self):
        self._ema    = None
        self.history = []

    def summary(self) -> dict:
        if not self.history:
            return {"status": "no_data"}
        arr = np.array(self.history)
        return {
            "total_chunks":      len(arr),
            "max_score":         float(np.max(arr)),
            "mean_score":        float(np.mean(arr)),
            "scam_chunks":       int(np.sum(arr >= self.scam_threshold)),
            "suspicious_chunks": int(np.sum(
                (arr >= self.suspicious_threshold) & (arr < self.scam_threshold)
            )),
        }


# ================================================================
# SAVE / LOAD / TFLITE
# ================================================================

def save_models(sequence_model, prosody_model, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    seq_path  = os.path.join(save_dir, "sequence_model.keras")
    pros_path = os.path.join(save_dir, "prosody_model.keras")
    sequence_model.save(seq_path)
    prosody_model.save(pros_path)
    print(f"  Sequence model -> {seq_path}")
    print(f"  Prosody model  -> {pros_path}")


def load_models(save_dir="saved_models"):
    seq_path  = os.path.join(save_dir, "sequence_model.keras")
    pros_path = os.path.join(save_dir, "prosody_model.keras")
    for path in [seq_path, pros_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n  Model not found: {path}"
                f"\n  Run:  python train.py"
            )
    seq_model  = keras.models.load_model(seq_path)
    pros_model = keras.models.load_model(pros_path)
    print(f"  Loaded: {seq_path}")
    print(f"  Loaded: {pros_path}")
    return seq_model, pros_model


def export_tflite(save_dir="saved_models", tflite_dir="tflite_models"):
    """
    Converts .keras models to TFLite for Android.
    Two-step process required for TF 2.20 / Keras 3.x:
        1. model.export() -> SavedModel folder
        2. TFLiteConverter.from_saved_model() -> .tflite bytes
    """
    os.makedirs(tflite_dir, exist_ok=True)
    seq_model, pros_model = load_models(save_dir)

    for name, model in [
        ("sequence_model", seq_model),
        ("prosody_model",  pros_model),
    ]:
        export_path = os.path.join(save_dir, f"{name}_savedmodel")
        model.export(export_path)

        converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()

        out_path = os.path.join(tflite_dir, f"{name}.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)

        kb = os.path.getsize(out_path) / 1024
        print(f"  {name}.tflite  ({kb:.1f} KB)  ->  {out_path}")


# ================================================================
# SELF-TEST:  python model.py
# ================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("model.py -- Self Test")
    print("=" * 50)

    SEQ_LEN = 301
    FEAT_DIM = 39
    BATCH = 4

    # Test SequenceModel
    print("\nTest 1: SequenceModel")
    seq_model = build_sequence_model(sequence_length=SEQ_LEN, feature_dim=FEAT_DIM)
    seq_model.summary()
    fake = np.random.randn(BATCH, SEQ_LEN, FEAT_DIM).astype(np.float32)
    out  = seq_model.predict(fake, verbose=0)
    print(f"  Output shape: {out.shape}   range [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (BATCH, 1)
    assert 0 <= out.min() and out.max() <= 1
    print("  PASSED")

    # Test ProsodyModel
    print("\nTest 2: ProsodyModel")
    pros_model = build_prosody_model()
    pros_model.summary()
    fake2 = np.random.rand(BATCH, 5).astype(np.float32)
    out2  = pros_model.predict(fake2, verbose=0)
    print(f"  Output shape: {out2.shape}   range [{out2.min():.3f}, {out2.max():.3f}]")
    assert out2.shape == (BATCH, 1)
    print("  PASSED")

    # Test FusionEngine
    print("\nTest 3: FusionEngine")
    fusion = FusionEngine()
    scores = [(0.3,0.2),(0.5,0.4),(0.7,0.6),(0.8,0.7),(0.9,0.8)]
    print(f"  {'Frame':5s} {'Seq':5s} {'Pros':5s} {'Raw':5s} {'Smooth':6s} Alert")
    for i,(s,p) in enumerate(scores):
        r = fusion.fuse(s, p)
        print(f"  {i+1:3d}   {s:.2f}  {p:.2f}  "
              f"{r['raw_score']:.3f} {r['smooth_score']:.3f}  {r['alert_level']}")
    assert fusion.history[-1] > fusion.history[0]
    fusion.reset()
    assert fusion._ema is None
    print("  PASSED")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED -- model.py is ready")
    print("=" * 50)
    print(f"\nSequenceModel params: {seq_model.count_params():,}")
    print(f"ProsodyModel params:  {pros_model.count_params():,}")