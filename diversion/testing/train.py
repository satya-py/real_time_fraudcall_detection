# # train.py
# # ============================================================
# # CLEAN REBUILD — Step 3 of 4
# #
# # PURPOSE:
# #   Load your dataset, extract features, train both models,
# #   save everything needed for check_file.py to work.
# #
# # YOUR DATASET:
# #   processed_dataset/SCAM_CALLS/   <- 33 .wav files (~89s each)
# #   processed_dataset/NORMAL_CALLS/ <- 33 .wav files (~66s each)
# #
# # HOW TO RUN:
# #   cd diversion
# #   python train.py
# #
# # OUTPUT:
# #   saved_models/sequence_model.keras
# #   saved_models/prosody_model.keras
# #   tflite_models/sequence_model.tflite
# #   tflite_models/prosody_model.tflite
# #   training_config.json
# # ============================================================

# import os
# import json
# import random
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow import keras
# from pathlib import Path
# from sklearn.utils.class_weight import compute_class_weight
# import warnings
# warnings.filterwarnings("ignore")

# from features import FeatureExtractor
# from model import (
#     build_sequence_model,
#     build_prosody_model,
#     save_models,
#     export_tflite,
# )


# # ================================================================
# # CONFIGURATION
# # All values locked to your exact setup.
# # Do not change these unless you change your dataset.
# # ================================================================

# CONFIG = {
#     # Dataset
#     "dataset_dir":    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed_dataset"),  
#     "scam_subdir":    "SCAM_CALLS",
#     "normal_subdir":  "NORMAL_CALLS",

#     # Audio
#     "sample_rate":         16000,
#     "chunk_duration":      3.0,       # seconds per training sample
#     "min_chunk_duration":  1.5,       # discard chunks shorter than this

#     # Features — must match what features.py produces
#     "n_mfcc":      13,
#     "n_fft":       512,
#     "hop_length":  160,
#     "n_mels":      26,
#     "feature_dim": 39,    # 13 mfcc + 13 delta + 13 delta-delta
#     "sequence_length": 301,  # frames per 3s chunk (your librosa version)

#     # Training
#     "val_split":               0.2,   # 20% of FILES go to validation
#     "random_seed":             42,
#     "batch_size":              16,    # small batch = stable gradients, less memory
#     "epochs":                  80,    # early stopping will stop before this
#     "learning_rate":           1e-3,
#     "early_stopping_patience": 12,    # wait 12 epochs before stopping

#     # Augmentation
#     # We add Gaussian noise copies of NORMAL chunks only
#     # to compensate for scam having more total audio (48.9 vs 36.2 min)
#     "augment_normal":  True,
#     "noise_factor":    0.005,   # noise amplitude (0.5% of max signal)

#     # Thresholds for alert levels
#     "scam_threshold":       0.65,
#     "suspicious_threshold": 0.40,

#     # Output paths
#     "save_dir":    "saved_models",
#     "tflite_dir":  "tflite_models",
#     "config_path": "training_config.json",
# }


# # ================================================================
# # STEP 1: DISCOVER FILES
# # ================================================================

# def discover_files(config: dict) -> dict:
#     """
#     Scans SCAM_CALLS/ and NORMAL_CALLS/ for .wav files.

#     Returns:
#         {
#             'scam':   [Path('SCAM_CALLS/scam_1.wav'), ...],
#             'normal': [Path('NORMAL_CALLS/normal_1.wav'), ...]
#         }

#     Files are sorted so every run uses the same order.
#     Consistent order + fixed random seed = reproducible splits.
#     """
#     dataset_path = Path(config["dataset_dir"])
#     files = {"scam": [], "normal": []}

#     for class_name, subdir in [
#         ("scam",   config["scam_subdir"]),
#         ("normal", config["normal_subdir"]),
#     ]:
#         class_dir = dataset_path / subdir

#         if not class_dir.exists():
#             raise FileNotFoundError(
#                 f"\n  Folder not found: {class_dir}"
#                 f"\n  Make sure you are running from inside 'diversion/':"
#                 f"\n    cd diversion"
#                 f"\n    python train.py"
#                 f"\n  Current directory: {os.getcwd()}"
#             )

#         # sorted() ensures same order every run regardless of OS file system
#         for fp in sorted(class_dir.glob("*.wav")):
#             files[class_name].append(fp)

#     print(f"  SCAM   files: {len(files['scam'])}")
#     print(f"  NORMAL files: {len(files['normal'])}")

#     if len(files["scam"]) == 0 or len(files["normal"]) == 0:
#         raise ValueError("No .wav files found. Check folder paths.")

#     return files


# # ================================================================
# # STEP 2: FILE-LEVEL TRAIN / VAL SPLIT
# # ================================================================

# def split_files(files: dict, val_split: float, seed: int) -> dict:
#     """
#     Splits at the FILE level — not the sample level.

#     WHY THIS MATTERS:
#         If chunks from the same file appear in both train and val,
#         the model memorises that specific caller's voice.
#         File-level split ensures the model is tested on callers
#         it has never heard during training.

#     With 33 files per class and val_split=0.2:
#         Train: 27 files per class
#         Val:    6 files per class

#     Returns:
#         {
#             'train': {'scam': [Path,...], 'normal': [Path,...]},
#             'val':   {'scam': [Path,...], 'normal': [Path,...]}
#         }
#     """
#     splits = {"train": {}, "val": {}}

#     for class_name, file_list in files.items():
#         # Shuffle with fixed seed for reproducibility
#         rng = random.Random(seed)
#         shuffled = file_list.copy()
#         rng.shuffle(shuffled)

#         # Ensure at least 1 file in validation
#         n_val = max(1, int(len(shuffled) * val_split))

#         splits["val"][class_name]   = shuffled[:n_val]
#         splits["train"][class_name] = shuffled[n_val:]

#         print(f"  {class_name:6s}: {len(splits['train'][class_name]):2d} train "
#               f"| {len(splits['val'][class_name]):2d} val")

#     return splits


# # ================================================================
# # STEP 3: LOAD ONE FILE AND CUT INTO CHUNKS
# # ================================================================

# def load_and_chunk(
#     file_path:          Path,
#     sample_rate:        int,
#     chunk_duration:     float,
#     min_chunk_duration: float,
# ) -> list:
#     """
#     Loads one .wav file and returns a list of fixed-length chunks.

#     CHUNKING EXAMPLE:
#         chunk_duration  = 3.0s
#         sample_rate     = 16000
#         chunk_samples   = 3.0 * 16000 = 48000

#         scam_1.wav = 61.8s = 988,800 samples
#         chunk 0: samples[0      : 48000 ]  -> 3.0s  kept
#         chunk 1: samples[48000  : 96000 ]  -> 3.0s  kept
#         ...
#         chunk 19: samples[912000: 960000]  -> 3.0s  kept
#         chunk 20: samples[960000: 988800]  -> 1.8s  >= min(1.5s) -> pad to 3s kept

#     PADDING:
#         Last chunk padded with zeros on the right if >= min_chunk_duration.
#         Zeros = silence — the model learns to ignore silence.

#     Returns: list of numpy float32 arrays, each of length chunk_samples
#     """
#     chunk_samples = int(chunk_duration     * sample_rate)
#     min_samples   = int(min_chunk_duration * sample_rate)

#     try:
#         # librosa.load handles all of:
#         #   - WAV decoding
#         #   - Stereo to mono conversion (averages channels)
#         #   - Resampling to target sample_rate
#         audio, _ = librosa.load(
#             str(file_path),
#             sr=sample_rate,
#             mono=True,
#             res_type="kaiser_fast",   # faster resampling algorithm
#         )
#         # audio.dtype = float32, values in [-1.0, 1.0]

#     except Exception as e:
#         print(f"    WARNING: Could not load {file_path.name}: {e}")
#         return []

#     # Discard files that are too short even for one chunk
#     if len(audio) < min_samples:
#         print(f"    WARNING: {file_path.name} too short ({len(audio)/sample_rate:.1f}s), skipping")
#         return []

#     chunks = []
#     start  = 0

#     while start + min_samples <= len(audio):
#         end   = start + chunk_samples
#         chunk = audio[start:end].copy()

#         if len(chunk) < chunk_samples:
#             # Last chunk: shorter than chunk_duration
#             # Pad right side with zeros (silence)
#             # np.pad(array, (left_pad, right_pad))
#             chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

#         chunks.append(chunk.astype(np.float32))
#         start += chunk_samples  # non-overlapping: advance by full chunk

#     return chunks


# # ================================================================
# # STEP 4: AUGMENTATION
# # ================================================================

# def add_noise(audio: np.ndarray, noise_factor: float) -> np.ndarray:
#     """
#     Creates one augmented copy by adding Gaussian noise.

#     FORMULA:
#         augmented[i] = audio[i] + noise_factor * N(0,1)
#         noise_factor = 0.005 -> noise amplitude = 0.5% of max signal

#     WHY THIS HELPS:
#         Real phone calls have varying background noise.
#         Training on both clean and noisy versions makes the model
#         robust to noise without changing the learned speech patterns.

#     np.clip ensures output stays in valid audio range [-1, 1].
#     """
#     noise = np.random.randn(len(audio)).astype(np.float32)
#     return np.clip(audio + noise_factor * noise, -1.0, 1.0)


# # ================================================================
# # STEP 5: BUILD FEATURE ARRAYS FROM ALL FILES
# # ================================================================

# def build_arrays(
#     file_splits: dict,
#     config:      dict,
#     split_name:  str,    # "train" or "val"
# ) -> tuple:
#     """
#     Processes every file in the split, extracts features,
#     and returns ready-to-use numpy arrays.

#     AUGMENTATION LOGIC:
#         Training split, normal class only:
#             Original chunk  -> extract features -> label 0
#             Noisy copy      -> extract features -> label 0
#         This adds ~594 extra normal samples to compensate
#         for scam having more total audio.

#         Validation split: NO augmentation ever.
#         We want validation to reflect real-world conditions.

#     Returns:
#         X_seq  -> shape (N, 301, 39)  float32
#         X_pros -> shape (N, 5)        float32
#         y      -> shape (N,)          float32  (0.0 or 1.0)

#     N = total number of chunks across all files in this split.
#     """
#     extractor = FeatureExtractor(
#         sample_rate=config["sample_rate"],
#         n_mfcc=config["n_mfcc"],
#         n_fft=config["n_fft"],
#         hop_length=config["hop_length"],
#         n_mels=config["n_mels"],
#     )

#     seq_len  = config["sequence_length"]   # 301
#     feat_dim = config["feature_dim"]       # 39

#     X_seq_list  = []
#     X_pros_list = []
#     y_list      = []

#     is_train = (split_name == "train")

#     for class_name, file_list in file_splits[split_name].items():
#         label = 1.0 if class_name == "scam" else 0.0

#         print(f"    [{split_name}] {class_name:6s}: {len(file_list)} files ...")

#         for file_path in file_list:
#             chunks = load_and_chunk(
#                 file_path,
#                 config["sample_rate"],
#                 config["chunk_duration"],
#                 config["min_chunk_duration"],
#             )

#             if not chunks:
#                 continue

#             # Build list of chunks to process for this file
#             # Start with original chunks
#             to_process = list(chunks)

#             # Augmentation: training split, normal class only
#             if is_train and config["augment_normal"] and class_name == "normal":
#                 for chunk in chunks:
#                     # One noisy copy per original chunk
#                     to_process.append(
#                         add_noise(chunk, config["noise_factor"])
#                     )
#                 # to_process now has 2x the original chunks for normal files

#             # Extract features from every chunk
#             for chunk in to_process:
#                 features = extractor.extract_all(chunk)

#                 if not features["valid"]:
#                     continue

#                 combined = features["combined"]   # shape (T, 39)
#                 prosody  = features["prosody"]    # shape (5,)

#                 # ---- Pad or truncate to exact sequence_length ----
#                 # This is essential: the model expects exactly 301 frames.
#                 # Even though all 3s chunks should give 301 frames,
#                 # padded end-of-file chunks might give fewer.
#                 T = combined.shape[0]

#                 if T < seq_len:
#                     # Pad with zeros at the bottom
#                     # np.pad(array, ((top_rows, bottom_rows), (left_cols, right_cols)))
#                     pad = seq_len - T
#                     combined = np.pad(combined, ((0, pad), (0, 0)))

#                 elif T > seq_len:
#                     # Truncate to first seq_len rows
#                     combined = combined[:seq_len, :]

#                 # combined.shape is now exactly (301, 39)

#                 X_seq_list.append(combined)
#                 X_pros_list.append(prosody)
#                 y_list.append(label)

#     # Convert lists to numpy arrays
#     X_seq  = np.array(X_seq_list,  dtype=np.float32)
#     # shape: (N, 301, 39)

#     X_pros = np.array(X_pros_list, dtype=np.float32)
#     # shape: (N, 5)

#     y      = np.array(y_list,      dtype=np.float32)
#     # shape: (N,)  values are 0.0 or 1.0

#     scam_n   = int(np.sum(y))
#     normal_n = int(np.sum(1 - y))
#     print(f"    {split_name} total: {len(y)} samples "
#           f"({scam_n} scam, {normal_n} normal)")

#     return X_seq, X_pros, y


# # ================================================================
# # STEP 6: CLASS WEIGHTS
# # ================================================================

# def get_class_weights(y: np.ndarray) -> dict:
#     """
#     Computes per-class loss weights.

#     After augmentation:
#         scam:   ~800 samples  (no augmentation)
#         normal: ~1188 samples (augmented with noise copies)

#     Formula: weight[c] = total / (n_classes * count[c])
#     Example:
#         total = 1988, scam = 800, normal = 1188
#         weight[scam=1]   = 1988 / (2 * 800)  = 1.24
#         weight[normal=0] = 1988 / (2 * 1188) = 0.84

#     Effect: the model is penalised 1.24x more for missing a scam call
#     than for a false positive on a normal call.
#     For a fraud detector, missing a scam is worse than a false alarm.
#     """
#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.array([0, 1]),
#         y=y,
#     )
#     d = {0: float(weights[0]), 1: float(weights[1])}
#     print(f"    Class weights: normal={d[0]:.3f}  scam={d[1]:.3f}")
#     return d


# # ================================================================
# # STEP 7: CALLBACKS
# # ================================================================

# def get_callbacks(model_name: str, save_dir: str, patience: int) -> list:
#     """
#     Three callbacks that make training smarter:

#     1. ModelCheckpoint:
#        Saves model ONLY when val_auc improves.
#        'save_best_only=True' means we always keep the best weights,
#        not the last weights (which may have started overfitting).

#     2. EarlyStopping:
#        Stops training when val_loss has not improved for `patience` epochs.
#        'restore_best_weights=True' rolls back to the saved best checkpoint.
#        With patience=12, the model gets 12 chances after the best epoch
#        before training stops.

#     3. ReduceLROnPlateau:
#        When val_loss stops improving for 5 epochs, multiply LR by 0.5.
#        Smaller learning rate = smaller steps = better chance of finding
#        a good minimum instead of oscillating around it.
#        min_lr=1e-6 prevents the LR from becoming uselessly tiny.
#     """
#     best_path = os.path.join(save_dir, f"{model_name}_best.keras")

#     return [
#         keras.callbacks.ModelCheckpoint(
#             filepath=best_path,
#             monitor="val_auc",
#             mode="max",
#             save_best_only=True,
#             verbose=1,
#         ),
#         keras.callbacks.EarlyStopping(
#             monitor="val_loss",
#             patience=patience,
#             restore_best_weights=True,
#             verbose=1,
#         ),
#         keras.callbacks.ReduceLROnPlateau(
#             monitor="val_loss",
#             factor=0.5,
#             patience=5,
#             min_lr=1e-6,
#             verbose=1,
#         ),
#     ]


# # ================================================================
# # MAIN TRAINING PIPELINE
# # ================================================================

# def train():

#     print("=" * 60)
#     print("  SCAM DETECTION — CLEAN REBUILD TRAINING")
#     print(f"  TensorFlow: {tf.__version__}")
#     print(f"  Directory:  {os.getcwd()}")
#     print("=" * 60)

#     # Fix all random seeds for reproducibility
#     # Same seed = same train/val split = same results every run
#     np.random.seed(CONFIG["random_seed"])
#     tf.random.set_seed(CONFIG["random_seed"])
#     random.seed(CONFIG["random_seed"])

#     os.makedirs(CONFIG["save_dir"],   exist_ok=True)
#     os.makedirs(CONFIG["tflite_dir"], exist_ok=True)

#     # ----------------------------------------------------------
#     # 1. Discover files
#     # ----------------------------------------------------------
#     print("\n[1/8] Scanning dataset...")
#     files = discover_files(CONFIG)

#     # ----------------------------------------------------------
#     # 2. File-level split
#     # ----------------------------------------------------------
#     print("\n[2/8] File-level train/val split...")
#     splits = split_files(files, CONFIG["val_split"], CONFIG["random_seed"])

#     # ----------------------------------------------------------
#     # 3. Build training arrays
#     # ----------------------------------------------------------
#     print("\n[3/8] Extracting TRAINING features...")
#     print("      (takes 3-8 minutes on CPU — one time only)")
#     X_seq_tr, X_pros_tr, y_tr = build_arrays(splits, CONFIG, "train")

#     # ----------------------------------------------------------
#     # 4. Build validation arrays
#     # ----------------------------------------------------------
#     print("\n[4/8] Extracting VALIDATION features...")
#     X_seq_val, X_pros_val, y_val = build_arrays(splits, CONFIG, "val")

#     # Print shapes for verification
#     print(f"\n  X_seq_tr  shape: {X_seq_tr.shape}")
#     print(f"  X_pros_tr shape: {X_pros_tr.shape}")
#     print(f"  y_tr      shape: {y_tr.shape}")
#     print(f"  X_seq_val shape: {X_seq_val.shape}")

#     # Safety check
#     if X_seq_tr.shape[0] < 4:
#         raise ValueError(
#             "Too few training samples. "
#             "Check that your audio files are longer than 1.5 seconds."
#         )

#     # Verify sequence length matches config
#     actual_seq_len = X_seq_tr.shape[1]
#     if actual_seq_len != CONFIG["sequence_length"]:
#         print(f"\n  WARNING: sequence length mismatch!")
#         print(f"  Config says: {CONFIG['sequence_length']}")
#         print(f"  Data gives:  {actual_seq_len}")
#         print(f"  Updating config to {actual_seq_len}")
#         CONFIG["sequence_length"] = actual_seq_len

#     # ----------------------------------------------------------
#     # 5. Class weights
#     # ----------------------------------------------------------
#     print("\n[5/8] Computing class weights...")
#     class_weights = get_class_weights(y_tr)

#     # ----------------------------------------------------------
#     # 6. Train Sequence Model
#     # ----------------------------------------------------------
#     print("\n[6/8] Training Sequence Model (Conv1D + GlobalAvgPool)...")
#     print(f"      Input shape: (batch, {CONFIG['sequence_length']}, {CONFIG['feature_dim']})")

#     seq_model = build_sequence_model(
#         sequence_length=CONFIG["sequence_length"],
#         feature_dim=CONFIG["feature_dim"],
#     )
#     seq_model.summary()

#     seq_model.fit(
#         x=X_seq_tr,
#         y=y_tr,
#         validation_data=(X_seq_val, y_val),
#         # validation_data: evaluated after every epoch
#         # These files were NEVER seen during training

#         batch_size=CONFIG["batch_size"],
#         # batch_size=16: compute gradient on 16 samples at a time
#         # Smaller batch = noisier but more frequent gradient updates
#         # Good for small datasets

#         epochs=CONFIG["epochs"],
#         # Maximum epochs — EarlyStopping will likely stop much sooner

#         class_weight=class_weights,
#         # Multiply loss by class_weight[label] for each sample
#         # Scam samples have higher weight -> model focuses on not missing scams

#         callbacks=get_callbacks(
#             "sequence_model",
#             CONFIG["save_dir"],
#             CONFIG["early_stopping_patience"],
#         ),
#         verbose=1,
#     )

#     # ----------------------------------------------------------
#     # 7. Train Prosody Model
#     # ----------------------------------------------------------
#     print("\n[7/8] Training Prosody Model (MLP)...")
#     print(f"      Input shape: (batch, 5)")

#     pros_model = build_prosody_model(input_dim=5)
#     pros_model.summary()

#     pros_model.fit(
#         x=X_pros_tr,
#         y=y_tr,
#         validation_data=(X_pros_val, y_val),
#         batch_size=CONFIG["batch_size"],
#         epochs=CONFIG["epochs"],
#         class_weight=class_weights,
#         callbacks=get_callbacks(
#             "prosody_model",
#             CONFIG["save_dir"],
#             CONFIG["early_stopping_patience"],
#         ),
#         verbose=1,
#     )

#     # ----------------------------------------------------------
#     # 8. Save models + export TFLite + save config
#     # ----------------------------------------------------------
#     print("\n[8/8] Saving models...")
#     save_models(seq_model, pros_model, CONFIG["save_dir"])

#     print("\n[+] Exporting TFLite models for Android...")
#     try:
#         export_tflite(CONFIG["save_dir"], CONFIG["tflite_dir"])
#     except Exception as e:
#         print(f"  WARNING: TFLite export failed: {e}")
#         print("  Keras models are still saved and check_file.py will work.")

#     # Save training config
#     # check_file.py reads this to know sequence_length, thresholds, etc.
#     training_config = {
#         "sequence_length":      CONFIG["sequence_length"],
#         "feature_dim":          CONFIG["feature_dim"],
#         "n_mfcc":               CONFIG["n_mfcc"],
#         "n_fft":                CONFIG["n_fft"],
#         "hop_length":           CONFIG["hop_length"],
#         "n_mels":               CONFIG["n_mels"],
#         "sample_rate":          CONFIG["sample_rate"],
#         "chunk_duration":       CONFIG["chunk_duration"],
#         "scam_threshold":       CONFIG["scam_threshold"],
#         "suspicious_threshold": CONFIG["suspicious_threshold"],
#     }

#     with open(CONFIG["config_path"], "w") as f:
#         json.dump(training_config, f, indent=2)

#     print(f"\n[+] training_config.json saved")
#     print(f"    sequence_length = {CONFIG['sequence_length']}")

#     # ----------------------------------------------------------
#     # Final evaluation on validation set
#     # ----------------------------------------------------------
#     print("\n" + "=" * 60)
#     print("  TRAINING COMPLETE")
#     print("=" * 60)

#     print("\nSequence Model — Validation:")
#     seq_model.evaluate(X_seq_val, y_val, verbose=1)

#     print("\nProsody Model — Validation:")
#     pros_model.evaluate(X_pros_val, y_val, verbose=1)

#     print("\nFiles saved:")
#     print(f"  {CONFIG['save_dir']}/sequence_model.keras")
#     print(f"  {CONFIG['save_dir']}/prosody_model.keras")
#     print(f"  {CONFIG['tflite_dir']}/sequence_model.tflite")
#     print(f"  {CONFIG['tflite_dir']}/prosody_model.tflite")
#     print(f"  {CONFIG['config_path']}")

#     print("\nNext step:")
#     print("  python check_file.py path/to/audio.wav")

#     return seq_model, pros_model, training_config


# # ================================================================
# # ENTRY POINT
# # ================================================================

# if __name__ == "__main__":

#     # Warn if not in the right directory
#     cwd = Path(os.getcwd())
#     if not (cwd / "processed_dataset").exists():
#         print("=" * 60)
#         print("  WARNING: 'processed_dataset' folder not found here.")
#         print(f"  Current directory: {cwd}")
#         print("  Please run from inside the 'diversion' folder:")
#         print("  cd diversion")
#         print("  python train.py")
#         print("=" * 60)
#         print()

#     train()

# train.py
# ============================================================
# COMBINED TRAINING — Call Recordings + Wake Word Clips
#
# DATASETS USED:
#   1. processed_dataset/SCAM_CALLS/    33 files, 49-200s each
#      processed_dataset/NORMAL_CALLS/  33 files, 31-178s each
#      -> chunked into 3-second segments
#
#   2. wake_words/SCAM/    16 wav files, 2.7-4.2s each
#      wake_words/NORMAL/  16 wav files, 2.2-4.3s each
#      -> used directly as-is (already ~3 seconds)
#
# WHY COMBINING HELPS:
#   Call recordings: "what does a scam CALL sound like?"
#   Wake word clips: "what do scam PHRASES sound like?"
#   Together the model learns both call-level and phrase-level patterns.
#
# HOW TO RUN:
#   cd D:\d_drive_project\Realtime_fraud_call_detection\diversion
#   python train.py
# ============================================================

import os
import json
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

from features import FeatureExtractor
from model import (
    build_sequence_model,
    build_prosody_model,
    save_models,
    export_tflite,
)


# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    # Dataset 1: Full call recordings
    "calls_dir":           "processed_dataset",
    "calls_scam_subdir":   "SCAM_CALLS",
    "calls_normal_subdir": "NORMAL_CALLS",

    # Dataset 2: Wake word clips (absolute path)
    "wake_dir":            "D:/d_drive_project/Realtime_fraud_call_detection/wake_words/wake_words",
    "wake_scam_subdir":    "SCAM",
    "wake_normal_subdir":  "NORMAL",

    # Audio
    "sample_rate":          16000,
    "chunk_duration":       3.0,
    "min_chunk_duration":   1.5,

    # Features
    "n_mfcc":       13,
    "n_fft":        512,
    "hop_length":   160,
    "n_mels":       26,
    "feature_dim":  39,
    "sequence_length": 301,

    # Training
    "val_split":               0.2,
    "random_seed":             42,
    "batch_size":              16,
    "epochs":                  80,
    "learning_rate":           1e-3,
    "early_stopping_patience": 12,

    # Augmentation
    "augment":                 True,
    "noise_factor":            0.005,
    "wake_augment_multiplier": 5,    # 5 copies per wake word clip (only 16 files)
    "calls_augment_normal":    True, # noise copies of normal call chunks

    # Thresholds
    "scam_threshold":       0.60,   # lowered from 0.65 for better sensitivity
    "suspicious_threshold": 0.40,

    # Output
    "save_dir":    r"D:\\d_drive_project\\Realtime_fraud_call_detection\\saved_models",
    "tflite_dir":  r"D:\d_drive_project\Realtime_fraud_call_detection\tflite_models",
    "config_path": "training_config.json",
}


# ================================================================
# AUGMENTATION
# ================================================================

def augment_audio(audio: np.ndarray, aug_type: int, noise_factor: float) -> np.ndarray:
    """
    4 augmentation types:
        0: Gaussian noise      - simulates phone line noise
        1: Time shift          - simulates different phrase start points
        2: Amplitude scaling   - simulates different microphone volumes
        3: Noise + scale       - combined, most aggressive

    All types teach the model to ignore recording-specific variation
    and focus on the actual speech patterns instead.
    """
    audio = audio.copy()
    if aug_type == 0:
        audio = audio + noise_factor * np.random.randn(len(audio)).astype(np.float32)
    elif aug_type == 1:
        audio = np.roll(audio, random.randint(0, 800))
    elif aug_type == 2:
        audio = audio * random.uniform(0.7, 1.3)
    elif aug_type == 3:
        audio = audio + noise_factor * np.random.randn(len(audio)).astype(np.float32)
        audio = audio * random.uniform(0.8, 1.2)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ================================================================
# UTILITIES
# ================================================================

def load_audio_file(file_path: Path, sample_rate: int):
    try:
        audio, _ = librosa.load(str(file_path), sr=sample_rate,
                                mono=True, res_type="kaiser_fast")
        return audio.astype(np.float32)
    except Exception as e:
        print(f"    WARNING: Cannot load {file_path.name}: {e}")
        return None


def pad_or_truncate(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) < target:
        return np.pad(audio, (0, target - len(audio)))
    return audio[:target]


def chunk_audio(audio, sample_rate, chunk_duration, min_chunk_duration):
    chunk_samples = int(chunk_duration * sample_rate)
    min_samples   = int(min_chunk_duration * sample_rate)
    chunks = []
    start  = 0
    while start + min_samples <= len(audio):
        chunk = audio[start:start + chunk_samples].copy()
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
        start += chunk_samples
    return chunks


def extract_features(chunk, extractor, seq_len, feat_dim):
    features = extractor.extract_all(chunk)
    if not features["valid"]:
        return np.zeros((seq_len, feat_dim), dtype=np.float32), np.zeros(5, dtype=np.float32), False
    combined = features["combined"]
    T = combined.shape[0]
    if T < seq_len:
        combined = np.pad(combined, ((0, seq_len - T), (0, 0)))
    elif T > seq_len:
        combined = combined[:seq_len, :]
    return combined, features["prosody"], True


def make_extractor(config):
    return FeatureExtractor(
        sample_rate=config["sample_rate"],
        n_mfcc=config["n_mfcc"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )


# ================================================================
# FILE-LEVEL SPLITS
# ================================================================

def split_files_for(file_dict, val_split, seed):
    rng = random.Random(seed)
    splits = {"train": {}, "val": {}}
    for class_name, file_list in file_dict.items():
        shuffled = file_list.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        splits["val"][class_name]   = shuffled[:n_val]
        splits["train"][class_name] = shuffled[n_val:]
        print(f"    {class_name}: {len(splits['train'][class_name])} train "
              f"| {len(splits['val'][class_name])} val")
    return splits


def split_call_files(config):
    calls_path = Path(config["calls_dir"])
    files = {}
    for class_name, subdir in [("scam", config["calls_scam_subdir"]),
                                ("normal", config["calls_normal_subdir"])]:
        class_dir = calls_path / subdir
        if not class_dir.exists():
            raise FileNotFoundError(
                f"\n  Not found: {class_dir}"
                f"\n  Run from inside diversion/ folder"
            )
        files[class_name] = sorted(class_dir.glob("*.wav"))
    print(f"  Call recordings: {len(files['scam'])} scam, {len(files['normal'])} normal")
    return split_files_for(files, config["val_split"], config["random_seed"])


def split_wake_files(config):
    wake_path = Path(config["wake_dir"])
    files = {}
    for class_name, subdir in [("scam", config["wake_scam_subdir"]),
                                ("normal", config["wake_normal_subdir"])]:
        class_dir = wake_path / subdir
        if not class_dir.exists():
            raise FileNotFoundError(f"\n  Not found: {class_dir}")
        wav_files = sorted(class_dir.glob("*.wav"))
        files[class_name] = wav_files
        print(f"  Wake {class_name}: {len(wav_files)} wav (AAC skipped)")
    return split_files_for(files, config["val_split"], config["random_seed"] + 1)


# ================================================================
# LOAD CALL RECORDINGS
# ================================================================

def load_call_recordings(config, split, call_splits):
    extractor = make_extractor(config)
    seq_len   = config["sequence_length"]
    feat_dim  = config["feature_dim"]
    is_train  = (split == "train")
    seq_list, pros_list, lbl_list = [], [], []

    for class_name, file_list in call_splits[split].items():
        label = 1.0 if class_name == "scam" else 0.0
        print(f"    [calls/{split}] {class_name}: {len(file_list)} files")

        for file_path in file_list:
            audio = load_audio_file(file_path, config["sample_rate"])
            if audio is None:
                continue

            chunks = chunk_audio(audio, config["sample_rate"],
                                 config["chunk_duration"], config["min_chunk_duration"])

            for chunk in chunks:
                combined, prosody, valid = extract_features(chunk, extractor, seq_len, feat_dim)
                if valid:
                    seq_list.append(combined)
                    pros_list.append(prosody)
                    lbl_list.append(label)

                # One noise copy for normal chunks during training
                if is_train and config["calls_augment_normal"] and class_name == "normal":
                    aug = augment_audio(chunk, 0, config["noise_factor"])
                    c2, p2, v2 = extract_features(aug, extractor, seq_len, feat_dim)
                    if v2:
                        seq_list.append(c2)
                        pros_list.append(p2)
                        lbl_list.append(label)

    return seq_list, pros_list, lbl_list


# ================================================================
# LOAD WAKE WORD CLIPS
# ================================================================

def load_wake_words(config, split, wake_splits):
    extractor     = make_extractor(config)
    seq_len       = config["sequence_length"]
    feat_dim      = config["feature_dim"]
    chunk_samples = int(config["chunk_duration"] * config["sample_rate"])
    is_train      = (split == "train")
    aug_mult      = config["wake_augment_multiplier"]
    seq_list, pros_list, lbl_list = [], [], []

    for class_name, file_list in wake_splits[split].items():
        label = 1.0 if class_name == "scam" else 0.0
        expected = len(file_list) * (1 + (aug_mult if is_train else 0))
        print(f"    [wake/{split}] {class_name}: {len(file_list)} files -> ~{expected} samples")

        for file_path in file_list:
            audio = load_audio_file(file_path, config["sample_rate"])
            if audio is None:
                continue

            # Wake word clips are already ~3s: pad or truncate to exact size
            chunk = pad_or_truncate(audio, chunk_samples)

            combined, prosody, valid = extract_features(chunk, extractor, seq_len, feat_dim)
            if valid:
                seq_list.append(combined)
                pros_list.append(prosody)
                lbl_list.append(label)

            # Heavy augmentation for wake words (only 16 files per class)
            # Cycles through all 4 augmentation types for maximum variety
            if is_train and config["augment"]:
                for aug_idx in range(aug_mult):
                    aug_chunk = augment_audio(chunk, aug_idx % 4, config["noise_factor"])
                    c2, p2, v2 = extract_features(aug_chunk, extractor, seq_len, feat_dim)
                    if v2:
                        seq_list.append(c2)
                        pros_list.append(p2)
                        lbl_list.append(label)

    return seq_list, pros_list, lbl_list


# ================================================================
# COMBINE BOTH DATASETS
# ================================================================

def build_combined_arrays(call_splits, wake_splits, config, split):
    print(f"\n  Loading call recordings [{split}]...")
    seq_c, pros_c, lbl_c = load_call_recordings(config, split, call_splits)

    print(f"\n  Loading wake word clips [{split}]...")
    seq_w, pros_w, lbl_w = load_wake_words(config, split, wake_splits)

    all_seq  = seq_c  + seq_w
    all_pros = pros_c + pros_w
    all_lbl  = lbl_c  + lbl_w

    # Shuffle so batches contain mix of both sources
    combined = list(zip(all_seq, all_pros, all_lbl))
    random.Random(config["random_seed"]).shuffle(combined)
    all_seq, all_pros, all_lbl = zip(*combined)

    X_seq  = np.array(all_seq,  dtype=np.float32)
    X_pros = np.array(all_pros, dtype=np.float32)
    y      = np.array(all_lbl,  dtype=np.float32)

    scam_n   = int(np.sum(y))
    normal_n = int(np.sum(1 - y))
    print(f"\n  {split} combined: {len(y)} samples ({scam_n} scam, {normal_n} normal)")
    print(f"    from calls:      {len(seq_c)} samples")
    print(f"    from wake words: {len(seq_w)} samples")

    return X_seq, X_pros, y


# ================================================================
# TRAINING CALLBACKS
# ================================================================

def get_callbacks(model_name, save_dir, patience):
    best_path = os.path.join(save_dir, f"{model_name}_best.keras")
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=best_path, monitor="val_auc",
            mode="max", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
    ]


# ================================================================
# MAIN
# ================================================================

def train():

    print("=" * 60)
    print("  COMBINED TRAINING")
    print("  Call Recordings + Wake Word Clips")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  Directory:  {os.getcwd()}")
    print("=" * 60)

    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])

    os.makedirs(CONFIG["save_dir"],   exist_ok=True)
    os.makedirs(CONFIG["tflite_dir"], exist_ok=True)

    # 1. Split datasets
    print("\n[1/7] Splitting datasets...")
    print("  Call recordings:")
    call_splits = split_call_files(CONFIG)
    print("  Wake word clips:")
    wake_splits = split_wake_files(CONFIG)

    # 2. Training arrays
    print("\n[2/7] Building TRAINING arrays...")
    print("      (takes 5-10 minutes — one time only)")
    X_seq_tr, X_pros_tr, y_tr = build_combined_arrays(
        call_splits, wake_splits, CONFIG, "train"
    )

    # 3. Validation arrays
    print("\n[3/7] Building VALIDATION arrays...")
    X_seq_val, X_pros_val, y_val = build_combined_arrays(
        call_splits, wake_splits, CONFIG, "val"
    )

    print(f"\n  X_seq_tr  : {X_seq_tr.shape}")
    print(f"  X_pros_tr : {X_pros_tr.shape}")
    print(f"  X_seq_val : {X_seq_val.shape}")

    actual_seq = X_seq_tr.shape[1]
    if actual_seq != CONFIG["sequence_length"]:
        print(f"  Updating sequence_length: {CONFIG['sequence_length']} -> {actual_seq}")
        CONFIG["sequence_length"] = actual_seq

    # 4. Class weights
    print("\n[4/7] Computing class weights...")
    weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y_tr
    )
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  normal={class_weights[0]:.3f}  scam={class_weights[1]:.3f}")

    # 5. Train Sequence Model
    print("\n[5/7] Training Sequence Model...")
    seq_model = build_sequence_model(
        sequence_length=CONFIG["sequence_length"],
        feature_dim=CONFIG["feature_dim"],
    )
    seq_model.summary()
    seq_model.fit(
        x=X_seq_tr, y=y_tr,
        validation_data=(X_seq_val, y_val),
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        class_weight=class_weights,
        callbacks=get_callbacks("sequence_model", CONFIG["save_dir"],
                                CONFIG["early_stopping_patience"]),
        verbose=1,
    )

    # 6. Train Prosody Model
    print("\n[6/7] Training Prosody Model...")
    pros_model = build_prosody_model(input_dim=5)
    pros_model.summary()
    pros_model.fit(
        x=X_pros_tr, y=y_tr,
        validation_data=(X_pros_val, y_val),
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        class_weight=class_weights,
        callbacks=get_callbacks("prosody_model", CONFIG["save_dir"],
                                CONFIG["early_stopping_patience"]),
        verbose=1,
    )

    # 7. Save
    print("\n[7/7] Saving models...")
    save_models(seq_model, pros_model, CONFIG["save_dir"])

    print("\n[+] Exporting TFLite...")
    try:
        export_tflite(CONFIG["save_dir"], CONFIG["tflite_dir"])
    except Exception as e:
        print(f"  TFLite warning: {e}")
        print("  Keras models saved. check_file.py will work fine.")

    cfg_out = {
        "sequence_length":      CONFIG["sequence_length"],
        "feature_dim":          CONFIG["feature_dim"],
        "n_mfcc":               CONFIG["n_mfcc"],
        "n_fft":                CONFIG["n_fft"],
        "hop_length":           CONFIG["hop_length"],
        "n_mels":               CONFIG["n_mels"],
        "sample_rate":          CONFIG["sample_rate"],
        "chunk_duration":       CONFIG["chunk_duration"],
        "scam_threshold":       CONFIG["scam_threshold"],
        "suspicious_threshold": CONFIG["suspicious_threshold"],
    }
    with open(CONFIG["config_path"], "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"[+] training_config.json saved")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    print("\nSequence Model — Validation:")
    seq_model.evaluate(X_seq_val, y_val, verbose=1)

    print("\nProsody Model — Validation:")
    pros_model.evaluate(X_pros_val, y_val, verbose=1)

    print("\nNext step:")
    print("  python check_file.py path/to/audio.wav --verbose")


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    if not (cwd / "processed_dataset").exists():
        print("WARNING: Run from inside diversion/ folder:")
        print("  cd D:\\d_drive_project\\Realtime_fraud_call_detection\\diversion")
        print("  python train.py")
        print()
    train()