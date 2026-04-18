# check_file.py
# ============================================================
# SCAM CALL DETECTOR — File Checker
#
# PURPOSE:
#   Load your trained models and analyse any .wav audio file
#   for scam call patterns.
#
# HOW TO RUN:
#   python check_file.py "C:/path/to/audio.wav"
#   python check_file.py "D:/calls/suspicious_call.wav"
#   python check_file.py "D:/calls/suspicious_call.wav" --verbose
#
# OUTPUT:
#   Chunk-by-chunk analysis + final verdict with confidence score
#
# REQUIREMENTS:
#   - trained models in saved_models/ folder (run train.py first)
#   - training_config.json in the same directory as this script
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path


# ================================================================
# STEP 0: RESOLVE PATHS RELATIVE TO THIS SCRIPT
# ================================================================

# Always resolve paths relative to where check_file.py lives,
# not wherever you run python from.
SCRIPT_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR     = Path(r"D:\d_drive_project\Realtime_fraud_call_detection\saved_models") 
CONFIG_PATH  = SCRIPT_DIR / "training_config.json"


# ================================================================
# STEP 1: LOAD CONFIG
# ================================================================

def load_config() -> dict:
    """
    Reads training_config.json saved by train.py.
    This ensures check_file.py uses the EXACT same parameters
    that were used during training (sequence_length, thresholds, etc.)
    """
    if not CONFIG_PATH.exists():
        print(f"\n  ERROR: training_config.json not found at:")
        print(f"    {CONFIG_PATH}")
        print(f"\n  Please run train.py first:")
        print(f"    python train.py")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print(f"  Config loaded from: {CONFIG_PATH}")
    print(f"    sequence_length  = {config['sequence_length']}")
    print(f"    feature_dim      = {config['feature_dim']}")
    print(f"    sample_rate      = {config['sample_rate']}")
    print(f"    chunk_duration   = {config['chunk_duration']}s")
    print(f"    scam_threshold   = {config['scam_threshold']}")
    print(f"    suspicious_thr   = {config['suspicious_threshold']}")

    return config


# ================================================================
# STEP 2: LOAD MODELS
# ================================================================

def load_models():
    """
    Loads both trained Keras models from saved_models/.
    Deferred import so TF only loads when needed.
    """
    try:
        from tensorflow import keras
    except ImportError:
        print("\n  ERROR: TensorFlow not installed.")
        print("  Install it with:  pip install tensorflow")
        sys.exit(1)

    seq_path  = SAVE_DIR / "sequence_model.keras"
    pros_path = SAVE_DIR / "prosody_model.keras"

    for path in [seq_path, pros_path]:
        if not path.exists():
            print(f"\n  ERROR: Model not found: {path}")
            print(f"\n  Please run train.py first:")
            print(f"    python train.py")
            sys.exit(1)

    print(f"\n  Loading models from: {SAVE_DIR}")
    seq_model  = keras.models.load_model(str(seq_path))
    pros_model = keras.models.load_model(str(pros_path))
    print(f"  Sequence model: loaded  ({seq_model.count_params():,} params)")
    print(f"  Prosody model:  loaded  ({pros_model.count_params():,} params)")

    return seq_model, pros_model


# ================================================================
# STEP 3: LOAD AND CHUNK AUDIO FILE
# ================================================================

def load_and_chunk(file_path: Path, config: dict) -> list:
    """
    Loads the audio file and splits it into fixed-length chunks.
    Identical logic to train.py so features are computed the same way.

    Returns: list of float32 numpy arrays, each of length chunk_samples
    """
    sample_rate        = config["sample_rate"]
    chunk_duration     = config["chunk_duration"]
    chunk_samples      = int(chunk_duration * sample_rate)
    min_chunk_duration = chunk_duration / 2          # allow half-length final chunk
    min_samples        = int(min_chunk_duration * sample_rate)

    print(f"\n  Loading: {file_path}")

    if not file_path.exists():
        print(f"\n  ERROR: File not found: {file_path}")
        sys.exit(1)

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    if file_path.suffix.lower() not in supported:
        print(f"\n  WARNING: Unsupported extension '{file_path.suffix}'")
        print(f"  Supported formats: {', '.join(supported)}")
        print(f"  Attempting to load anyway...")

    try:
        audio, sr = librosa.load(
            str(file_path),
            sr=sample_rate,
            mono=True,
            res_type="kaiser_fast",
        )
    except Exception as e:
        print(f"\n  ERROR: Could not load audio file: {e}")
        sys.exit(1)

    duration_sec = len(audio) / sample_rate
    print(f"  Duration:  {duration_sec:.1f}s  ({len(audio):,} samples at {sample_rate}Hz)")

    if len(audio) < min_samples:
        print(f"\n  ERROR: Audio too short ({duration_sec:.1f}s).")
        print(f"  Minimum required: {min_chunk_duration:.1f}s")
        sys.exit(1)

    chunks = []
    start  = 0

    while start + min_samples <= len(audio):
        end   = start + chunk_samples
        chunk = audio[start:end].copy()

        if len(chunk) < chunk_samples:
            # Pad last chunk with silence
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunks.append(chunk.astype(np.float32))
        start += chunk_samples

    print(f"  Chunks:    {len(chunks)} × {chunk_duration:.0f}s")
    return chunks


# ================================================================
# STEP 4: EXTRACT FEATURES FROM ONE CHUNK
# ================================================================

def extract_features(chunk: np.ndarray, config: dict) -> dict:
    """
    Extracts sequence + prosody features from one audio chunk.
    Uses the same FeatureExtractor class from features.py.
    """
    # Import from features.py in the same directory as this script
    sys.path.insert(0, str(SCRIPT_DIR))
    from features import FeatureExtractor

    extractor = FeatureExtractor(
        sample_rate=config["sample_rate"],
        n_mfcc=config["n_mfcc"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )

    return extractor.extract_all(chunk)


# ================================================================
# STEP 5: PAD / TRUNCATE SEQUENCE TO EXPECTED LENGTH
# ================================================================

def normalise_sequence(combined: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Makes sure the feature matrix is exactly (seq_len, feature_dim).
    Same logic used in train.py build_arrays().
    """
    T = combined.shape[0]

    if T < seq_len:
        pad = seq_len - T
        combined = np.pad(combined, ((0, pad), (0, 0)))
    elif T > seq_len:
        combined = combined[:seq_len, :]

    return combined   # shape (seq_len, 39)


# ================================================================
# STEP 6: FUSION + EMA SMOOTHING
# ================================================================

class FusionEngine:
    """
    Combines sequence and prosody scores with EMA smoothing.
    Mirrors model.py FusionEngine exactly.
    """

    def __init__(
        self,
        w_sequence:           float = 0.6,
        w_prosody:            float = 0.4,
        alpha:                float = 0.35,
        scam_threshold:       float = 0.65,
        suspicious_threshold: float = 0.40,
    ):
        self.w_sequence           = w_sequence
        self.w_prosody            = w_prosody
        self.alpha                = alpha
        self.scam_threshold       = scam_threshold
        self.suspicious_threshold = suspicious_threshold
        self._ema                 = None
        self.history              = []

    def fuse(self, seq_score: float, pros_score: float) -> dict:
        raw = self.w_sequence * seq_score + self.w_prosody * pros_score

        if self._ema is None:
            smooth = raw
        else:
            smooth = self.alpha * raw + (1.0 - self.alpha) * self._ema

        self._ema = smooth
        self.history.append(smooth)

        if smooth >= self.scam_threshold:
            alert = "SCAM"
        else:
            alert = "SAFE"

        return {
            "raw_score":    float(raw),
            "smooth_score": float(smooth),
            "alert_level":  alert,
            "seq_score":    float(seq_score),
            "pros_score":   float(pros_score),
        }
    



# ================================================================
# STEP 7: FINAL VERDICT
# ================================================================

def final_verdict(fusion: FusionEngine, config: dict) -> dict:
    """
    Aggregates chunk-level results into a single file-level verdict.

    VERDICT LOGIC:
        max_score  >= scam_threshold       -> SCAM
        mean_score >= suspicious_threshold -> SUSPICIOUS
        otherwise                          -> SAFE

    Using max_score for SCAM catches short bursts of scam speech
    even if the rest of the call is normal.
    Using mean_score for SUSPICIOUS avoids false alarms from one bad chunk.
    """
    arr = np.array(fusion.history)

    max_score  = float(np.max(arr))
    mean_score = float(np.mean(arr))

    scam_thr = config["scam_threshold"]
    sus_thr  = config["suspicious_threshold"]

    scam_chunks       = int(np.sum(arr >= scam_thr))
    suspicious_chunks = int(np.sum((arr >= sus_thr) & (arr < scam_thr)))
    safe_chunks       = len(arr) - scam_chunks - suspicious_chunks
    scam_ratio        = scam_chunks / len(arr)
    if(scam_ratio>=0.30 and max_score >=scam_thr):
        verdict    = "SCAM"
        confidence = max_score
        reason     = f"Peak score {max_score:.3f} exceeded scam threshold {scam_thr} with {scam_ratio:.1%} of chunks flagged"
    else:
        verdict    = "SAFE"
        confidence = 1.0 - mean_score
        reason     = f"Average score {mean_score:.3f} is below all thresholds"

    return {
        "verdict":           verdict,
        "confidence":        confidence,
        "max_score":         max_score,
        "mean_score":        mean_score,
        "total_chunks":      len(arr),
        "scam_chunks":       scam_chunks,
        "suspicious_chunks": suspicious_chunks,
        "safe_chunks":       safe_chunks,
        "reason":            reason,
    }


# ================================================================
# DISPLAY HELPERS
# ================================================================

COLORS = {
    "SCAM":       "\033[91m",   # bright red
    "SUSPICIOUS": "\033[93m",   # bright yellow
    "SAFE":       "\033[92m",   # bright green
    "RESET":      "\033[0m",
    "BOLD":       "\033[1m",
    "DIM":        "\033[2m",
}

ICONS = {
    "SCAM":       "🚨",
    "SAFE": "⚠️",
    "SAFE":       "✅",
}

BAR_WIDTH = 30


def score_bar(score: float, width: int = BAR_WIDTH) -> str:
    """Visual progress bar: [████████░░░░░░░░░░░░░░░░░░░░░░] 0.45"""
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.3f}"


def alert_color(level: str) -> str:
    return COLORS.get(level, "") + level + COLORS["RESET"]


# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyse_file(file_path: str, verbose: bool = False):
    """
    Full pipeline: load -> chunk -> extract -> predict -> fuse -> verdict.
    """

    print("=" * 60)
    print(f"  SCAM CALL DETECTOR")
    print("=" * 60)

    # ---- Load config ----
    print("\n[1/5] Loading configuration...")
    config = load_config()

    # ---- Load models ----
    print("\n[2/5] Loading models...")
    seq_model, pros_model = load_models()

    # ---- Load audio ----
    print("\n[3/5] Loading audio file...")
    audio_path = Path(file_path)
    chunks     = load_and_chunk(audio_path, config)

    # ---- Pre-build feature extractor once (avoid re-importing every chunk) ----
    sys.path.insert(0, str(SCRIPT_DIR))
    from features import FeatureExtractor

    extractor = FeatureExtractor(
        sample_rate=config["sample_rate"],
        n_mfcc=config["n_mfcc"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )

    seq_len  = config["sequence_length"]
    fusion   = FusionEngine(
        scam_threshold=config["scam_threshold"],
        suspicious_threshold=config["suspicious_threshold"],
    )

    # ---- Analyse each chunk ----
    print(f"\n[4/5] Analysing {len(chunks)} chunks...")

    if verbose:
        print(f"\n  {'Chunk':>5}  {'Time':>8}  {'Seq':>6}  {'Pros':>6}  "
              f"{'Raw':>6}  {'Smooth':>6}  Alert")
        print("  " + "-" * 58)

    skipped = 0

    for i, chunk in enumerate(chunks):
        start_time = i * config["chunk_duration"]
        end_time   = start_time + config["chunk_duration"]

        # Extract features
        feats = extractor.extract_all(chunk)

        if not feats["valid"]:
            skipped += 1
            if verbose:
                print(f"  {i+1:>5}  {start_time:>4.0f}-{end_time:>3.0f}s  "
                      f"{'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  SKIPPED (too short)")
            continue

        # Normalise sequence length
        combined = normalise_sequence(feats["combined"], seq_len)
        prosody  = feats["prosody"]

        # Predict with both models
        # Model expects batch dimension: (1, seq_len, 39) and (1, 5)
        seq_input  = combined[np.newaxis, :, :]     # (1, 301, 39)
        pros_input = prosody[np.newaxis, :]          # (1, 5)

        seq_score  = float(seq_model.predict(seq_input,   verbose=0)[0][0])
        pros_score = float(pros_model.predict(pros_input, verbose=0)[0][0])

        # Fuse scores
        result = fusion.fuse(seq_score, pros_score)

        if verbose:
            print(f"  {i+1:>5}  {start_time:>4.0f}-{end_time:>3.0f}s  "
                  f"{seq_score:>6.3f}  {pros_score:>6.3f}  "
                  f"{result['raw_score']:>6.3f}  {result['smooth_score']:>6.3f}  "
                  f"{result['alert_level']}")

    if skipped > 0:
        print(f"\n  Note: {skipped} chunk(s) skipped (too short / silent)")

    if not fusion.history:
        print("\n  ERROR: No valid chunks were processed.")
        print("  The audio file may be too short or completely silent.")
        sys.exit(1)

    # ---- Final verdict ----
    print(f"\n[5/5] Computing final verdict...")
    verdict_data = final_verdict(fusion, config)

    # ================================================================
    # PRINT RESULTS
    # ================================================================

    v      = verdict_data["verdict"]
    c      = COLORS.get(v, "")
    reset  = COLORS["RESET"]
    bold   = COLORS["BOLD"]
    icon   = ICONS.get(v, "")

    print("\n" + "=" * 60)
    print(f"  RESULT:  {c}{bold}{icon}  {v}{reset}")
    print("=" * 60)

    print(f"\n  File:         {audio_path.name}")
    print(f"  Duration:     {len(chunks) * config['chunk_duration']:.0f}s  "
          f"({len(chunks)} chunks × {config['chunk_duration']:.0f}s)")

    print(f"\n  Score Summary:")
    print(f"    Peak score:   {score_bar(verdict_data['max_score'])}")
    print(f"    Mean score:   {score_bar(verdict_data['mean_score'])}")

    print(f"\n  Chunk Breakdown:")
    print(f"    🚨 Scam       : {verdict_data['scam_chunks']:>3} chunks")
    print(f"    ⚠️  Safe : {verdict_data['suspicious_chunks']:>3} chunks")
    print(f"    ✅ Safe        : {verdict_data['safe_chunks']:>3} chunks")

    print(f"\n  Confidence:   {verdict_data['confidence']:.1%}")
    print(f"  Reason:       {verdict_data['reason']}")

    print(f"\n  Thresholds:")
    print(f"    Scam         >= {config['scam_threshold']}")
    print(f"    Suspicious   >= {config['suspicious_threshold']}")

    print("\n" + "=" * 60)

    return verdict_data


# ================================================================
# ENTRY POINT
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scam Call Detector — Analyse a .wav audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_file.py "C:/calls/audio.wav"
  python check_file.py "D:/suspicious.wav" --verbose
  python check_file.py "C:/Users/user/Downloads/call.wav" -v
        """
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Full path to the audio file (.wav, .mp3, .flac, etc.)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-chunk scores during analysis"
    )

    args = parser.parse_args()

    result = analyse_file(args.file_path, verbose=args.verbose)

    # Exit code: 1 = SCAM, 2 = SUSPICIOUS, 0 = SAFE
    # Useful if you want to use this in shell scripts or automation
    exit_codes = {"SCAM": 1, "SAFE": 2, "SAFE": 0}
    sys.exit(exit_codes.get(result["verdict"], 0))


if __name__ == "__main__":
    main()