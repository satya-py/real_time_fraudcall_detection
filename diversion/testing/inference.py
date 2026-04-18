# inference.py
# ============================================================
# SCAM CALL DETECTOR — Inference Reference Code
#
# PURPOSE:
#   This file shows the Android developer exactly what the model
#   does step by step. They will rewrite this in Kotlin/Java.
#
# THE 5 STEPS:
#   1. Load audio (16000 Hz, mono)
#   2. Cut into 3-second chunks
#   3. Extract features (MFCC + Delta + Delta-Delta + Prosody)
#   4. Run both models, fuse scores
#   5. Return SCAM or SAFE verdict
#
# HOW TO RUN:
#   python inference.py "path/to/audio.wav"
#
# MODEL FILES NEEDED:
#   tflite_models/sequence_model.tflite  <- for Android
#   tflite_models/prosody_model.tflite   <- for Android
#   OR
#   saved_models/sequence_model.keras    <- for Python only
#   saved_models/prosody_model.keras     <- for Python only
# ============================================================

import numpy as np
import librosa
import scipy.stats
from pathlib import Path


# ================================================================
# SETTINGS
# These exact values must be hardcoded in Android too.
# ================================================================

SAMPLE_RATE      = 16000   # Hz — all audio resampled to this
CHUNK_DURATION   = 3.0     # seconds per chunk
CHUNK_SAMPLES    = int(CHUNK_DURATION * SAMPLE_RATE)  # = 48000 samples

N_MFCC           = 13      # number of MFCC coefficients
N_FFT            = 512     # FFT window size (32ms)
HOP_LENGTH       = 160     # frame shift (10ms)
N_MELS           = 26      # mel filter banks

SEQUENCE_LENGTH  = 301     # time frames per chunk (fixed)
FEATURE_DIM      = 39      # 13 mfcc + 13 delta + 13 delta-delta

SCAM_THRESHOLD   = 0.75    # above this = SCAM
EMA_ALPHA        = 0.35    # smoothing factor (0=no change, 1=no memory)

W_SEQUENCE       = 0.6     # weight for sequence model score
W_PROSODY        = 0.4     # weight for prosody model score


# ================================================================
# STEP 1: LOAD AUDIO
# ================================================================

def load_audio(file_path: str) -> np.ndarray:
    """
    Loads any audio file and converts to:
        - mono (single channel)
        - 16000 Hz sample rate
        - float32 values between -1.0 and 1.0

    ANDROID EQUIVALENT:
        Use AudioRecord with:
            sampleRateInHz = 16000
            channelConfig  = CHANNEL_IN_MONO
            audioFormat    = ENCODING_PCM_FLOAT
        Or use MediaCodec to decode MP3/AAC files.
    """
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        res_type="kaiser_fast",
    )
    return audio.astype(np.float32)


# ================================================================
# STEP 2: CUT INTO 3-SECOND CHUNKS
# ================================================================

def chunk_audio(audio: np.ndarray) -> list:
    """
    Cuts audio into non-overlapping 3-second chunks.

    Each chunk = 48000 samples (3.0s × 16000 Hz).
    Last chunk is padded with silence if shorter than 1.5s.
    Chunks shorter than 1.5s are discarded.

    ANDROID EQUIVALENT:
        Process audio in a circular buffer.
        Every time buffer reaches 48000 samples, extract and process.
        For real-time: use a sliding window that fires every 3 seconds.

    Example:
        60-second call → 20 chunks of 3 seconds each
    """
    min_samples = CHUNK_SAMPLES // 2   # 24000 = 1.5 seconds minimum

    chunks = []
    start  = 0

    while start + min_samples <= len(audio):
        chunk = audio[start : start + CHUNK_SAMPLES].copy()

        if len(chunk) < CHUNK_SAMPLES:
            # Pad last short chunk with silence (zeros)
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        chunks.append(chunk)
        start += CHUNK_SAMPLES

    return chunks


# ================================================================
# STEP 3A: EXTRACT MFCC FEATURES (Sequence Input)
# ================================================================

def extract_mfcc_features(chunk: np.ndarray) -> np.ndarray:
    """
    Extracts a (301, 39) feature matrix from one audio chunk.

    THE 39 FEATURES PER FRAME:
        Columns  0-12: Raw MFCCs      — vocal tract shape
        Columns 13-25: Delta MFCCs    — rate of change (urgency)
        Columns 26-38: Delta-Delta    — acceleration (stress patterns)

    CMVN NORMALISATION:
        Subtract mean, divide by std for each coefficient.
        Makes features consistent regardless of recording volume.

    WHY 301 FRAMES:
        3 seconds / 10ms per frame = 300 frames.
        librosa adds 1 extra frame → 301.
        This is fixed for your trained model.

    ANDROID EQUIVALENT:
        Use TarsosDSP library for MFCC extraction.
        Or implement FFT manually using Android's AudioRecord buffer.
        Key parameters to match exactly:
            frameLength  = 512  (N_FFT)
            frameShift   = 160  (HOP_LENGTH)
            numMFCC      = 13   (N_MFCC)
            numMelFilters = 26  (N_MELS)

    INPUT:  float32 array of length 48000
    OUTPUT: float32 array of shape (301, 39)
    """

    # Step A: Compute 13 raw MFCCs
    # Shape: (13, T) where T ≈ 301 time frames
    mfcc = librosa.feature.mfcc(
        y=chunk,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    # Step B: Compute Delta MFCCs (velocity)
    # delta[t] = (mfcc[t+1] - mfcc[t-1]) / 2
    # Captures how fast the vocal tract is moving
    delta = librosa.feature.delta(mfcc, order=1)

    # Step C: Compute Delta-Delta MFCCs (acceleration)
    # delta2[t] = (delta[t+1] - delta[t-1]) / 2
    # Captures sudden changes — characteristic of scripted urgency
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Step D: Stack all 3 into one matrix
    # Shape: (39, T) — 13+13+13 = 39 features per frame
    combined = np.vstack([mfcc, delta, delta2])

    # Step E: CMVN normalisation per coefficient
    # Each of the 39 rows gets mean=0, std=1 normalisation
    mean = np.mean(combined, axis=1, keepdims=True)
    std  = np.std(combined,  axis=1, keepdims=True)
    combined = (combined - mean) / (std + 1e-8)

    # Step F: Transpose to (T, 39) — time-first format for Conv1D
    combined = combined.T

    # Step G: Pad or truncate to exactly SEQUENCE_LENGTH=301 frames
    T = combined.shape[0]
    if T < SEQUENCE_LENGTH:
        combined = np.pad(combined, ((0, SEQUENCE_LENGTH - T), (0, 0)))
    elif T > SEQUENCE_LENGTH:
        combined = combined[:SEQUENCE_LENGTH, :]

    return combined.astype(np.float32)
    # Shape: (301, 39)


# ================================================================
# STEP 3B: EXTRACT PROSODY FEATURES (Prosody Input)
# ================================================================

def extract_prosody_features(chunk: np.ndarray) -> np.ndarray:
    """
    Extracts 5 prosody features from one audio chunk.

    THE 5 FEATURES:
        [0] pitch_mean   — average fundamental frequency (F0)
                           High pitch = stress/urgency
        [1] pitch_std    — pitch variation
                           High variation = emotional speech
        [2] energy_mean  — average loudness (RMS)
                           Scam calls often louder/more assertive
        [3] energy_std   — loudness variation
                           Scripted speech has different energy pattern
        [4] speech_rate  — syllables per second (approximation)
                           Scam calls often faster/more urgent

    All values normalised to approximately [0, 1].

    ANDROID EQUIVALENT:
        pitch_mean: use autocorrelation or YIN algorithm on each frame
        energy:     compute RMS of each 160-sample frame
        speech_rate: count energy peaks using findpeaks equivalent

    INPUT:  float32 array of length 48000
    OUTPUT: float32 array of shape (5,)
    """

    # Pitch (F0) via STFT peak-picking
    # pitches shape: (freq_bins, time_frames)
    # magnitudes shape: (freq_bins, time_frames)
    pitches, magnitudes = librosa.piptrack(
        y=chunk,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # Extract dominant pitch per frame (highest magnitude bin)
    pitch_per_frame = []
    for t in range(pitches.shape[1]):
        mag_col = magnitudes[:, t]
        if mag_col.max() > 0:
            idx = mag_col.argmax()
            pitch_per_frame.append(pitches[idx, t])

    if pitch_per_frame:
        voiced = [p for p in pitch_per_frame if p > 50]  # filter unvoiced
        pitch_mean = float(np.mean(voiced)) / 400.0 if voiced else 0.0
        pitch_std  = float(np.std(voiced))  / 150.0 if voiced else 0.0
    else:
        pitch_mean = 0.0
        pitch_std  = 0.0

    # RMS energy per frame
    # Each frame = 160 samples = 10ms
    rms = librosa.feature.rms(
        y=chunk,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]  # shape: (T,)

    energy_mean = float(np.mean(rms)) / 0.5
    energy_std  = float(np.std(rms))  / 0.3

    # Speech rate: count energy peaks as syllable proxy
    # A peak = one syllable onset
    threshold   = float(np.mean(rms)) * 1.5
    peaks       = np.sum(np.diff((rms > threshold).astype(int)) > 0)
    speech_rate = float(peaks) / (CHUNK_DURATION * 10.0)

    # Clip all values to [0, 1]
    prosody = np.array([
        np.clip(pitch_mean,  0, 1),
        np.clip(pitch_std,   0, 1),
        np.clip(energy_mean, 0, 1),
        np.clip(energy_std,  0, 1),
        np.clip(speech_rate, 0, 1),
    ], dtype=np.float32)

    return prosody
    # Shape: (5,)


# ================================================================
# STEP 4A: RUN MODELS (Python / Keras version)
# ================================================================

def load_keras_models():
    """
    Loads the trained Keras models.
    Used for Python inference only.
    Android uses the TFLite version (see load_tflite_models below).
    """
    from tensorflow import keras

    SEQ_PATH  = r"D:\d_drive_project\Realtime_fraud_call_detection\saved_models\sequence_model.keras"
    PROS_PATH = r"D:\d_drive_project\Realtime_fraud_call_detection\saved_models\prosody_model.keras"

    seq_model  = keras.models.load_model(SEQ_PATH)
    pros_model = keras.models.load_model(PROS_PATH)

    print(f"Sequence model: {seq_model.count_params():,} parameters")
    print(f"Prosody model:  {pros_model.count_params():,} parameters")

    return seq_model, pros_model


def predict_keras(seq_model, pros_model,
                  mfcc_features: np.ndarray,
                  prosody_features: np.ndarray) -> tuple:
    """
    Runs both Keras models on one chunk's features.

    INPUT:
        mfcc_features    shape (301, 39)
        prosody_features shape (5,)

    OUTPUT:
        seq_score  float [0,1] — sequence model scam probability
        pros_score float [0,1] — prosody model scam probability

    ANDROID EQUIVALENT: use predict_tflite() below
    """
    # Add batch dimension: (301,39) → (1,301,39)
    seq_input  = mfcc_features[np.newaxis, :, :]
    # Add batch dimension: (5,) → (1,5)
    pros_input = prosody_features[np.newaxis, :]

    seq_score  = float(seq_model.predict(seq_input,   verbose=0)[0][0])
    pros_score = float(pros_model.predict(pros_input, verbose=0)[0][0])

    return seq_score, pros_score


# ================================================================
# STEP 4B: RUN MODELS (TFLite version — use this for Android)
# ================================================================

def load_tflite_models():
    """
    Loads both TFLite models.
    This is what Android uses.

    ANDROID EQUIVALENT:
        // Load model from assets folder
        Interpreter seqInterpreter = new Interpreter(
            loadModelFile(context, "sequence_model.tflite")
        );
        Interpreter prosInterpreter = new Interpreter(
            loadModelFile(context, "prosody_model.tflite")
        );
    """
    import tensorflow as tf

    SEQ_TFLITE  = r"D:\d_drive_project\Realtime_fraud_call_detection\tflite_models\sequence_model.tflite"
    PROS_TFLITE = r"D:\d_drive_project\Realtime_fraud_call_detection\tflite_models\prosody_model.tflite"

    seq_interpreter  = tf.lite.Interpreter(model_path=SEQ_TFLITE)
    pros_interpreter = tf.lite.Interpreter(model_path=PROS_TFLITE)

    seq_interpreter.allocate_tensors()
    pros_interpreter.allocate_tensors()

    print("TFLite models loaded")
    return seq_interpreter, pros_interpreter


def predict_tflite(seq_interp, pros_interp,
                   mfcc_features: np.ndarray,
                   prosody_features: np.ndarray) -> tuple:
    """
    Runs both TFLite models on one chunk.

    This is the function the Android developer should replicate in Kotlin.

    INPUT:
        mfcc_features    float32 array shape (301, 39)
        prosody_features float32 array shape (5,)

    OUTPUT:
        seq_score  float [0,1]
        pros_score float [0,1]

    ANDROID EQUIVALENT (Kotlin):
        // Sequence model
        val seqInput = Array(1) { Array(301) { FloatArray(39) } }
        // fill seqInput[0] with mfcc_features
        val seqOutput = Array(1) { FloatArray(1) }
        seqInterpreter.run(seqInput, seqOutput)
        val seqScore = seqOutput[0][0]

        // Prosody model
        val prosInput = Array(1) { FloatArray(5) }
        // fill prosInput[0] with prosody_features
        val prosOutput = Array(1) { FloatArray(1) }
        prosInterpreter.run(prosInput, prosOutput)
        val prosScore = prosOutput[0][0]
    """

    # ---- Sequence model ----
    seq_input_details  = seq_interp.get_input_details()
    seq_output_details = seq_interp.get_output_details()

    # Model expects shape (1, 301, 39)
    seq_input = mfcc_features[np.newaxis, :, :].astype(np.float32)
    seq_interp.set_tensor(seq_input_details[0]['index'], seq_input)
    seq_interp.invoke()
    seq_score = float(seq_interp.get_tensor(
        seq_output_details[0]['index'])[0][0])

    # ---- Prosody model ----
    pros_input_details  = pros_interp.get_input_details()
    pros_output_details = pros_interp.get_output_details()

    # Model expects shape (1, 5)
    pros_input = prosody_features[np.newaxis, :].astype(np.float32)
    pros_interp.set_tensor(pros_input_details[0]['index'], pros_input)
    pros_interp.invoke()
    pros_score = float(pros_interp.get_tensor(
        pros_output_details[0]['index'])[0][0])

    return seq_score, pros_score


# ================================================================
# STEP 5: FUSE SCORES + EMA SMOOTHING
# ================================================================

def fuse_scores(seq_score: float,
                pros_score: float,
                ema_state: float) -> tuple:
    """
    Combines two model scores into one smoothed risk score.

    WEIGHTED AVERAGE:
        raw = 0.6 * seq_score + 0.4 * pros_score
        Sequence model gets more weight (sees 11,739 values vs 5)

    EMA SMOOTHING:
        smooth = 0.35 * raw + 0.65 * previous_smooth
        Prevents one noisy chunk from spiking the score.
        ema_state = None means first chunk (no history yet).

    ANDROID EQUIVALENT (Kotlin):
        val raw = 0.6f * seqScore + 0.4f * prosScore
        val smooth = if (emaState == null) raw
                     else 0.35f * raw + 0.65f * emaState
        val verdict = if (smooth >= SCAM_THRESHOLD) "SCAM" else "SAFE"

    INPUT:
        seq_score  float [0,1]
        pros_score float [0,1]
        ema_state  float or None (previous smoothed score)

    OUTPUT:
        smooth_score float [0,1]
        verdict      "SCAM" or "SAFE"
        new_ema      float (pass this back in as ema_state next chunk)
    """
    raw = W_SEQUENCE * seq_score + W_PROSODY * pros_score

    if ema_state is None:
        smooth = raw
    else:
        smooth = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * ema_state

    verdict = "SCAM" if smooth >= SCAM_THRESHOLD else "SAFE"

    return smooth, verdict, smooth   # last value = new ema_state


# ================================================================
# FULL PIPELINE — called once per audio file
# ================================================================

def analyse_audio(file_path: str, use_tflite: bool = False) -> dict:
    """
    Full inference pipeline from audio file to verdict.

    FLOW:
        audio file
            ↓ load_audio()
        float32 array (N samples)
            ↓ chunk_audio()
        list of 3s chunks
            ↓ for each chunk:
                extract_mfcc_features()    → (301, 39)
                extract_prosody_features() → (5,)
                predict_tflite/keras()     → seq_score, pros_score
                fuse_scores()              → smooth_score, verdict
            ↓
        final verdict = SCAM if any chunk reaches threshold with
                        enough consecutive SCAM chunks

    INPUT:  path to any .wav audio file
    OUTPUT: dict with verdict, confidence, chunk_results
    """

    print(f"\nAnalysing: {Path(file_path).name}")
    print("-" * 40)

    # Load models
    if use_tflite:
        seq_model, pros_model = load_tflite_models()
        predict_fn = predict_tflite
    else:
        seq_model, pros_model = load_keras_models()
        predict_fn = predict_keras

    # Load audio
    audio = load_audio(file_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s")

    # Cut into chunks
    chunks = chunk_audio(audio)
    print(f"Chunks:   {len(chunks)} × {CHUNK_DURATION:.0f}s")
    print()

    # Process each chunk
    ema_state     = None
    chunk_results = []
    scam_count    = 0

    print(f"{'Chunk':>5}  {'Time':>8}  {'Seq':>6}  {'Pros':>6}  "
          f"{'Fused':>6}  {'Smooth':>6}  Verdict")
    print("-" * 55)

    for i, chunk in enumerate(chunks):
        start_s = i * CHUNK_DURATION
        end_s   = start_s + CHUNK_DURATION

        # Extract features
        mfcc_feat    = extract_mfcc_features(chunk)
        prosody_feat = extract_prosody_features(chunk)

        # Run models
        seq_score, pros_score = predict_fn(
            seq_model, pros_model,
            mfcc_feat, prosody_feat
        )

        # Fuse + smooth
        smooth, verdict, ema_state = fuse_scores(
            seq_score, pros_score, ema_state
        )

        fused = W_SEQUENCE * seq_score + W_PROSODY * pros_score

        if verdict == "SCAM":
            scam_count += 1

        chunk_results.append({
            "chunk":      i + 1,
            "start_s":    start_s,
            "end_s":      end_s,
            "seq_score":  seq_score,
            "pros_score": pros_score,
            "fused":      fused,
            "smooth":     smooth,
            "verdict":    verdict,
        })

        icon = "🚨" if verdict == "SCAM" else "✅"
        print(f"  {i+1:>3}  {start_s:>4.0f}-{end_s:>3.0f}s  "
              f"{seq_score:>6.3f}  {pros_score:>6.3f}  "
              f"{fused:>6.3f}  {smooth:>6.3f}  {icon} {verdict}")

    # Final verdict
    scam_ratio  = scam_count / len(chunk_results) if chunk_results else 0
    all_smooths = [r["smooth"] for r in chunk_results]
    max_smooth  = max(all_smooths) if all_smooths else 0
    mean_smooth = sum(all_smooths) / len(all_smooths) if all_smooths else 0

    # SCAM if: peak score >= threshold AND >= 30% of chunks are SCAM
    # This prevents one noisy chunk from triggering SCAM verdict
    if max_smooth >= SCAM_THRESHOLD and scam_ratio >= 0.30:
        final_verdict  = "SCAM"
        confidence     = max_smooth
    else:
        final_verdict  = "SAFE"
        confidence     = 1.0 - mean_smooth

    print()
    print("=" * 40)
    icon = "🚨" if final_verdict == "SCAM" else "✅"
    print(f"RESULT: {icon} {final_verdict}")
    print(f"Confidence:  {confidence:.1%}")
    print(f"Peak score:  {max_smooth:.3f}  (threshold: {SCAM_THRESHOLD})")
    print(f"Mean score:  {mean_smooth:.3f}")
    print(f"Scam chunks: {scam_count}/{len(chunk_results)} = {scam_ratio:.0%}")
    print("=" * 40)

    return {
        "verdict":       final_verdict,
        "confidence":    confidence,
        "max_score":     max_smooth,
        "mean_score":    mean_smooth,
        "scam_ratio":    scam_ratio,
        "total_chunks":  len(chunk_results),
        "scam_chunks":   scam_count,
        "chunk_results": chunk_results,
    }


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file> [--tflite]")
        print()
        print("Examples:")
        print('  python inference.py "call.wav"')
        print('  python inference.py "call.wav" --tflite')
        sys.exit(1)

    audio_file = sys.argv[1]
    use_tflite = "--tflite" in sys.argv

    result = analyse_audio(audio_file, use_tflite=use_tflite)