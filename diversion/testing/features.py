# features.py
# ============================================================
# CLEAN REBUILD — Step 1 of 4
#
# PURPOSE:
#   Convert raw audio into feature matrices that a neural
#   network can learn scam patterns from.
#
# WHAT THIS FILE PRODUCES:
#   1. Combined MFCC matrix  → shape (T, 39)
#      = MFCCs (13) + Delta MFCCs (13) + Delta-Delta MFCCs (13)
#   2. Prosody vector        → shape (5,)
#   3. Spectral entropy      → single float
#
# WHY 39 FEATURES PER FRAME (not 13 like before):
#   MFCCs alone tell you the SHAPE of the vocal tract at each moment.
#   Delta MFCCs tell you HOW FAST that shape is changing.
#   Delta-Delta MFCCs tell you if the change is SPEEDING UP or SLOWING DOWN.
#
#   Scam callers have a characteristic urgency pattern:
#   rapid vocal changes during threats, slower during reassurance.
#   This acceleration pattern is captured by delta-delta features
#   and generalises across different scam callers.
#
# USAGE:
#   from features import FeatureExtractor
#   extractor = FeatureExtractor()
#   result = extractor.extract_all(audio_array)
#   # result['combined']  shape (T, 39)
#   # result['prosody']   shape (5,)
#   # result['entropy']   float
#   # result['valid']     bool
# ============================================================

import numpy as np
import librosa
import scipy.signal
import scipy.stats


class FeatureExtractor:
    """
    Extracts 3 types of features from raw audio.

    All methods expect:
        audio: 1D numpy float32 array
               resampled to self.sample_rate (16000 Hz)
               values in range [-1.0, 1.0]
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc:      int = 13,
        n_fft:       int = 512,
        hop_length:  int = 160,
        n_mels:      int = 26,
    ):
        """
        sample_rate : target Hz — all audio must be resampled to this
        n_mfcc      : number of MFCC coefficients (13 is standard)
        n_fft       : FFT window size in samples
                      512 samples at 16000 Hz = 32ms per analysis window
        hop_length  : step between consecutive windows
                      160 samples at 16000 Hz = 10ms between frames
        n_mels      : mel filterbank channels before DCT compression
                      26 mel bands → compressed to 13 MFCCs by DCT
        """
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mels      = n_mels

        # The combined feature dimension is 3 * n_mfcc
        # (MFCC + Delta + Delta-Delta, each has n_mfcc features)
        self.combined_dim = 3 * n_mfcc   # = 39

        # Minimum samples needed for at least one FFT frame
        self.min_samples = n_fft   # = 512 samples = 32ms


    # ================================================================
    # FEATURE 1: COMBINED MFCC (MFCC + DELTA + DELTA-DELTA)
    # ================================================================

    def extract_combined_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts MFCCs, Delta MFCCs, and Delta-Delta MFCCs.
        Concatenates them into one matrix.

        STEP BY STEP:

        Step 1 — Raw MFCCs:
            Frame audio into 32ms windows, apply FFT, apply 26 mel filters,
            take log, apply DCT → 13 coefficients per frame.
            Shape: (13, T)

        Step 2 — CMVN Normalisation:
            For each of the 13 coefficients, subtract its mean over all frames
            and divide by its std. This removes microphone/room effects.
            Formula: mfcc_norm[i,t] = (mfcc[i,t] - mean[i]) / (std[i] + 1e-8)

        Step 3 — Delta MFCCs (first derivative):
            Measures how fast each MFCC coefficient is changing.
            Uses a ±2 frame window for numerical stability.
            Formula: delta[t] = Σ(n * mfcc[t+n]) / Σ(n²)  for n=-2,-1,0,1,2
            Shape: (13, T)

            INTUITION: If MFCC[0] goes [0.1, 0.3, 0.5, 0.7] across frames,
            Delta[0] goes [0.2, 0.2, 0.2, 0.2] — constant speed.
            Scam speech has sudden delta spikes (rapid pitch changes).

        Step 4 — Delta-Delta MFCCs (second derivative):
            Apply delta again to delta → measures acceleration.
            Shape: (13, T)

            INTUITION: Constant speed → Delta-Delta ≈ 0.
            Sudden urgency → Delta spikes → Delta-Delta spikes.
            This is highly characteristic of scam caller emotional patterns.

        Step 5 — Concatenate:
            Stack the three (13, T) matrices vertically → (39, T)
            Transpose → (T, 39)

        INPUT:  audio → 1D float32 array
        OUTPUT: combined → 2D float32 array, shape (T, 39)
                           Returns shape (0, 39) if audio too short.
        """

        # Guard: need at least one FFT window
        if len(audio) < self.min_samples:
            return np.zeros((0, self.combined_dim), dtype=np.float32)

        # ---- Step 1: Raw MFCCs ----
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        # mfccs.shape = (13, T)
        # T = floor((len(audio) - n_fft) / hop_length) + 1
        # Example: 3 seconds → T = (48000-512)//160 + 1 = 297 frames

        # ---- Step 2: CMVN Normalisation ----
        # Compute mean and std across time (axis=1) for each coefficient
        mean = np.mean(mfccs, axis=1, keepdims=True)   # shape (13, 1)
        std  = np.std( mfccs, axis=1, keepdims=True)   # shape (13, 1)

        # Normalise: each coefficient now has mean≈0 and std≈1 over time
        # 1e-8 prevents division by zero during silence
        mfccs = (mfccs - mean) / (std + 1e-8)
        # mfccs.shape still (13, T)

        # ---- Step 3: Delta MFCCs ----
        # librosa.feature.delta computes the finite difference derivative
        # width=9 means it uses a ±4 frame window (more stable than ±1)
        # This is the standard HTK/Kaldi delta computation
        delta_mfccs = librosa.feature.delta(mfccs, width=9)
        # delta_mfccs.shape = (13, T)

        # ---- Step 4: Delta-Delta MFCCs ----
        # Apply delta to delta → second derivative (acceleration)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, width=9)
        # delta2_mfccs.shape = (13, T)

        # ---- Step 5: Concatenate along feature axis ----
        # np.vstack stacks arrays row-wise (axis=0)
        # Before stacking: 3 arrays of shape (13, T)
        # After stacking:  one array of shape (39, T)
        combined = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        # combined.shape = (39, T)

        # Transpose to (T, 39) — time is the sequence dimension
        combined = combined.T.astype(np.float32)
        # combined.shape = (T, 39)

        return combined


    # ================================================================
    # FEATURE 2: PROSODY VECTOR
    # ================================================================

    def extract_prosody(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts 5 features that capture HOW something is said,
        not what is said.

        WHY PROSODY FOR SCAM DETECTION:
            Scam callers follow an emotional script:
            - Open with urgency (high pitch, fast speech)
            - Threaten with authority (loud, dropping pitch)
            - Reassure to prevent hanging up (quieter, slower)
            This pattern produces characteristic energy and pitch statistics
            that differ from normal conversation regardless of language.

        THE 5 FEATURES:

        1. pitch_mean (normalised):
           Average fundamental frequency of voiced frames.
           High pitch → nervousness, urgency, or fake authority.
           Scam callers often use unnaturally high or low pitch.
           Range: 80-400 Hz → normalised by dividing by 400.

        2. pitch_std (normalised):
           Variation in pitch across voiced frames.
           High variation → emotional agitation, scripted ups and downs.
           Normal conversation: moderate and natural variation.
           Range: 0-150 Hz std → normalised by dividing by 150.

        3. energy_mean (normalised):
           Average RMS (Root Mean Square) loudness.
           RMS[t] = sqrt( mean(x[n]² for n in frame t) )
           Scam callers tend to speak louder than normal.
           Range: 0-0.5 → normalised by dividing by 0.5.

        4. energy_std (normalised):
           Variation in loudness across frames.
           High std → erratic volume (scripted emotional swings).
           Range: 0-0.3 → normalised by dividing by 0.3.

        5. speech_rate (normalised):
           Proxy for syllables per second via energy peak counting.
           Each syllable produces a local energy peak.
           Scam callers speak faster to reduce thinking time.
           Range: 0-10 peaks/second → normalised by dividing by 10.

        INPUT:  audio  → 1D float32 array
        OUTPUT: vector → 1D float32 array, shape (5,), all values in [0,1]
        """

        # ---- Pitch via piptrack ----
        # piptrack returns pitch estimates and their confidence (magnitude)
        # at each time-frequency bin
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # pitches.shape    = (n_fft//2 + 1, T) = (257, T)
        # magnitudes.shape = (257, T)

        # For each time frame t, find the frequency bin with highest magnitude
        # magnitudes.argmax(axis=0) returns index of max along frequency axis
        best_idx = magnitudes.argmax(axis=0)    # shape (T,)

        # Get the pitch at the best index for each frame
        # np.arange(T) creates [0,1,2,...,T-1] for column indexing
        T = pitches.shape[1]
        pitch_contour = pitches[best_idx, np.arange(T)]   # shape (T,)

        # Only use voiced frames (pitch > 50 Hz → actual voice, not silence)
        voiced = pitch_contour[pitch_contour > 50]

        if len(voiced) > 0:
            pitch_mean = float(np.mean(voiced))
            pitch_std  = float(np.std(voiced))
        else:
            # No voiced frames detected (silence or very noisy)
            pitch_mean = 0.0
            pitch_std  = 0.0

        # ---- RMS Energy ----
        # librosa.feature.rms returns shape (1, T) — the [0] gets the 1D array
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )[0]
        # rms.shape = (T,)

        energy_mean = float(np.mean(rms))
        energy_std  = float(np.std(rms))

        # ---- Speech Rate via Energy Peaks ----
        # Find local maxima in the RMS energy curve.
        # Each peak corresponds approximately to a syllable nucleus.
        #
        # find_peaks parameters:
        #   height=mean(rms)  → only count peaks above average energy
        #                        (ignores background noise peaks)
        #   distance=8        → peaks must be at least 8 frames apart
        #                        8 frames × 10ms = 80ms minimum syllable gap
        #                        prevents double-counting one syllable
        mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0
        peaks, _ = scipy.signal.find_peaks(
            rms,
            height=mean_rms,
            distance=8,
        )
        # peaks = array of frame indices where energy peaks occur

        duration_sec = len(audio) / self.sample_rate
        speech_rate  = len(peaks) / duration_sec if duration_sec > 0 else 0.0

        # ---- Normalise all features to [0, 1] ----
        # np.clip prevents values above 1 from extreme edge cases
        pitch_mean_n  = float(np.clip(pitch_mean  / 400.0, 0.0, 1.0))
        pitch_std_n   = float(np.clip(pitch_std   / 150.0, 0.0, 1.0))
        energy_mean_n = float(np.clip(energy_mean / 0.5,   0.0, 1.0))
        energy_std_n  = float(np.clip(energy_std  / 0.3,   0.0, 1.0))
        speech_rate_n = float(np.clip(speech_rate / 10.0,  0.0, 1.0))

        return np.array(
            [pitch_mean_n, pitch_std_n, energy_mean_n,
             energy_std_n, speech_rate_n],
            dtype=np.float32,
        )
        # shape = (5,), all values in [0.0, 1.0]


    # ================================================================
    # FEATURE 3: SPECTRAL ENTROPY
    # ================================================================

    def extract_entropy(self, audio: np.ndarray) -> float:
        """
        Measures how spread-out or concentrated the frequency energy is.

        FORMULA (Shannon Entropy applied to power spectrum):
            For each time frame t:
                p[f,t] = S[f,t] / sum_f(S[f,t])    ← normalise to sum=1
                H[t]   = -sum_f( p[f,t] * log(p[f,t]) )

            Final value = mean(H[t]) over all frames, normalised to [0,1]

        INTUITION:
            A single musical note: all energy at one frequency → H ≈ 0 (low)
            White noise: energy spread equally → H ≈ 1 (high)
            Human speech: somewhere in between → H ≈ 0.7-0.85
            Robotic TTS voices: more tonal than humans → H ≈ 0.5-0.7

            Scam callers using voice-changing software or TTS
            will have lower entropy than genuine human callers.

        INPUT:  audio   → 1D float32 array
        OUTPUT: entropy → float in [0, 1]
        """

        if len(audio) < self.min_samples:
            return 0.0

        # Short-Time Fourier Transform
        # Returns complex matrix: stft[f,t] = complex amplitude at frequency f, time t
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # stft.shape = (n_fft//2 + 1, T) = (257, T)

        # Power spectrum: |stft|² = energy at each frequency bin and time
        power = np.abs(stft) ** 2
        # power.shape = (257, T)

        # Normalise each time frame to sum to 1 (probability distribution)
        col_sums = np.sum(power, axis=0, keepdims=True)   # shape (1, T)

        # Find silent frames (total power is zero or near zero)
        # We will exclude these from entropy calculation entirely
        # rather than adding epsilon (epsilon causes log(epsilon) = large negative)
        silent_frames = (col_sums[0] < 1e-10)   # shape (T,) bool mask

        # Replace zero sums with 1.0 temporarily to avoid division by zero
        # The resulting entropy for these frames will be discarded anyway
        col_sums_safe = np.where(col_sums < 1e-10, 1.0, col_sums)
        p_norm = power / col_sums_safe
        # p_norm.shape = (257, T), voiced columns sum to 1.0

        # Add tiny epsilon BEFORE log to prevent log(0) = -inf
        # This is different from before — we add it to p_norm not col_sums
        p_norm = np.clip(p_norm, 1e-10, 1.0)

        # Shannon entropy per frame: H[t] = -sum_f( p[f,t] * log(p[f,t]) )
        entropy_frames = scipy.stats.entropy(p_norm, axis=0)
        # entropy_frames.shape = (T,)

        # Only average over voiced (non-silent) frames
        voiced_entropy = entropy_frames[~silent_frames]
        if len(voiced_entropy) == 0:
            return 0.0

        # Replace any remaining nan/inf with 0 before averaging
        voiced_entropy = np.where(np.isfinite(voiced_entropy), voiced_entropy, 0.0)
        mean_h = float(np.mean(voiced_entropy))

        # Normalise by theoretical maximum entropy
        # Maximum = log(number_of_frequency_bins) = log(257) ≈ 5.55
        max_h = float(np.log(self.n_fft // 2 + 1))
        return float(np.clip(mean_h / max_h, 0.0, 1.0))


    # ================================================================
    # MASTER METHOD: extract_all
    # ================================================================

    def extract_all(self, audio: np.ndarray) -> dict:
        """
        Extracts all features from one audio chunk.
        This is the only method train.py and check_file.py call.

        INPUT:
            audio → 1D numpy float32 array at self.sample_rate Hz

        OUTPUT:
            dict with keys:
                'combined' → ndarray shape (T, 39)
                             MFCCs + Delta + Delta-Delta concatenated
                             T ≈ chunk_duration * 100
                             (100 frames per second at 10ms hop)

                'prosody'  → ndarray shape (5,)
                             [pitch_mean, pitch_std, energy_mean,
                              energy_std, speech_rate]
                             all normalised to [0, 1]

                'entropy'  → float in [0, 1]
                             spectral entropy of the chunk

                'valid'    → bool
                             False if audio was too short to process

        EXAMPLE:
            extractor = FeatureExtractor()
            audio = np.random.randn(48000).astype(np.float32) * 0.1
            f = extractor.extract_all(audio)
            print(f['combined'].shape)   # (297, 39)
            print(f['prosody'].shape)    # (5,)
            print(f['entropy'])          # float between 0 and 1
            print(f['valid'])            # True
        """

        combined = self.extract_combined_mfcc(audio)

        # Check if audio was too short
        if combined.shape[0] == 0:
            return {
                'combined': np.zeros((0, self.combined_dim), dtype=np.float32),
                'prosody':  np.zeros(5, dtype=np.float32),
                'entropy':  0.0,
                'valid':    False,
            }

        prosody = self.extract_prosody(audio)
        entropy = self.extract_entropy(audio)

        return {
            'combined': combined,   # (T, 39)
            'prosody':  prosody,    # (5,)
            'entropy':  entropy,    # float
            'valid':    True,
        }


# ================================================================
# SELF-TEST — run this file directly to verify everything works
# python features.py
# ================================================================

if __name__ == '__main__':

    import os
    from pathlib import Path

    print("=" * 50)
    print("features.py — Self Test")
    print("=" * 50)

    extractor = FeatureExtractor(sample_rate=16000)

    # Test 1: Synthetic audio
    print("\nTest 1: Synthetic audio (3 seconds)")
    audio_3s = np.random.randn(48000).astype(np.float32) * 0.1
    f = extractor.extract_all(audio_3s)

    print(f"  valid:            {f['valid']}")
    print(f"  combined shape:   {f['combined'].shape}")
    print(f"  expected shape:   (297, 39)")
    print(f"  prosody shape:    {f['prosody'].shape}")
    print(f"  prosody values:   {f['prosody'].round(3)}")
    print(f"  entropy:          {f['entropy']:.4f}")

    assert f['valid'],                          "FAIL: valid should be True"
    assert f['combined'].shape[1] == 39,        "FAIL: feature dim should be 39"
    assert f['combined'].shape[0]  > 0,         "FAIL: should have time frames"
    assert f['prosody'].shape  == (5,),         "FAIL: wrong prosody shape"
    print(f"  sequence_length:  {f['combined'].shape[0]}  (locked in for model.py)")
    assert 0.0 <= f['entropy'] <= 1.0,          "FAIL: entropy out of range"
    assert np.all(f['prosody'] >= 0),           "FAIL: prosody below 0"
    assert np.all(f['prosody'] <= 1),           "FAIL: prosody above 1"
    print("  PASSED")

    # Test 2: Too-short audio
    print("\nTest 2: Too-short audio (100 samples)")
    short_audio = np.random.randn(100).astype(np.float32)
    f2 = extractor.extract_all(short_audio)
    print(f"  valid: {f2['valid']}")
    assert not f2['valid'], "FAIL: should be invalid"
    print("  PASSED")

    # Test 3: Real audio file (if dataset exists)
    scam_dir = Path("processed_dataset/SCAM_CALLS")
    if scam_dir.exists():
        first_file = sorted(scam_dir.glob("*.wav"))[0]
        print(f"\nTest 3: Real audio file ({first_file.name})")
        audio_real, _ = librosa.load(str(first_file), sr=16000, mono=True)
        # Take first 3 seconds
        chunk = audio_real[:48000]
        f3 = extractor.extract_all(chunk)
        print(f"  valid:          {f3['valid']}")
        print(f"  combined shape: {f3['combined'].shape}")
        print(f"  prosody:        {f3['prosody'].round(3)}")
        print(f"  entropy:        {f3['entropy']:.4f}")
        assert f3['valid'], "FAIL: real audio should be valid"
        print("  PASSED")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED — features.py is ready")
    print("=" * 50)