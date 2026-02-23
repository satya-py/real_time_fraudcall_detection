import numpy as np
import librosa
import scipy.stats

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=512, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_mfcc(self, audio):
        """
        Extract MFCCs (Mel Frequency Cepstral Coefficients).
        """
        # Safety check for very short audio
        if len(audio) < self.n_fft:
            # Return zeros if audio is too short
            return np.zeros((0, self.n_mfcc), dtype=np.float32)

        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc, 
                                     n_fft=self.n_fft, hop_length=self.hop_length)
        
        # CMVN (Cepstral Mean and Variance Normalization)
        # Normalize over time
        mean = np.mean(mfccs, axis=1, keepdims=True)
        std = np.std(mfccs, axis=1, keepdims=True)
        mfccs = (mfccs - mean) / (std + 1e-8)
        
        # Transpose to (Time, Features)
        return mfccs.T

    def extract_prosody(self, audio):
        """
        Extract basic prosodic features: Pitch, Energy, Speech Rate approximation.
        """
        # 1. Pitch (F0)
        # Using a fast method if possible, pyin is slow but robust.
        # Check librosa version, piptrack or pyin. Using piptrack for speed in prototype.
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, 
                                               n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Select pitch with highest magnitude at each time step
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch if pitch > 0 else 0) # 0 for unvoiced
        
        pitch_contour = np.array(pitch_contour)
        
        # 2. Energy (RMS)
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        
        # 3. Speech Rate (Zero Crossings proxy)
        # A true speech rate needs syllable counting which is hard without ASR.
        # We use peak prominence of the energy envelope as a proxy for syllable nuclei.
        # This is a heuristic.
        peaks, _ = scipy.signal.find_peaks(rms, height=np.mean(rms), distance=5)
        speech_rate_proxy = len(peaks) / (len(audio) / self.sample_rate) if len(audio) > 0 else 0
        
        return {
            "pitch_mean": np.mean(pitch_contour[pitch_contour > 0]) if np.any(pitch_contour > 0) else 0,
            "pitch_std": np.std(pitch_contour[pitch_contour > 0]) if np.any(pitch_contour > 0) else 0,
            "energy_mean": np.mean(rms),
            "energy_std": np.std(rms),
            "speech_rate": speech_rate_proxy
        }

    def extract_spectral_entropy(self, audio):
        """
        Calculate spectral entropy (measure of peakiness/flatness of spectrum).
        High entropy -> Noise/Flat. Low entropy -> Tonal/Peaky.
        """
        # Power Spectrum
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))**2
        # Normalize to treat as probability distribution
        psd_norm = S / np.sum(S, axis=0, keepdims=True)
        # Entropy per frame
        entropy = scipy.stats.entropy(psd_norm, axis=0)
        return np.mean(entropy) if len(entropy) > 0 else 0
