import numpy as np
import scipy.signal

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Bandpass filter for telephony speech (300-3400 Hz)
        self.b, self.a = scipy.signal.butter(4, [300, 3400], btype='band', fs=sample_rate)

    def apply_bandpass(self, audio):
        return scipy.signal.lfilter(self.b, self.a, audio)

    def compute_energy(self, audio):
        return np.mean(audio**2)

    def is_speech(self, audio, threshold=0.001):
        """
        Simple energy-based VAD.
        For production, a more robust VAD (e.g., WebRTC VAD) is recommended.
        """
        energy = self.compute_energy(audio)
        return energy > threshold

    def normalize(self, audio):
        """
        Normalize audio to -1 to 1 range, handling silence.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def process(self, audio):
        """
        Full preprocessing pipeline.
        """
        # 1. Bandpass
        filtered = self.apply_bandpass(audio)
        # 2. Normalize (optional, better done at feature extraction?
        # typically we want relative energy, so maybe don't normalize magnitude completely unless it's for spectral shape)
        # Let's keep energy info but avoid clipping.
        return filtered
