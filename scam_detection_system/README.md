# Privacy-Preserving On-Device Scam Call Detection

An acoustic-only, real-time scam detection system designed for mobile devices. It detects "scam intent" using phoneme patterns, prosody, and script repetition unique to fraud calls, WITHOUT speech-to-text or cloud processing.

## 🚀 Key Features

*   **Privacy-First:** No recording, No ASR, No Transcription, No Cloud.
*   **Real-Time:** <20ms latency per 0.5s chunk.
*   **Lightweight:** Designed for <10MB model size (quantized).
*   **Language Agnostic:** Relies on intonation and phonetic structure, not words.

## 🛠️ Architecture

The system uses a 3-pronged approach running on a rolling audio buffer:

1.  **Phoneme Pattern Model (CRNN):** Detects "scam-like" phonetic sequences (e.g., "compromised", "urgent") using coarse phoneme groups.
2.  **Prosody Model (MLP):** Detects urgency, aggression, or unnatural pauses.
3.  **Repetition Detector:** Identifies scripted loops or repeated phrases using self-similarity matrices.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🏃 Usage

Run the simulation loop (generates synthetic audio for demonstration):

```bash
python main.py
```

## 📱 Mobile Deployment (TFLite)

To deploy on Android:
1.  Export PyTorch models to ONNX.
2.  Convert ONNX to TFLite.
3.  Apply Post-Training Quantization (Dynamic Range or INT8).

Example Conversion Plan:
- `PhonemePatternModel` -> `model_a.tflite`
- `ProsodyModel` -> `model_b.tflite`

## ⚖️ Privacy Guarantee

- **Input:** Raw Audio (discarded after 10s).
- **Features:** MFCCs (phase discarded), Prosody stats.
- **Output:** Risk Score (0-1).
- **No ID:** No speaker recognition or biomteric storage.
