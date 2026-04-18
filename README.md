# 📞 Scam Call Detection System

Real-time AI-powered scam call detection system using multi-modal analysis of audio features, speech patterns, and conversation context.

The system combines an **Android mobile application (Kotlin)** with a **Python FastAPI backend** to detect scam calls live during phone conversations.

---

## 🎥 YouTube Pitch

▶️ Watch the demo video here: *(Add your YouTube link)*

---

# 🚀 Overview

This project uses a **4-model ensemble architecture** to detect scam calls in real time:

1. **Phoneme CNN** – Analyzes phonetic speech patterns  
2. **Urgency Detector (Prosody Model)** – Measures pitch, energy & speech rate anomalies  
3. **Repetition CNN** – Detects repetitive scam keyword patterns  
4. **Conversation Stage Tracker (Transformer-based)** – Tracks scam conversation progression  

The models work together to generate a continuous **risk score (0.0 – 1.0)**.

---

# 🏗️ System Architecture

Android App (Kotlin)
│
│ 4-second audio chunks (WebSocket)
▼
Python Backend (FastAPI)
│
▼
┌────────────────────────────┐
│ AI Ensemble Models │
│ - Phoneme CNN │
│ - Urgency Detector (MLP) │
│ - Repetition CNN │
│ - Stage Tracker (STT+NLP) │
└────────────────────────────┘
│
▼
Risk Score (0.0 - 1.0)
│
▼
User Alert System


---

# 🔥 Key Features

✅ Real-time 4-second sliding window analysis  
✅ Multi-model ensemble fusion  
✅ Live WebSocket audio streaming  
✅ Voice Activity Detection (VAD)  
✅ Conversation stage tracking  
✅ Continuous risk scoring  
✅ 5-level user alert system  

---

# 🚨 Risk Alert Levels

| Risk Score | Alert |
|------------|--------|
| < 0.15     | ✅ SAFE |
| 0.15–0.35  | 🟡 LOW RISK |
| 0.35–0.55  | 🟠 MODERATE |
| 0.55–0.80  | 🔴 HIGH RISK |
| > 0.80     | 🚨 SCAM ALERT |

---

# 📁 Project Structure
Scam-Call-Detection/
│
├── app/ # Android Application
│
├── Backend/ # FastAPI backend
│ ├── models/ # Trained AI models
│ ├── whisper.cpp/ # Speech-to-text engine
│ ├── newServer.py # FastAPI server
│ ├── feature_extraction.py
│ ├── audio_pipeline.py
│ └── config.py
│
├── dataset/
│ ├── NORMAL_CALLS/
│ └── SCAM_CALLS/
│
├── notebooks/ # Model training notebooks
│
└── training_config.json

---

# ⚙️ Backend Setup

### 1️⃣ Navigate to Backend
```bash
cd Backend

2️⃣ Install Dependencies
pip install -r requirements.txt
