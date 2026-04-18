# Real-Time Scam Detection Android App

This is the barebones Android frontend for the Scam Call Detection System. 

## Features
- Captures microphone audio in real-time.
- Streams audio chunks over WebSockets to your Python FastAPI backend.
- Receives risk scores and Risk Labels (e.g. SAFE, HIGH RISK, SCAM ALERT).
- Dynamically updates the UI to alert the user.

## How to Run

1. **Open in Android Studio:**
   - Open Android Studio.
   - Click **File > Open**.
   - Select this folder: `d:\d_drive_project\Realtime_fraud_call_detection\diversion\AndroidApp`
   - Wait for Gradle to sync automatically.

2. **Start the FastAPI Backend:**
   - Open a terminal in `d:\d_drive_project\Realtime_fraud_call_detection\diversion`
   - Install dependencies if needed: `pip install fastapi uvicorn websockets`
   - Run the server: `python fastapi_server.py`
   - It will start listening on `0.0.0.0:8000`.

3. **Network Configuration (Already Done!):**
   - The app has been automatically configured to connect to your PC's actual local IP (`ws://192.168.31.224:8000/ws/predict`).
   - Just ensure your Android device / Emulator is connected to the **same Wi-Fi network** as your PC.

4. **Run the App:**
   - Click the green Run (Play) button in Android Studio.
   - Grant Audio permissions when prompted.
   - Click "Start Monitoring Call" to begin streaming audio to your backend.

## Advanced Usage Details
For actual GSM call recording, newer Android versions strictly restrict third-party apps from recording voice calls without being configured as a default dialer or system app. The current implementation captures using `VOICE_COMMUNICATION` which works perfectly for testing and speakerphone scenarios.
