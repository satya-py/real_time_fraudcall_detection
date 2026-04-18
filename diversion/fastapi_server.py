import asyncio
import json
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI(title="Scam Call Detection API")

@app.get("/")
def read_root():
    return {"status": "Real-Time Scam Detection API is running. Connect via WebSocket to /ws/predict."}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("📱 Mobile App Connected for Audio Streaming!")
    
    chunk_count = 0
    try:
        while True:
            # Receive audio chunk (bytes) from Android App
            data = await websocket.receive_bytes()
            chunk_count += 1
            
            # This is where the inference pipeline processes the 4-second audio chunk
            # e.g., features = extract_features(data)
            # prediction = model.predict(features)
            
            # Since this is a setup scaffolding, we simulate the inference
            # We send a result periodically to simulate the 4 models prediction
            if chunk_count % 16 == 0:
                risk_score = random.uniform(0.0, 1.0)
                
                if risk_score < 0.3:
                    label = "SAFE"
                elif risk_score < 0.6:
                    label = "LOW RISK"
                elif risk_score < 0.8:
                    label = "MODERATE RISK"
                elif risk_score < 0.95:
                    label = "HIGH RISK"
                else:
                    label = "SCAM ALERT"

                result = {
                    "risk_score": round(risk_score, 4),
                    "risk_label": label
                }
                
                print(f"[{chunk_count} chunks received] Analyzed audio. Sending prediction: {result['risk_label']} ({result['risk_score']})")
                await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("📱 Mobile App Disconnected.")
    except Exception as e:
        print(f"Error during WebSocket streaming: {e}")

if __name__ == "__main__":
    # Host on 0.0.0.0 to allow mobile devices on the local network (like 192.168.x.x) to connect
    print("🚀 Starting FastAPI Server for Scam Call Detection...")
    print("Ensure your mobile device is on the same WiFi network.")
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True)
