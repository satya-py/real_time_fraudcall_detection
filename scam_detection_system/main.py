import numpy as np
import torch
import time
from buffer import RollingBuffer
from preprocessor import AudioPreprocessor
from features import FeatureExtractor
from models import PhonemePatternModel, ProsodyModel, RepetitionDetector, RiskFusionEngine
import os
import sys
import soundfile as sf
from collections import deque
import numpy as np

def main():
    # Audio Target setup
    audio_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\d_drive_project\Realtime_fraud_call_detection\scam_srijan.wav"
    if not os.path.exists(audio_path):
        print(f"Error: Target file {audio_path} does not exist.")
        return
        
    if os.path.getsize(audio_path) == 0:
        print(f"\n--- ERROR ---")
        print(f"File {os.path.basename(audio_path)} is mathematically empty (0 bytes)! This is a corrupted dataset file.")
        print("Please test on a different .wav file with actual audio data.")
        return
        
    try:
        audio, sr = sf.read(audio_path)
        audio = audio.astype(np.float32)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return

    print(f"Evaluating {os.path.basename(audio_path)} ...")

    # Configuration
    SAMPLE_RATE = 16000
    BUFFER_DURATION = 10 # seconds
    CHUNK_DURATION = 0.5 # seconds
    CHUNK_SIZE = int(CHUNK_DURATION * SAMPLE_RATE)
    RISK_WINDOW = 20              # try 10–30
    risk_buffer = deque(maxlen=RISK_WINDOW)
    
    # Initialize components
    buffer = RollingBuffer(BUFFER_DURATION, SAMPLE_RATE)
    preprocessor = AudioPreprocessor(SAMPLE_RATE)
    feature_extractor = FeatureExtractor(SAMPLE_RATE)
    
    # Initialize models (Dummy weights for now)
    phoneme_model = PhonemePatternModel()
    prosody_model = ProsodyModel(input_dim=5) # 5 prosody features
    repetition_detector = RepetitionDetector()
    fusion_engine = RiskFusionEngine()
    
    phoneme_model.eval()
    prosody_model.eval()
    
    # Load trained models if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    phoneme_model_path = os.path.join(script_dir, "phoneme_model.pth")
    prosody_model_path = os.path.join(script_dir, "prosody_model.pth")
    if os.path.exists(phoneme_model_path):
        phoneme_model.load_state_dict(torch.load(phoneme_model_path, weights_only=True))
        print("Loaded trained weights for PhonemePatternModel.")
    if os.path.exists(prosody_model_path):
        prosody_model.load_state_dict(torch.load(prosody_model_path, weights_only=True))
        print("Loaded trained weights for ProsodyModel.")
    
    print("Scam Detection System Initialized.")
    print("Simulating real-time audio stream...")
    
    # Simulation Loop
    try:
        # Determine total possible frames from loaded audio
        total_frames = int(np.ceil(len(audio) / CHUNK_SIZE))
        if total_frames > 30: total_frames = 30 # cap at 15 seconds
        
        for i in range(total_frames): # Loop through real chunks
            # Extract the correct chunk
            start_sample = i * CHUNK_SIZE
            end_sample = (i + 1) * CHUNK_SIZE
            if start_sample >= len(audio):
                print(f"Target evaluations completed.")
                break
                
            raw_audio = audio[start_sample:end_sample]

            # 2. Preprocessing
            clean_audio = preprocessor.process(raw_audio)
            
            # 3. Buffer Update
            buffer.add_chunk(clean_audio)
            
            # 4. Check VAD (Process only if speech is present in last N seconds)
            # For this demo, we assume the buffer has enough speech
            current_buffer = buffer.get_buffer()
            
            if not preprocessor.is_speech(clean_audio):
                print(f"Frame {i}: Silence detected. Skipping.")
                continue
                
            # Wait for at least 2 seconds of audio buffer before processing 
            # to match the 2-second sequence length trained into our PyTorch models
            if len(current_buffer) < SAMPLE_RATE * 2:
                print(f"Frame {i}: Buffering... ({len(current_buffer)/SAMPLE_RATE:.1f}s collected)")
                continue
                
            # 5. Feature Extraction
            # Extract features from the buffer (or last window)
            try:
                mfccs = feature_extractor.extract_mfcc(current_buffer)
                if mfccs.shape[0] == 0:
                    print(f"Frame {i}: Insufficient audio for MFCC. Skipping.")
                    continue
                
                prosody = feature_extractor.extract_prosody(current_buffer)
                
                # 6. Model Inference
                # Prepare tensors
                # MFCC shape: (Time, Channels). 
                # Conv1d expects (Batch, Channels, Time).
                mfcc_tensor = torch.tensor(mfccs).transpose(0, 1).unsqueeze(0).float()
                
                # Prosody vector
                prosody_vec = torch.tensor([
                    prosody['pitch_mean'], prosody['pitch_std'], 
                    prosody['energy_mean'], prosody['energy_std'],
                    prosody['speech_rate']
                ]).unsqueeze(0).float()
                
                with torch.no_grad():
                    risk_a = phoneme_model(mfcc_tensor).item()
                    risk_b = prosody_model(prosody_vec).item() 
                    
                # Repetition (Algorithmic)
                # Compute score expects (Features, Time)
                risk_c = repetition_detector.compute_score(mfccs.T)
                
                # 7. Fusion
                total_risk = fusion_engine.fuse([risk_a, risk_b, risk_c])
                
                # 8. Decision
                alert_level = "LOW"
                if total_risk > 0.7: alert_level = "HIGH (SCAM)"
                elif total_risk > 0.3: alert_level = "NORMAL "
                
                print(f"Frame {i} | Risk: {total_risk:.2f} | A: {risk_a:.2f} B: {risk_b:.2f} C: {risk_c:.2f} | Alert: {alert_level}")
            
            except Exception as e:
                print(f"Frame {i}: Error processing chunk - {e}")
                continue
            
            time.sleep(0.1) # Simulate processing time

    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()




