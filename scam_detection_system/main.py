import numpy as np
import torch
import time
from buffer import RollingBuffer
from preprocessor import AudioPreprocessor
from features import FeatureExtractor
from models import PhonemePatternModel, ProsodyModel, RepetitionDetector, RiskFusionEngine

def main():
    # Configuration
    SAMPLE_RATE = 16000
    BUFFER_DURATION = 10 # seconds
    CHUNK_DURATION = 0.5 # seconds
    CHUNK_SIZE = int(CHUNK_DURATION * SAMPLE_RATE)
    
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
    
    print("Scam Detection System Initialized.")
    print("Simulating real-time audio stream...")
    
    # Simulation Loop
    try:
        for i in range(20): # Simulate 10 seconds (20 chunks)
            # 1. Simulate Audio Chunk (Random noise for demo)
            # In real usage, this comes from mic stream
            # Generate synthetic 'speech-like' noise
            raw_audio = np.random.normal(0, 0.1, CHUNK_SIZE).astype(np.float32)
            
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
                elif total_risk > 0.3: alert_level = "MEDIUM (SUSPICIOUS)"
                
                print(f"Frame {i} | Risk: {total_risk:.2f} | A: {risk_a:.2f} B: {risk_b:.2f} C: {risk_c:.2f} | Alert: {alert_level}")
            
            except Exception as e:
                print(f"Frame {i}: Error processing chunk - {e}")
                continue
            
            time.sleep(0.1) # Simulate processing time

    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
