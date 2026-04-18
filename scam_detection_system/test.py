import sys
import numpy as np
import soundfile as sf
import torch
import warnings
warnings.filterwarnings('ignore')

from preprocessor import AudioPreprocessor
from features import FeatureExtractor
from models import RepetitionDetector, PhonemePatternModel, ProsodyModel

try:
    pa = AudioPreprocessor(16000)
    fe = FeatureExtractor(16000)
    rd = RepetitionDetector()
    phoneme_model = PhonemePatternModel()
    prosody_model = ProsodyModel(input_dim=5)
    
    phoneme_model.load_state_dict(torch.load('phoneme_model.pth', weights_only=True))
    prosody_model.load_state_dict(torch.load('prosody_model.pth', weights_only=True))
    phoneme_model.eval()
    prosody_model.eval()
    
    for label, p in [('NORMAL', r'D:\d_drive_project\Realtime_fraud_call_detection\processed_dataset\NORMAL_CALLS\normal_10.wav'), ('SCAM', r'D:\d_drive_project\Realtime_fraud_call_detection\processed_dataset\SCAM_CALLS\scam_11.wav')]:
        print(f"\n--- {label} ---")
        a, _ = sf.read(p)
        a = a.astype('float32') # whole file
        ca = pa.process(a)
        
        # simulate buffer growing
        buffer = []
        for i in range(1, 10):
            sz = i * 8000 # 0.5s -> 4.5s
            if sz > len(ca): break
            chunk = ca[:sz]
            
            mfcc = fe.extract_mfcc(chunk)
            score_c = rd.compute_score(mfcc.T)
            
            mfcc_tensor = torch.tensor(mfcc).transpose(0, 1).unsqueeze(0).float()
            
            prosody = fe.extract_prosody(chunk)
            prosody_vec = torch.tensor([
                prosody['pitch_mean'], prosody['pitch_std'], 
                prosody['energy_mean'], prosody['energy_std'],
                prosody['speech_rate']
            ]).unsqueeze(0).float()
            
            with torch.no_grad():
                score_a = phoneme_model(mfcc_tensor).item()
                score_b = prosody_model(prosody_vec).item()
            
            print(f'LEN: {i*0.5}s - A: {score_a:.2f}, B: {score_b:.2f}, C: {score_c:.2f}')

except Exception as e:
    print(f"Error: {e}")
