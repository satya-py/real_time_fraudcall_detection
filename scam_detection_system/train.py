import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import numpy as np

from models import PhonemePatternModel, ProsodyModel
from features import FeatureExtractor
from preprocessor import AudioPreprocessor

def load_data(data_dir, preprocessor, feature_extractor):
    mfcc_data = []
    prosody_data = []
    labels = []
    
    # Process Normal (0) and Scam (1)
    classes = {"NORMAL_CALLS": 0, "SCAM_CALLS": 1}
    for class_name, label in classes.items():
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found.")
            continue
            
        for filepath in glob.glob(os.path.join(folder_path, "*.wav")):
            try:
                audio, sr = sf.read(filepath)
                # Ensure 16kHz
                if sr != preprocessor.sample_rate:
                    pass 
                audio = audio.astype(np.float32)
                clean_audio = preprocessor.process(audio)
                
                # Split into 2-second chunks (32000 samples)
                chunk_samples = preprocessor.sample_rate * 2
                
                # We need some minimum length
                if len(clean_audio) < chunk_samples:
                    continue
                    
                for start_idx in range(0, len(clean_audio) - chunk_samples + 1, chunk_samples):
                    chunk = clean_audio[start_idx : start_idx + chunk_samples]
                    
                    mfccs = feature_extractor.extract_mfcc(chunk)
                    if mfccs.shape[0] == 0:
                        continue
                        
                    prosody = feature_extractor.extract_prosody(chunk)
                    prosody_vec = [
                        prosody['pitch_mean'], prosody['pitch_std'],
                        prosody['energy_mean'], prosody['energy_std'],
                        prosody['speech_rate']
                    ]
                    
                    # mfcc shape from extractor: (Time, n_mfcc)
                    # Model expects (Channels, Time) for single item before unsqueeze
                    mfcc_data.append(torch.tensor(mfccs).T.float()) 
                    prosody_data.append(torch.tensor(prosody_vec).float())
                    labels.append(label)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                
    return mfcc_data, prosody_data, labels

def train_models():
    sample_rate = 16000
    preprocessor = AudioPreprocessor(sample_rate)
    feature_extractor = FeatureExtractor(sample_rate)
    
    data_dir = r"D:\d_drive_project\Realtime_fraud_call_detection\processed_dataset"
    print("Loading data...")
    mfcc_data, prosody_data, labels = load_data(data_dir, preprocessor, feature_extractor)
    
    if not labels:
        print("No valid data found to train.")
        return
        
    print(f"Loaded {len(labels)} samples.")
    
    # Initialize models
    phoneme_model = PhonemePatternModel()
    prosody_model = ProsodyModel(input_dim=5)
    
    criterion = nn.BCELoss()
    optimizer_p = optim.Adam(phoneme_model.parameters(), lr=0.001)
    optimizer_pr = optim.Adam(prosody_model.parameters(), lr=0.001)
    
    from torch.utils.data import TensorDataset, DataLoader

    epochs = 15
    print("Training PhonemePatternModel & ProsodyModel...")
    
    # Pre-stack all data for full batch training
    M = torch.stack(mfcc_data)
    P = torch.stack(prosody_data)
    Y = torch.tensor(labels).float().unsqueeze(1)
    
    dataset = TensorDataset(M, P, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        phoneme_model.train()
        prosody_model.train()
        
        total_p_loss = 0
        total_pr_loss = 0
        
        for m_batch, p_batch, y_batch in loader:
            # Train PhonemePatternModel
            optimizer_p.zero_grad()
            out_p = phoneme_model(m_batch)
            loss_p = criterion(out_p, y_batch)
            loss_p.backward()
            optimizer_p.step()
            total_p_loss += loss_p.item() * m_batch.size(0)
            
            # Train ProsodyModel
            optimizer_pr.zero_grad()
            out_pr = prosody_model(p_batch)
            loss_pr = criterion(out_pr, y_batch)
            loss_pr.backward()
            optimizer_pr.step()
            total_pr_loss += loss_pr.item() * p_batch.size(0)
            
        print(f"Epoch {epoch+1}/{epochs} - Phoneme Loss: {total_p_loss/len(labels):.4f}, Prosody Loss: {total_pr_loss/len(labels):.4f}")
        
    # Save weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(phoneme_model.state_dict(), os.path.join(script_dir, "phoneme_model.pth"))
    torch.save(prosody_model.state_dict(), os.path.join(script_dir, "prosody_model.pth"))
    print("Models saved: phoneme_model.pth, prosody_model.pth")

if __name__ == '__main__':
    train_models()
