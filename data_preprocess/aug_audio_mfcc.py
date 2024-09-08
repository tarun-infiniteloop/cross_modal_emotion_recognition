import os
import librosa
import numpy as np
from tqdm import tqdm

# Path to the directory containing the .wav files
audio_files_dir = "/raid/home/adityalk/sid/tarun/clip/data/Aug_audio_wav"

# Path to the directory where MFCC features will be saved
features_dir = "/raid/home/adityalk/sid/tarun/clip/data/mfcc_aug"
os.makedirs(features_dir, exist_ok=True)

# Target sampling rate
target_sr = 16000

# Function to extract MFCC features and save them to a file
def extract_and_save_mfcc(audio_path, save_path, target_sr):
    # Load the audio file with its original sampling rate
    y, sr = librosa.load(audio_path, sr=None)
    # Resample to the target sampling rate if necessary
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # Save the MFCC features as a .npy file
    np.save(save_path, mfcc)

# List all wav files
wav_files = [f for f in os.listdir(audio_files_dir) if f.endswith('.wav')]

# Process each file with a progress bar
for audio_file in tqdm(wav_files, desc="Processing audio files"):
    audio_path = os.path.join(audio_files_dir, audio_file)
    save_path = os.path.join(features_dir, audio_file.replace('.wav', '.npy'))
    
    # Extract and save MFCC features
    extract_and_save_mfcc(audio_path, save_path, target_sr)

print("All audio files have been processed.")
