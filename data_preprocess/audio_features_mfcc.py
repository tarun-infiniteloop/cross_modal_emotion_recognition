import os
import pandas as pd
import librosa
import numpy as np

# Path to the IEMOCAP full release dataset
iemocap_path = "/home/taruns/project/IEMOCAP_full_release"

# Path to the original CSV file
original_csv_path = "../iemocap.csv"

# Path to the new CSV file
new_csv_path = "../iemocap_features.csv"

# Path to the directory where MFCC features will be saved
features_dir = "../data/mfcc"
os.makedirs(features_dir, exist_ok=True)

# Read the original CSV file
df = pd.read_csv(original_csv_path)

# Function to extract MFCC features and save them to a file
def extract_and_save_mfcc(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    np.save(save_path, mfcc)
    return mfcc.shape[1], librosa.get_duration(y=y, sr=sr)

# Iterate over the rows in the DataFrame and process each audio file
feature_paths = []
durations = []
for index, row in df.iterrows():
    file_name = row['FileName']
    audio_path = os.path.join(iemocap_path, "Session" + file_name[4], "sentences", "wav", "_".join(file_name.split("_")[:-1]), file_name + ".wav")
    save_path = os.path.join(features_dir, file_name + ".npy")

    # Extract and save MFCC features
    num_frames, duration = extract_and_save_mfcc(audio_path, save_path)
    feature_paths.append(save_path)
    durations.append(duration)

    # Print progress
    print(f"Processed {index + 1}/{len(df)}: {file_name}")

# Add new columns to the DataFrame
df['FeaturePath'] = feature_paths
df['Duration'] = durations

# Save the new DataFrame to a CSV file
df.to_csv(new_csv_path, index=False)

print(f"Finished processing. New CSV file saved to {new_csv_path}")
