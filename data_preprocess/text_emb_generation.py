import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and move it to the device
model = SentenceTransformer('all-mpnet-base-v2').to(device)

# Read the CSV file
df = pd.read_csv('../iemocap.csv')

# Create the output directory if it doesn't exist
output_dir = '../data/sentence_emd_full'
os.makedirs(output_dir, exist_ok=True)

# Iterate over the rows in the dataframe
for index, row in df.iterrows():
    sentence = row['Sentences']
    file_name = row['FileName']
    
    # Generate the embedding
    embedding = model.encode(sentence, convert_to_tensor=True, device=device)
    
    # Move the embedding back to CPU for saving
    embedding = embedding.cpu().numpy()
    
    # Save the embedding to a file
    np.save(os.path.join(output_dir, file_name + '.npy'), embedding)

    # Optional: Print the progress
    print(f'Processed {index + 1} out of {len(df)} sentences.')

print('Embeddings generation and saving completed.')