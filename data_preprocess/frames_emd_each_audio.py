import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os
from PIL import Image

# Path to the model and preprocessor directories
saved_model_path = '../model'
saved_processor_path = '../processor'

processor = CLIPProcessor.from_pretrained(saved_processor_path)
clip_model = CLIPModel.from_pretrained(saved_model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model.to(device)

def batch_process_images(image_paths, batch_size, processor, model, device):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        tokens = processor(
            text=None,
            images=batch_images,
            return_tensors="pt"
        ).to(device)
        batch_embeddings = model.get_image_features(**tokens)
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

main_folder = '../data/frames_each_audio'
output_folder = '../data/frames_emd_each_audio'
os.makedirs(output_folder, exist_ok=True)

subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
total_subfolders = len(subfolders)
processed_subfolders = 0

for subfolder in subfolders:
    image_paths = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if image_paths:
        # Create embeddings for images that don't have corresponding embeddings in the output folder
        new_image_paths = []
        subfolder_name = os.path.basename(subfolder)
        output_subfolder = os.path.join(output_folder, subfolder_name)
        os.makedirs(output_subfolder, exist_ok=True)
        for i, image_path in enumerate(image_paths):
            output_path = os.path.join(output_subfolder, f'embedding_{i}.npy')
            if not os.path.exists(output_path):
                new_image_paths.append(image_path)

        if new_image_paths:
            embeddings = batch_process_images(new_image_paths, batch_size=500, processor=processor, model=clip_model, device=device)
            for i, emb in enumerate(embeddings):
                output_path = os.path.join(output_subfolder, f'embedding_{i}.npy')
                np.save(output_path, emb)

    processed_subfolders += 1
    print(f"Processed {processed_subfolders}/{total_subfolders} subfolders.")

print("All subfolders processed.")
