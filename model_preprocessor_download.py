# This file is to download the model and preprocessor for the CLIP model.
# GPU cluster don't have internet access, so we need to download the model and preprocessor in our local machine and then upload it to the cluster.

from transformers import CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# Save the model and processor
processor.save_pretrained('./processor')
model.save_pretrained('./model')