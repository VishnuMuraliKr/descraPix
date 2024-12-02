import os
import torch
from PIL import Image
from transformers import CLIPProcessor
from torch.nn.functional import cosine_similarity

def preprocess_image(image_path, processor):    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

def compute_similarity(model, processor, description, image_folder):

    #description preprocessing
    inputs = processor(text=description, return_tensors="pt", padding=True)
    text_embedding = model.get_text_features(**inputs)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    results = []
    
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_tensor = preprocess_image(file_path, processor)
            with torch.no_grad():
                image_embedding = model.get_image_features(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                similarity = cosine_similarity(text_embedding, image_embedding).item()
            results.append((filename, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)
