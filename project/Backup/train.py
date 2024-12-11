import os
import csv
import torch
import numpy as np
from src.utils import load_image_encodings
from sklearn.metrics.pairwise import cosine_similarity

def save_embeddings(encodings, embeddings_dir):
    """Save embeddings to .pt files in the specified directory."""
    os.makedirs(embeddings_dir, exist_ok=True)
    for unique_id, data in encodings.items():
        embedding_path = os.path.join(embeddings_dir, f"{unique_id}_embedding.pt")
        torch.save({'unique_id': unique_id, 'name': data['name'], 'embedding': data['embedding']}, embedding_path)

def load_embeddings(embeddings_dir):
    """Load embeddings from .pt files in the specified directory."""
    encodings = {}
    for file_name in os.listdir(embeddings_dir):
        if file_name.endswith('_embedding.pt'):
            file_path = os.path.join(embeddings_dir, file_name)
            data = torch.load(file_path)
            encodings[data['unique_id']] = {
                'name': data['name'],
                'embedding': data['embedding']
            }
    return encodings

def compute_encodings_from_images(labels_csv_path, images_dir):
    """Compute embeddings from images."""
    encodings = {}
    with open(labels_csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            unique_id = row['unique_id']
            name = row['name']
            image_path = os.path.join(images_dir, f"{unique_id}.png")
            embedding = load_image_encodings([image_path], base_dir="")
            if embedding:
                encodings[unique_id] = {
                    'name': name,
                    'embedding': embedding[0]
                }
    return encodings
