import os
import csv
import torch
from src.utils import load_image_encodings

def save_embeddings(encodings, embeddings_dir):
    """
    Save embeddings to .pt files in the specified directory.
    :param encodings: Dictionary containing unique_id, name, and embeddings.
    :param embeddings_dir: Path to the directory for storing embeddings.
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    for unique_id, data in encodings.items():
        embedding_path = os.path.join(embeddings_dir, f"{unique_id}_embedding.pt")
        torch.save({'unique_id': unique_id, 'name': data['name'], 'embedding': data['embedding']}, embedding_path)

def load_embeddings(embeddings_dir):
    """
    Load embeddings from .pt files in the specified directory.
    :param embeddings_dir: Path to the directory containing embeddings.
    :return: Dictionary of embeddings loaded into memory.
    """
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
    """
    Compute embeddings from images based on the labels.csv.
    :param labels_csv_path: Path to the CSV file with unique_id and name.
    :param images_dir: Path to the directory containing face images.
    :return: Dictionary of computed embeddings.
    """
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
