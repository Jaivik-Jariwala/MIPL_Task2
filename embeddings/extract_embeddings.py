import os
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pickle
from embeddings.save_embeddings import save_embedding


def extract_embeddings(dataset_dir, embeddings_dir, device):
    """
    Extract embeddings for images in the dataset and save them.

    Args:
        dataset_dir (str): Directory containing images of persons.
        embeddings_dir (str): Directory to save embeddings.
        device (torch.device): PyTorch device (CPU/GPU).
    """
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing {person_name}...")
        embeddings = []
        for img_name in tqdm(os.listdir(person_dir), desc=f"Processing {person_name}"):
            img_path = os.path.join(person_dir, img_name)
            try:
                image = Image.open(img_path)
                face, _ = mtcnn(image, return_prob=True)
                if face is not None:
                    face_tensor = face[0].unsqueeze(0).to(device)
                    embedding = model(face_tensor).detach().cpu().numpy()
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        save_embedding(person_name, embeddings, embeddings_dir)
