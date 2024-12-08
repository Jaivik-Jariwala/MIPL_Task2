import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import pickle


def load_embeddings(embedding_pt_dir):
    """
    Load pre-computed facial embeddings from .pkl files like Jainaksh_embeddings.pkl and other.

    Args:
        embedding_pt_dir (str): Path to the directory containing .pkl files of embeddings.

    Returns:
        dict: A dictionary where keys are person names and values are their embeddings as numpy arrays.
    """

    embeddings = {}
    for file in os.listdir(embedding_pt_dir):
        if file.endswith('.pkl'):
            person_name = os.path.splitext(file)[0]
            with open(os.path.join(embedding_pt_dir, file), 'rb') as f:
                embeddings[person_name] = pickle.load(f)
    return embeddings


def recognize_faces_in_video(video_path, output_path, embeddings, device, mtcnn, model, confidence_threshold):
    """
    Recognize and annotate faces in a video based on pre-computed embeddings.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the annotated video.
        embeddings (dict): Dictionary of embeddings with person names as keys and numpy arrays as values.
        device (torch.device): Device to run the computations on (e.g., 'cpu' or 'cuda').
        mtcnn (MTCNN): Face detection model for detecting faces in frames.
        model (InceptionResnetV1): Pre-trained model to extract facial embeddings.
        confidence_threshold (float): Threshold for classifying a face as recognized or unknown.
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                try:
                    face = cv2.resize(face, (160, 160))
                    face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    face_tensor = face_tensor.to(device)
                    embedding = model(face_tensor).detach().cpu().numpy()

                    # Compare embeddings
                    distances = {name: np.sum((embedding - emb) ** 2) for name, emb in embeddings.items()}
                    best_match = min(distances, key=distances.get)
                    confidence = np.exp(-distances[best_match]) / sum(np.exp(-d) for d in distances.values())

                    if confidence > confidence_threshold:
                        label = f"{best_match} ({confidence:.2f})"
                    else:
                        label = "Unknown"

                    # Annotate video
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error processing face: {e}")

        video_writer.write(frame)

    cap.release()
    video_writer.release()
    print(f"Annotated video saved at {output_path}")
