import os
import cv2
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

def load_image_encodings(image_paths, base_dir=""):
    """Load face encodings from images."""
    encodings = []
    for image_path in image_paths:
        full_path = os.path.join(base_dir, image_path) if base_dir else image_path
        image = face_recognition.load_image_file(full_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodings.append(face_encodings[0])
    return encodings

def cosine_similarity_score(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0]



"""GPT BACKUP GENERATED"""

'''
class FaceRecognizerWithMetrics:
    def __init__(self, encodings, similarity_metric='cosine'):
        self.encodings = encodings
        self.similarity_metric = similarity_metric
        self.similarity_functions = {
            'cosine': cosine_similarity_score,
            'euclidean': euclidean_distance_score,
            'manhattan': manhattan_distance_score,
        }

    def recognize_face(self, face_encoding):
        """Recognize face using the selected similarity metric."""
        best_match = {"name": "UNKNOWN", "score": -1}
        similarity_function = self.similarity_functions.get(self.similarity_metric)
        
        if not similarity_function:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")

        for unique_id, data in self.encodings.items():
            similarity = similarity_function(face_encoding, data['embedding'])
            if similarity > best_match['score']:
                best_match = {"name": data['name'], "score": similarity}
        
        return best_match['name']

recognizer = FaceRecognizerWithMetrics(encodings, similarity_metric='euclidean')

# Use the recognizer to identify a face
name = recognizer.recognize_face(face_encoding)
print(f"Recognized as: {name}")


'''