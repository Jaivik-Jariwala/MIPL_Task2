import numpy as np
from backup.utils import cosine_similarity_score
from backup.config import COSINE_THRESHOLD

class CosineFaceRecognizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def recognize_face(self, face_encoding):
        """Recognize face using cosine similarity."""
        best_match = {"name": "UNKNOWN", "score": -1}
        for unique_id, data in self.encodings.items():
            similarity = cosine_similarity_score(face_encoding, data['embedding'])
            if similarity > best_match['score'] and similarity >= COSINE_THRESHOLD:
                best_match = {"name": data['name'], "score": similarity}
        return best_match['name']
