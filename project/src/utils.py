import face_recognition
import os

def load_image_encodings(image_paths, base_dir=""):
    """
    Load face encodings from a list of image paths.
    :param image_paths: List of image paths to load and encode.
    :param base_dir: Base directory to prepend to image paths.
    :return: List of encodings.
    """
    encodings = []
    for image_path in image_paths:
        full_path = os.path.join(base_dir, image_path) if base_dir else image_path
        image = face_recognition.load_image_file(full_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodings.append(face_encodings[0])
    return encodings
