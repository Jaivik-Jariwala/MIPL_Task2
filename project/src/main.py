from src.config import LABELS_CSV_PATH, IMAGES_DIR, EMBEDDINGS_DIR
from src.train import compute_encodings_from_images, save_embeddings, load_embeddings
from src.model import FaceRecognizer

def main():
    # Step 1: Compute or load embeddings
    if not os.listdir(EMBEDDINGS_DIR):  # Check if embeddings directory is empty
        print("Embeddings not found. Computing and saving embeddings...")
        encodings = compute_encodings_from_images(LABELS_CSV_PATH, IMAGES_DIR)
        save_embeddings(encodings, EMBEDDINGS_DIR)
    else:
        print("Loading existing embeddings...")
        encodings = load_embeddings(EMBEDDINGS_DIR)

    # Step 2: Initialize FaceRecognizer
    recognizer = FaceRecognizer(encodings)

    # Step 3: Process video (e.g., detect faces in a video file)
    print("Processing video...")
    # Your video processing code here...

if __name__ == "__main__":
    main()
