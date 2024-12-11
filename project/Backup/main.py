from backup.config import LABELS_CSV_PATH, IMAGES_DIR, EMBEDDINGS_DIR, BACKUP_OUTPUT_DIR, VIDEOS_DIR
from backup.train import compute_encodings_from_images, save_embeddings, load_embeddings
from backup.model import CosineFaceRecognizer
from backup.video_processor import process_video
import os

def main():
    embeddings_dir = EMBEDDINGS_DIR
    encodings = {}

    if not os.listdir(embeddings_dir):  # If embeddings folder is empty
        print("Computing embeddings...")
        encodings = compute_encodings_from_images(LABELS_CSV_PATH, IMAGES_DIR)
        save_embeddings(encodings, embeddings_dir)
    else:
        print("Loading embeddings...")
        encodings = load_embeddings(embeddings_dir)

    recognizer = CosineFaceRecognizer(encodings)

    input_video_path = os.path.join(VIDEOS_DIR, 'input_video.mp4')
    output_video_path = os.path.join(BACKUP_OUTPUT_DIR, 'output_video.mp4')

    print("Processing video...")
    process_video(input_video_path, output_video_path, recognizer)
    print(f"Video processing complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    main()
