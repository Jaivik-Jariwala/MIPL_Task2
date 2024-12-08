from embeddings.extract_embeddings import extract_embeddings
from recognition.output_processing import process_videos
from config import CONFIG


if __name__ == "__main__":
    # Step 1: Extract embeddings
    extract_embeddings(CONFIG["dataset_dir"], CONFIG["embedding_pt_dir"], CONFIG["device"])

    # Step 2: Process videos and save outputs
    process_videos(CONFIG["input_video_dir"], CONFIG["output_dir"], CONFIG["embedding_pt_dir"], CONFIG["device"], CONFIG["confidence_threshold"])
