import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '../dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_CSV_PATH = os.path.join(DATASET_DIR, 'labels.csv')
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, '../videos/input_video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, '../videos/output_video.mp4')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, '../embeddings')

