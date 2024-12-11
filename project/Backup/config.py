import os

# Define paths for backup operations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV_PATH = os.path.join(BASE_DIR, '../dataset/labels.csv')
IMAGES_DIR = os.path.join(BASE_DIR, '../dataset/images')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, '../embeddings')
VIDEOS_DIR = os.path.join(BASE_DIR, '../videos')
BACKUP_OUTPUT_DIR = os.path.join(BASE_DIR, '../backup_output')

# Threshold for cosine similarity
COSINE_THRESHOLD = 0.6

# Ensure backup output directory exists
os.makedirs(BACKUP_OUTPUT_DIR, exist_ok=True)
