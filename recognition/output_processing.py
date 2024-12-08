import os
from recognition.recognize_faces import recognize_faces_in_video, load_embeddings
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


def process_videos(input_dir, output_dir, embeddings_dir, device, confidence_threshold):
    """
    Process input videos, recognize faces, and save output videos in the same directory structure.

    Args:
        input_dir (str): Directory containing input videos.
        output_dir (str): Directory to save output annotated videos.
        embeddings_dir (str): Directory containing saved embeddings.
        device (torch.device): PyTorch device (CPU/GPU).
        confidence_threshold (float): Threshold for recognizing faces.
    """
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load embeddings
    embeddings = load_embeddings(embeddings_dir)

    # Process videos
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi','.MOV')):
                input_video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                output_video_path = os.path.join(output_subdir, file)
                recognize_faces_in_video(input_video_path, output_video_path, embeddings, device, mtcnn, model, confidence_threshold)
