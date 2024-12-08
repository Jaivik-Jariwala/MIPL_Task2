import torch

CONFIG = {
    "dataset_dir": "Data/", 
    # Dataset containing person images
    
    "embeddings_dir": "embeddings/",  
    # Directory to store embeddings functions
    
    "embedding_pt_dir": "embedding_pt/", 
    # Directory to save embeddings
    
    "output_dir": "Output/", 
    # Output video path
    
    "input_video_dir": "Test/", 
    # Input video path
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    # Device configuration
    
    "confidence_threshold": 0.63, 
    # Threshold for face recognition
}
