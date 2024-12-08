# Face Recognition System

This is a face recognition system that processes images and videos to detect and recognize faces. The system uses deep learning models like **MTCNN** for face detection and **InceptionResnetV1** for embedding generation and recognition. It allows you to extract embeddings from a dataset of images and recognize faces in videos using these embeddings.

## Project Structure

```
project/
│
├── Data/
│   ├── Jainaksh/
│   │   ├── IMG_7814.JPG
│   │   ├── ...
│   ├── Umang/
│       ├── IMG_7802.JPG
│       ├── ...
│
├── embedding_pt/
│   ├── extract_embeddings.py
│   ├── save_embeddings.py
│
├── embeddings/
│   ├── Jainaksh_embeddings.pkl
│   ├── Umang_embeddings.pkl
│
├── Output/
│   ├── log.txt
│
├── recognition/
│   ├── recognize_faces.py
│   ├── output_processing.py
│
├── Test/
│   ├── Cam/
│   │   ├── Phone_AIGMS_testing.mp4
│   │   ├── ...
│   ├── CCTV/
│   │   ├── CCTV_Testing.mp4
│   │   ├── ...
│   ├── Person/
│       ├── jainaksh.mp4
│       ├── umang.mp4
│
├── config.py
├── main.py
├── output.py
└── README.md
```

### Directory and File Overview:

* **Data/** : Contains the images of the persons in the dataset. Each person's images are stored in separate subdirectories.
* **embedding_pt/** : Contains the scripts for extracting and saving face embeddings.
* **embeddings/** : Stores the precomputed face embeddings of each person.
* **Output/** : Stores the output log and any generated videos.
* **recognition/** : Contains the scripts for recognizing faces in videos and processing the output.
* **Test/** : Contains test videos (both individual and real-world scenarios like CCTV).
* **config.py** : Configuration file that contains paths and device settings.
* **main.py** : The main script that runs the entire process: extracting embeddings and recognizing faces in a video.
* **output.py** : Handles any additional output processing, such as logging or saving results.

## Setup and Installation

To run this project, ensure you have the following dependencies installed:

### Prerequisites:

* Python 3.x
* PyTorch (with GPU support, if available)
* OpenCV
* `facenet_pytorch`
* Other required libraries (`tqdm`, `PIL`, `numpy`)

### Install dependencies:

```bash
pip install torch torchvision facenet-pytorch opencv-python numpy tqdm Pillow
```

## Configuration

Before running the code, make sure to update the paths and parameters in the `config.py` file:

```python
CONFIG = {
    'dataset_dir': 'Data/',  # Directory containing person images
    'embeddings_dir': 'embeddings/',  # Directory to save embeddings
    'output_video_path': 'Output/output_video.mp4',  # Output video path
    'input_video_path': 'Test/Cam/Phone_AIGMS_testing.mp4',  # Input video path
    'annotated_video_path': 'Output/annotated_video.mp4',  # Annotated video output path
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device configuration (CPU or CUDA)
    'confidence_threshold': 0.5  # Threshold for face recognition confidence
}
```

## How to Use

### Step 1: Extract Face Embeddings from Images

To extract face embeddings from a set of images, run the following command:

```bash
python main.py
```

This will:

* Load images from the `Data/` directory.
* Use **MTCNN** to detect faces.
* Use **InceptionResnetV1** to generate embeddings for each face.
* Save the embeddings in the `embeddings/` directory.

### Step 2: Recognize Faces in Video

After extracting the embeddings, you can use the system to recognize faces in a video. The system will:

* Detect faces in the input video (`Test/Cam/Phone_AIGMS_testing.mp4`).
* Compare the embeddings of detected faces with the stored embeddings in the `embeddings/` directory.
* Annotate the video with the recognized person's name and confidence level.
* Save the annotated video to the `Output/` directory.

Run the following command to process the video:

```bash
python main.py
```

### Step 3: Process the Output

The output video with recognized faces will be saved in the `Output/` folder as `annotated_video.mp4`. You can also check the logs in `Output/log.txt` for details on the process.

## Code Explanation

1. **`main.py`** : This is the entry point of the program. It handles the extraction of embeddings from images and recognition of faces in videos.
2. **`config.py`** : Holds the configuration settings, including paths and device selection.
3. **`embedding_pt/extract_embeddings.py`** : Contains the logic for extracting face embeddings using **MTCNN** and  **InceptionResnetV1** .
4. **`recognition/recognize_faces.py`** : Implements the face recognition in video. It detects faces in video frames and compares their embeddings to identify the person.
5. **`recognition/output_processing.py`** : Handles any additional output-related operations like saving results or logging.

## Example Workflow

1. **Prepare Dataset** : Place images of individuals in the `Data/` directory. Each person should have their own subdirectory.
2. **Extract Embeddings** : Run the `main.py` script to extract embeddings from the images in the dataset.
3. **Test with Video** : Use the `Test/Cam/Phone_AIGMS_testing.mp4` video to test face recognition. The system will annotate the faces in the video and save the output.
4. **Output** : View the annotated video and logs in the `Output/` folder.
