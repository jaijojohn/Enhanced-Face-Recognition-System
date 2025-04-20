# Enhanced Face Recognition System

A facial recognition system with occlusion detection capabilities.

## Project Structure

```
Enhanced-Face-Recognition-System/
│
├── config.py                    # Configuration settings
│
├── data/                        # Data directory
│   ├── known_faces/             # Captured face images
│   └── models/                  # Trained models and encodings
│
├── src/                         # Source code
│   ├── utils/                   # Utility modules
│   │   ├── face_detector.py     # Face detection functionality
│   │   ├── face_encoder.py      # Face encoding functionality
│   │   └── occlusion_detector.py # Occlusion detection
│   │
│   ├── face_capture.py          # Face capture functionality
│   ├── train_faces.py           # Training functionality
│   └── face_recognition.py      # Main recognition application
```

## Components

- **face_detector.py**: Handles face detection using MTCNN
- **face_encoder.py**: Encodes detected faces using dlib's face recognition model
- **occlusion_detector.py**: Detects occlusions in faces using SAM (Segment Anything Model)
- **face_capture.py**: Utility for capturing and storing face samples
- **train_faces.py**: Processes known faces and generates encodings
- **face_recognition.py**: Main application for real-time face recognition

## Usage

1. Capture face samples: `python src/face_capture.py`
2. Train the system: `python src/train_faces.py`
3. Run face recognition: `python src/face_recognition.py`
