import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "data" / "models"
KNOWN_FACES_DIR = BASE_DIR / "data" / "known_faces"

# Face Detection
FACE_DETECTOR = "mtcnn"  # Changed from "retinaface"
MIN_FACE_SIZE = 40       # Minimum pixel width (adjust as needed)
MIN_CONFIDENCE = 0.9     # Confidence threshold (0-1)

# Recognition
TOLERANCE = 0.5          # Lower = stricter recognition

# SAM (Occlusion)
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_PATH = MODELS_DIR / "sam_vit_h_4b8939.pth”