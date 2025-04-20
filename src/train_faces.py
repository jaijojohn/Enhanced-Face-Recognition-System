import os
import cv2
import json
import numpy as np
import dlib  # Must import dlib before use
import sys

# Add root directory to path to allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import KNOWN_FACES_DIR, MODELS_DIR
from utils.face_detector import MTCNNDetector
from utils.face_encoder import FaceEncoder

def train():
    print("[INFO] Initializing models...")
    detector = MTCNNDetector()
    encoder = FaceEncoder()
    encodings = []
    names = []
    details = []

    print("[INFO] Processing known faces...")
    for person_dir in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_dir)
        if not os.path.isdir(person_path):
            continue

        # Load metadata
        with open(os.path.join(person_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)

        for image_file in os.listdir(person_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Failed to load {image_file}")
                continue

            # Detect faces
            faces = detector.detect(image)
            if not faces:
                print(f"[WARNING] No faces in {image_file}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for (x1, y1, x2, y2) in faces:
                face_box = dlib.rectangle(x1, y1, x2, y2)
                encoding = encoder.encode(rgb, face_box)
                encodings.append(encoding)
                names.append(metadata["name"])
                details.append(metadata["details"])

    # Save data
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(MODELS_DIR / "encodings.npy", np.array(encodings))
    np.save(MODELS_DIR / "names.npy", np.array(names))
    np.save(MODELS_DIR / "details.npy", np.array(details))
    
    print(f"[SUCCESS] Trained on {len(encodings)} face samples")

if __name__ == "__main__":
    train()