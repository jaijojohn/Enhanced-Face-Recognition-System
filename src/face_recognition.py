import cv2
import numpy as np
import dlib
import torch
import time
import sys
import os
from datetime import datetime

# Add root directory to path to allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    TOLERANCE, MODELS_DIR,
    SAM_MODEL_TYPE, SAM_CHECKPOINT_PATH
)
from utils.face_detector import MTCNNDetector
from utils.face_encoder import FaceEncoder
from utils.occlusion_detector import OcclusionDetector
from segment_anything import sam_model_registry, SamPredictor

def draw_info_panel(frame, name, details, status, confidence, fps):
    """Draws the information panel at the bottom of the frame"""
    panel_height = 120
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
    
    # Color scheme
    colors = {
        "recognized": (0, 255, 0),
        "occluded": (0, 165, 255),
        "unknown": (0, 0, 255),
        "background": (45, 45, 45),
        "text": (255, 255, 255)
    }
    
    # Draw panel background
    cv2.rectangle(panel, (0, 0), (panel.shape[1], panel.shape[0]), 
                  colors["background"], -1)
    
    # Draw status indicator
    cv2.rectangle(panel, (10, 10), (30, 30), colors[status], -1)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    line_height = 25
    
    info_lines = [
        f"STATUS: {status.upper()}",
        f"NAME: {name}",
        f"DETAILS: {details}",
        f"CONFIDENCE: {confidence}",
        f"FPS: {fps:.1f}",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(panel, line, (40, y_offset + i*line_height), 
                    font, 0.5, colors["text"], 1)
    
    return np.vstack((frame, panel))

def main():
    # Initialize models
    print("[INFO] Loading models...")
    detector = MTCNNDetector()
    encoder = FaceEncoder()
    
    # Initialize SAM for occlusion detection (optional)
    use_occlusion = False  # Set to False to disable occlusion checks
    if use_occlusion:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=device)
        occlusion_detector = OcclusionDetector(SamPredictor(sam))
    else:
        occlusion_detector = None
    
    # Load known faces
    print("[INFO] Loading known face encodings...")
    known_encodings = np.load(MODELS_DIR / "encodings.npy")
    known_names = np.load(MODELS_DIR / "names.npy")
    known_details = np.load(MODELS_DIR / "details.npy")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # For FPS calculation
    prev_time = 0
    fps = 0
    
    print("[INFO] Starting recognition pipeline...")
    while True:
        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time)) if prev_time else 30
        prev_time = curr_time
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Default values when no face detected
        current_name = "No face detected"
        current_details = ""
        status = "unknown"
        confidence = "N/A"
        
        # Detect faces
        faces = detector.detect(frame)
        
        if faces:
            # Process first face only for simplicity
            (x1, y1, x2, y2) = faces[0]
            
            # Add 20% padding
            padding = int(0.2 * (x2 - x1))
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            # Get face encoding
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_box = dlib.rectangle(x1, y1, x2, y2)
            encoding = encoder.encode(rgb, face_box)
            
            # Find closest match
            distances = np.linalg.norm(known_encodings - encoding, axis=1)
            best_match = np.argmin(distances)
            min_distance = distances[best_match]
            
            # Simplified occlusion check (can be disabled)
            if use_occlusion and occlusion_detector:
                occlusion = occlusion_detector.check(rgb, [x1, y1, x2, y2])
            else:
                occlusion = {"partial": False}
            
            # Recognition logic with adjusted thresholds
            if min_distance < TOLERANCE and not occlusion["partial"]:
                status = "recognized"
                color = (0, 255, 0)  # Green
                current_name = known_names[best_match]
                current_details = known_details[best_match]
                confidence = f"{(1 - min_distance)*100:.1f}%"
            elif occlusion.get("partial", False):
                status = "occluded"
                color = (0, 165, 255)  # Orange
                current_name = "Unknown (occluded)"
                current_details = "Face partially covered"
                confidence = "N/A"
            else:
                status = "unknown"
                color = (0, 0, 255)  # Red
                current_name = "Unknown person"
                current_details = f"Nearest match: {known_names[best_match]}"
                confidence = f"{(1 - min_distance)*100:.1f}%"
            
            # Draw face bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence bar
            if confidence != "N/A":
                bar_width = int((1 - min_distance) * (x2 - x1))
                cv2.rectangle(frame, (x1, y2 + 5), 
                              (x1 + bar_width, y2 + 8), color, -1)
            
            # Add status text
            cv2.putText(frame, f"{status.upper()}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add information panel
        frame_with_info = draw_info_panel(
            frame, current_name, current_details, 
            status, confidence, fps
        )
        
        # Display
        cv2.imshow("Face Recognition System", frame_with_info)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()