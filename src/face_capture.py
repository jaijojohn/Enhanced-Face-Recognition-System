import cv2
import os
import json
import sys
from datetime import datetime

# Add root directory to path to allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import KNOWN_FACES_DIR

# Configuration
SAMPLES_PER_PERSON = 20  # Number of samples to capture
DATA_DIR = KNOWN_FACES_DIR
os.makedirs(DATA_DIR, exist_ok=True)

def capture_samples():
    # Get person details
    person_id = input("Enter Person ID (e.g., 001): ").strip()
    person_name = input("Enter Full Name: ").strip()
    person_details = input("Enter Details (e.g., position): ").strip()
    
    # Create person directory
    person_dir = f"{DATA_DIR}/{person_id}_{person_name}"
    os.makedirs(person_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "id": person_id,
        "name": person_name,
        "details": person_details,
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{person_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    print(f"Capturing {SAMPLES_PER_PERSON} samples for {person_name}...")
    print("Press SPACE to capture, ESC to finish early")
    
    sample_count = 0
    while sample_count < SAMPLES_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display instructions
        cv2.putText(frame, f"Sample {sample_count+1}/{SAMPLES_PER_PERSON}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | ESC: Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Capture Samples", frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_dir}/{timestamp}_{sample_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            sample_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Completed! {sample_count} samples saved to {person_dir}")

if __name__ == "__main__":
    capture_samples()