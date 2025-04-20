import numpy as np
from segment_anything import SamPredictor

class OcclusionDetector:
    def __init__(self, sam_predictor):
        self.sam = sam_predictor

    def check(self, image, face_box):
        """Check for face occlusions
        Args:
            face_box: Should be [x1, y1, x2, y2] as numpy array
        """
        # Convert to numpy array and reshape
        box_np = np.array(face_box).reshape(2, 2)  # Shape: (2,2)
        
        # Get SAM mask
        self.sam.set_image(image)
        masks, _, _ = self.sam.predict(box=box_np)
        mask = masks[0]  # Get the first mask
        
        # Define regions
        h, w = mask.shape
        return {
            'eyes': np.mean(mask[:h//3, :]) < 0.4,  # Upper 1/3
            'mouth': np.mean(mask[2*h//3:, :]) < 0.4,  # Lower 1/3
            'partial': np.mean(mask) < 0.7  # Overall coverage
        }