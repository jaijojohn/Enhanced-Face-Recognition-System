from mtcnn import MTCNN
import cv2

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()  # Remove unsupported parameters
    
    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        
        faces = []
        for res in results:
            if res['confidence'] > 0.9:  # Confidence threshold
                x, y, w, h = res['box']
                faces.append((x, y, x+w, y+h))
        return faces