import os
# Set environment variable for PyTorch 2.6+ compatibility
os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = '1'

from ultralytics import YOLO
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List

# Configure Tesseract path for macOS Homebrew installation
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

class SafetyComplianceDetector:
    def __init__(self):
        """Initialize YOLO model - yolov8n.pt auto-downloads on first run"""
        self.model = YOLO('yolov8n.pt')
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image for safety compliance
        Returns: {
            'people_count': int,
            'signage_text': str,
            'violations': List[str],
            'detections': List[Dict],
            'compliance_score': int (0-100)
        }
        """
        # 1. Run YOLO detection
        results = self.model(image_path, conf=0.5)
        
        # 2. Extract person detections
        people_count = 0
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Class 0 = person in COCO dataset
                if class_id == 0:
                    people_count += 1
                    coords = box.xyxy[0].tolist()
                    detections.append({
                        'type': 'person',
                        'confidence': round(confidence, 2),
                        'bbox': [round(c, 2) for c in coords]
                    })
        
        # 3. Run Tesseract OCR
        img = Image.open(image_path)
        try:
            signage_text = pytesseract.image_to_string(img).strip()
        except Exception as e:
            # If Tesseract is not installed, return empty string
            print(f"Warning: Tesseract OCR not available: {e}")
            signage_text = ""
        
        # 4. Compliance analysis
        violations = self._check_compliance(people_count, signage_text)
        compliance_score = self._calculate_score(people_count, signage_text, violations)
        
        return {
            'people_count': people_count,
            'signage_text': signage_text,
            'violations': violations,
            'detections': detections,
            'compliance_score': compliance_score
        }
    
    def _check_compliance(self, people_count: int, signage_text: str) -> List[str]:
        """Check for compliance violations based on signage and detections"""
        violations = []
        text_upper = signage_text.upper()
        
        # Check for various safety requirements
        if people_count > 0:
            if any(keyword in text_upper for keyword in ['HARD HAT', 'HELMET', 'HEAD PROTECTION']):
                violations.append(f'Hard hat required zone: {people_count} worker(s) detected. Manual verification needed.')
            
            if any(keyword in text_upper for keyword in ['HIGH VISIBILITY', 'HI-VIS', 'REFLECTIVE']):
                violations.append(f'High-visibility clothing required: {people_count} worker(s) detected. Manual verification needed.')
            
            if any(keyword in text_upper for keyword in ['AUTHORIZED', 'RESTRICTED', 'NO ENTRY']):
                violations.append(f'Restricted area: {people_count} unauthorized person(s) may be present.')
            
            if any(keyword in text_upper for keyword in ['DANGER', 'HAZARD', 'CAUTION']):
                violations.append(f'Hazard zone: {people_count} worker(s) in potentially dangerous area.')
        
        if not signage_text and people_count > 0:
            violations.append('No safety signage detected in image with workers present.')
        
        return violations
    
    def _calculate_score(self, people_count: int, signage_text: str, violations: List[str]) -> int:
        """Calculate compliance score (0-100)"""
        score = 100
        
        # Deduct points for violations
        score -= len(violations) * 15
        
        # Bonus for proper signage
        if signage_text and people_count > 0:
            score += 10
        
        # Ensure score is within bounds
        return max(0, min(100, score))

def create_annotated_image(image_path: str, detections: List[Dict], output_path: str):
    """Draw bounding boxes on image for detected people"""
    img = cv2.imread(image_path)
    
    for detection in detections:
        if detection['type'] == 'person':
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle (red for people)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add label
            label = f"Person {detection['confidence']}"
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, img)
    return output_path

