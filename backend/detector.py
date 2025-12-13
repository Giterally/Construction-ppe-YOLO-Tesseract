import os
import re
# Set environment variable for PyTorch 2.6+ compatibility
os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = '1'

from ultralytics import YOLO
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional

# Configure Tesseract path for macOS Homebrew installation
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

class SafetyComplianceDetector:
    def __init__(self, openai_client=None):
        """
        Initialize YOLO model - yolov8n.pt auto-downloads on first run
        openai_client: Optional OpenAI client for AI-powered text cleanup
        """
        self.model = YOLO('yolov8n.pt')
        self.openai_client = openai_client
        
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
        
        # 3. Run Tesseract OCR with preprocessing
        signage_text = self._extract_text_with_ocr(image_path)
        
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
    
    def _extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from image using Tesseract OCR with proper preprocessing
        Tries multiple preprocessing methods and OCR configurations for best results
        """
        try:
            # Load image with OpenCV for preprocessing
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                # Fallback to PIL if OpenCV fails
                img = Image.open(image_path)
                return self._ocr_with_config(img)
            
            # Method 1: Grayscale + threshold (best for high contrast signs)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text extraction
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL Image for Tesseract
            img_processed = Image.fromarray(thresh)
            text1 = self._ocr_with_config(img_processed, psm=6)  # Uniform block
            
            # Method 2: Grayscale + Otsu thresholding
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_processed2 = Image.fromarray(thresh2)
            text2 = self._ocr_with_config(img_processed2, psm=7)  # Single text line
            
            # Method 3: Enhanced contrast + grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_processed3 = Image.fromarray(thresh3)
            text3 = self._ocr_with_config(img_processed3, psm=6)
            
            # Method 4: Original image (fallback)
            img_original = Image.open(image_path)
            text4 = self._ocr_with_config(img_original, psm=6)
            
            # Return the longest non-empty result (most likely to be correct)
            results = [text1, text2, text3, text4]
            non_empty = [t for t in results if t.strip()]
            
            if non_empty:
                # Return the result with most characters (usually most accurate)
                return max(non_empty, key=len).strip()
            else:
                return ""
                
        except Exception as e:
            print(f"OCR preprocessing error: {e}")
            # Fallback to simple OCR
            try:
                img = Image.open(image_path)
                return self._ocr_with_config(img)
            except Exception as e2:
                print(f"Warning: Tesseract OCR not available: {e2}")
                return ""
    
    def _ocr_with_config(self, img: Image.Image, psm: int = 6) -> str:
        """
        Run Tesseract OCR with specific configuration
        PSM modes:
        6 = Assume a single uniform block of text
        7 = Treat the image as a single text line
        8 = Treat the image as a single word
        """
        try:
            # Tesseract configuration for signs and text
            custom_config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]- '
            
            text = pytesseract.image_to_string(img, config=custom_config).strip()
            # Clean up OCR errors
            text = self._clean_ocr_text(text)
            return text
        except Exception as e:
            print(f"OCR config error: {e}")
            # Fallback to default OCR
            try:
                text = pytesseract.image_to_string(img).strip()
                text = self._clean_ocr_text(text)
                return text
            except:
                return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean up common OCR errors and artifacts
        """
        if not text:
            return text
        
        # Common OCR character substitutions
        replacements = {
            # Common misreads
            'f]': 'r',
            'f,': 'r',
            'fal': 'ling',
            'IAN': 'AN',
            'ae': 'a',
            '[a]': '',
            ']': '',
            '[': '',
            # Fix common letter confusions
            '0': 'O',  # In words, 0 is usually O
            '1': 'I',  # In words, 1 is usually I
            '5': 'S',  # Sometimes confused
            # Remove trailing artifacts
            'f]': 'r',
            'f,': 'r',
        }
        
        # Apply replacements
        cleaned = text
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Remove standalone punctuation artifacts
        # Remove brackets and their contents if they're artifacts
        cleaned = re.sub(r'\[[a-z]\]', '', cleaned, flags=re.IGNORECASE)
        # Remove trailing punctuation that's likely OCR error
        cleaned = re.sub(r'[\]\[,;]+$', '', cleaned)
        
        # Fix common word patterns in safety signs
        safety_word_fixes = {
            'Dangerf': 'Danger',
            'Dangerf]': 'Danger',
            'Scaffoldingfal': 'Scaffolding',
            'Scaffoldingf': 'Scaffolding',
            'incomplete[a]': 'incomplete',
            'incomplete]': 'incomplete',
            'incompletef': 'incomplete',
            'HARD HAT': 'HARD HAT',
            'HARD HATf': 'HARD HAT',
            'REQUIREDf': 'REQUIRED',
            'REQUIRED]': 'REQUIRED',
        }
        
        for wrong, correct in safety_word_fixes.items():
            if wrong in cleaned:
                cleaned = cleaned.replace(wrong, correct)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Use AI to further clean and segment if available
        if cleaned and self.openai_client:
            cleaned = self._ai_cleanup_text(cleaned)
        
        return cleaned
    
    def _ai_cleanup_text(self, text: str) -> str:
        """
        Use OpenAI to clean up OCR text, fix errors, and segment into readable sections
        """
        if not text or not self.openai_client:
            return text
        
        try:
            prompt = f"""You are cleaning up OCR text from a construction safety sign. 

Raw OCR text: "{text}"

Tasks:
1. Fix OCR character errors (e.g., "Dangerr" → "Danger", "Scaffoldingling" → "Scaffolding")
2. Add proper spacing between words that got merged (e.g., "Dangeroussite" → "Dangerous site")
3. Segment different parts of the sign with " | " separator
4. Capitalize appropriately for safety signs
5. Remove artifacts and fix punctuation

Return ONLY the cleaned, segmented text. Use " | " to separate different sections/phrases.
Keep it concise and accurate.

Examples:
- "a Dangerr . AN Scaffoldingling, incomplete" → "Danger | Scaffolding incomplete"
- "WARNING Dangeroussite Nochildrenallowed" → "WARNING | Dangerous site | No children allowed"

Cleaned text:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for text cleanup
                messages=[
                    {"role": "system", "content": "You are a text cleaning expert specializing in OCR error correction for safety signs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=200
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            # Remove quotes if AI wrapped the response
            cleaned_text = cleaned_text.strip('"\'')
            return cleaned_text
            
        except Exception as e:
            print(f"AI text cleanup error: {e}")
            # Fallback to rule-based cleanup
            return self._rule_based_segmentation(text)
    
    def _rule_based_segmentation(self, text: str) -> str:
        """
        Rule-based text segmentation when AI is not available
        """
        if not text:
            return text
        
        # Common safety sign keywords that should be separated
        keywords = [
            'WARNING', 'DANGER', 'CAUTION', 'NOTICE',
            'HARD HAT', 'PROTECTIVE FOOTWEAR', 'HIGH VISIBILITY',
            'NO CHILDREN', 'NO ADMITTANCE', 'AUTHORIZED PERSONNEL',
            'SCAFFOLDING', 'INCOMPLETE', 'COMPLETE'
        ]
        
        # Add spaces before capital letters (for merged words)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix common merged words
        fixes = {
            'Dangeroussite': 'Dangerous site',
            'Nochildrenallowed': 'No children allowed',
            'Noadmittancefor': 'No admittance for',
            'unauthorisedpersonnel': 'unauthorised personnel',
            'Thisisahard': 'This is a hard',
            'hatarea': 'hat area',
            'Protectivefootwear': 'Protective footwear',
            'mustbeworn': 'must be worn',
            'Highvisibility': 'High visibility',
            'Egjacketsmustbeworn': 'e.g. jackets must be worn',
            'Scaffoldingling': 'Scaffolding',
            'Dangerr': 'Danger',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        # Add separators before common section markers
        text = re.sub(r'\s+(WARNING|DANGER|CAUTION|NOTICE|HARD HAT|PROTECTIVE|HIGH VISIBILITY|NO CHILDREN|NO ADMITTANCE|SCAFFOLDING)', r' | \1', text, flags=re.IGNORECASE)
        
        # Clean up multiple separators
        text = re.sub(r'\s*\|\s*\|\s*', ' | ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
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

