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
        # Import OpenAI for type hints
        from openai import OpenAI
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image for safety compliance
        Returns: {
            'people_count': int,
            'signage_text': str,
            'violations': List[str],
            'detections': List[Dict],
            'compliance_score': int (0-100),
            'ocr_processing_steps': List[Dict]  # New: processing steps for debugging
        }
        """
        # 1. Run YOLO detection with higher confidence threshold
        results = self.model(image_path, conf=0.6)
        
        # 2. Extract all detections (not just people)
        people_count = 0
        detections = []
        
        # Get class names from model (COCO dataset has 80 classes)
        class_names = self.model.names
        
        # Filter out classes unlikely to be on construction sites
        UNLIKELY_CLASSES = [
            'airplane', 'boat', 'train',  # Transportation not on construction sites
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',  # Animals
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  # Food
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',  # Kitchen items
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'tv',  # Electronics/appliances
            'couch', 'bed', 'dining table', 'toilet',  # Furniture
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',  # Personal items
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'frisbee', 'skis', 'snowboard',  # Sports equipment
            'book', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',  # Miscellaneous
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter',  # Street infrastructure (usually not on construction sites)
            'bench', 'potted plant', 'vase', 'clock'  # Decorative items
        ]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[class_id]  # Get class name (e.g., 'person', 'car', 'bus')
                
                # Skip unlikely classes for construction sites
                if class_name.lower() in [c.lower() for c in UNLIKELY_CLASSES]:
                    continue
                
                coords = box.xyxy[0].tolist()
                detections.append({
                    'type': class_name,
                    'confidence': round(confidence, 2),
                    'bbox': [round(c, 2) for c in coords]
                })
                
                # Count people separately for compliance metrics
                if class_id == 0:  # Class 0 = person in COCO dataset
                    people_count += 1
        
        # 3. Run Tesseract OCR with preprocessing
        signage_text, ocr_steps = self._extract_text_with_ocr(image_path)
        
        # 4. Compliance analysis
        violations = self._check_compliance(people_count, signage_text)
        compliance_score = self._calculate_score(people_count, signage_text, violations)
        
        return {
            'people_count': people_count,
            'signage_text': signage_text,
            'violations': violations,
            'detections': detections,
            'compliance_score': compliance_score,
            'ocr_processing_steps': ocr_steps
        }
    
    def _extract_text_with_ocr(self, image_path: str) -> tuple[str, List[Dict]]:
        """
        Extract text from image using Tesseract OCR with proper preprocessing
        Tries multiple preprocessing methods and OCR configurations for best results
        Returns: (final_text, processing_steps)
        """
        steps = []
        try:
            # Load image with OpenCV for preprocessing
            img_cv = cv2.imread(image_path)
            steps.append({"step": 1, "name": "Load Image", "status": "started"})
            if img_cv is None:
                steps.append({"step": 1, "name": "Load Image", "status": "failed", "note": "OpenCV failed, using PIL fallback"})
                # Fallback to PIL if OpenCV fails
                img = Image.open(image_path)
                text, sub_steps = self._ocr_with_config(img, return_steps=True)
                steps.extend(sub_steps)
                return text, steps
            steps.append({"step": 1, "name": "Load Image", "status": "completed", "note": f"Image size: {img_cv.shape}"})
            
            steps.append({"step": 2, "name": "Preprocessing Methods", "status": "started"})
            # Method 1: Grayscale + threshold (best for high contrast signs)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            steps.append({"step": 2.1, "name": "Grayscale Conversion", "status": "completed"})
            
            # Apply adaptive thresholding for better text extraction
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            steps.append({"step": 2.2, "name": "Adaptive Thresholding", "status": "completed", "method": "Method 1"})
            
            # Convert back to PIL Image for Tesseract
            img_processed = Image.fromarray(thresh)
            text1, steps1 = self._ocr_with_config(img_processed, psm=6, return_steps=True, method_name="Method 1: Adaptive Threshold")
            steps.extend(steps1)
            
            # Method 2: Grayscale + Otsu thresholding
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_processed2 = Image.fromarray(thresh2)
            steps.append({"step": 2.3, "name": "Otsu Thresholding", "status": "completed", "method": "Method 2"})
            text2, steps2 = self._ocr_with_config(img_processed2, psm=7, return_steps=True, method_name="Method 2: Otsu Threshold")
            steps.extend(steps2)
            
            # Method 3: Enhanced contrast + grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_processed3 = Image.fromarray(thresh3)
            steps.append({"step": 2.4, "name": "CLAHE Contrast Enhancement + Otsu", "status": "completed", "method": "Method 3"})
            text3, steps3 = self._ocr_with_config(img_processed3, psm=6, return_steps=True, method_name="Method 3: CLAHE + Otsu")
            steps.extend(steps3)
            
            # Method 4: Original image (fallback)
            img_original = Image.open(image_path)
            steps.append({"step": 2.5, "name": "Original Image (No Preprocessing)", "status": "completed", "method": "Method 4"})
            text4, steps4 = self._ocr_with_config(img_original, psm=6, return_steps=True, method_name="Method 4: Original Image")
            steps.extend(steps4)
            
            steps.append({"step": 3, "name": "Text Validation & Filtering", "status": "started"})
            # Collect all results with validation
            results = [
                ("Method 1: Adaptive Threshold", text1),
                ("Method 2: Otsu Threshold", text2),
                ("Method 3: CLAHE + Otsu", text3),
                ("Method 4: Original Image", text4)
            ]
            non_empty = [(name, t) for name, t in results if t.strip()]
            
            steps.append({"step": 3.1, "name": "Raw OCR Results", "status": "completed", 
                         "results": [{"method": name, "text": text[:50] + "..." if len(text) > 50 else text, "length": len(text)} 
                                     for name, text in results]})
            
            if non_empty:
                # Filter out gibberish and get confidence scores
                valid_results = []
                for method_name, text in non_empty:
                    is_valid = self._is_valid_text(text)
                    steps.append({"step": 3.2, "name": f"Validate: {method_name}", "status": "completed",
                                "text_preview": text[:50] + "..." if len(text) > 50 else text,
                                "is_valid": is_valid})
                    
                    if is_valid:
                        # Get confidence score for this text
                        conf = self._get_ocr_confidence(img_cv, text)
                        steps.append({"step": 3.3, "name": f"Confidence Check: {method_name}", "status": "completed",
                                    "confidence": round(conf, 1), "passed": conf > 30})
                        if conf > 30:  # Minimum 30% confidence
                            valid_results.append((text, conf, method_name))
                
                if valid_results:
                    # Return the result with highest confidence
                    best_text, best_conf, best_method = max(valid_results, key=lambda x: x[1])
                    steps.append({"step": 3.4, "name": "Select Best Result", "status": "completed",
                                "selected_method": best_method, "confidence": round(best_conf, 1)})
                    
                    # Final AI validation for low-confidence results
                    if best_conf < 50 and self.openai_client:
                        ai_valid = self._ai_validate_text(best_text)
                        steps.append({"step": 3.5, "name": "AI Validation", "status": "completed",
                                    "confidence": round(best_conf, 1), "ai_valid": ai_valid})
                        if not ai_valid:
                            steps.append({"step": 3.6, "name": "Final Result", "status": "rejected", 
                                        "reason": "AI validation failed - text appears to be gibberish"})
                            return "", steps
                    
                    # Apply final cleaning
                    final_text = self._clean_ocr_text(best_text)
                    steps.append({"step": 4, "name": "Final Text Cleaning", "status": "completed",
                                "original": best_text[:100] + "..." if len(best_text) > 100 else best_text,
                                "cleaned": final_text[:100] + "..." if len(final_text) > 100 else final_text})
                    
                    steps.append({"step": 5, "name": "OCR Processing Complete", "status": "completed",
                                "final_text": final_text, "final_length": len(final_text)})
                    return final_text.strip(), steps
                else:
                    steps.append({"step": 3.6, "name": "Final Result", "status": "rejected", 
                                "reason": "No valid text found after filtering (all results failed validation or confidence check)"})
                    return "", steps
            else:
                steps.append({"step": 3.6, "name": "Final Result", "status": "rejected", 
                            "reason": "No text extracted from any preprocessing method"})
                return "", steps
                
        except Exception as e:
            steps.append({"step": "error", "name": "OCR Processing Error", "status": "failed", "error": str(e)})
            print(f"OCR preprocessing error: {e}")
            # Fallback to simple OCR
            try:
                img = Image.open(image_path)
                text, sub_steps = self._ocr_with_config(img, return_steps=True)
                steps.extend(sub_steps)
                return text, steps
            except Exception as e2:
                steps.append({"step": "error", "name": "Fallback OCR Error", "status": "failed", "error": str(e2)})
                print(f"Warning: Tesseract OCR not available: {e2}")
                return "", steps
    
    def _ocr_with_config(self, img: Image.Image, psm: int = 6, return_steps: bool = False, method_name: str = "") -> str | tuple[str, List[Dict]]:
        """
        Run Tesseract OCR with specific configuration
        PSM modes:
        6 = Assume a single uniform block of text
        7 = Treat the image as a single text line
        8 = Treat the image as a single word
        """
        steps = []
        if return_steps:
            steps.append({"step": f"ocr_{psm}", "name": f"Tesseract OCR (PSM {psm})", "status": "started", "method": method_name})
        
        try:
            # Tesseract configuration for signs and text
            custom_config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]- '
            
            raw_text = pytesseract.image_to_string(img, config=custom_config).strip()
            if return_steps:
                steps.append({"step": f"ocr_{psm}_raw", "name": "Raw OCR Output", "status": "completed", 
                            "text": raw_text[:100] + "..." if len(raw_text) > 100 else raw_text, "length": len(raw_text)})
            
            # Clean up OCR errors
            cleaned_text = self._clean_ocr_text(raw_text)
            if return_steps:
                steps.append({"step": f"ocr_{psm}_cleaned", "name": "Initial Text Cleaning", "status": "completed",
                            "original": raw_text[:100] + "..." if len(raw_text) > 100 else raw_text,
                            "cleaned": cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text})
            
            if return_steps:
                steps.append({"step": f"ocr_{psm}", "name": f"Tesseract OCR (PSM {psm})", "status": "completed", 
                            "final_text": cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text})
                return cleaned_text, steps
            return cleaned_text
        except Exception as e:
            if return_steps:
                steps.append({"step": f"ocr_{psm}_error", "name": "OCR Error", "status": "failed", "error": str(e)})
            print(f"OCR config error: {e}")
            # Fallback to default OCR
            try:
                text = pytesseract.image_to_string(img).strip()
                text = self._clean_ocr_text(text)
                if return_steps:
                    steps.append({"step": f"ocr_{psm}_fallback", "name": "Fallback OCR", "status": "completed",
                                "text": text[:100] + "..." if len(text) > 100 else text})
                    return text, steps
                return text
            except:
                if return_steps:
                    steps.append({"step": f"ocr_{psm}_error", "name": "OCR Failed", "status": "failed"})
                    return "", steps
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
        
        # Always run rule-based segmentation first (works even without AI)
        if cleaned:
            cleaned = self._rule_based_segmentation(cleaned)
        
        # Use AI to further improve if available
        if cleaned and self.openai_client:
            try:
                ai_cleaned = self._ai_cleanup_text(cleaned)
                # Use AI result if it's better (longer or more structured)
                if ai_cleaned and len(ai_cleaned) > len(cleaned) * 0.8:  # AI result should be reasonable
                    cleaned = ai_cleaned
            except Exception as e:
                print(f"AI text cleanup error, using rule-based result: {e}")
                # Keep rule-based result if AI fails
        
        return cleaned
    
    def _get_ocr_confidence(self, img_cv, text: str) -> float:
        """
        Get average confidence score for OCR text by re-running OCR with confidence data
        """
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_pil = Image.fromarray(gray)
            
            # Get detailed data with confidence scores
            data = pytesseract.image_to_data(img_pil, config='--psm 6 --oem 3', output_type=pytesseract.Output.DICT)
            
            confidences = [float(conf) for conf in data['conf'] if int(conf) > 0]
            if confidences:
                return sum(confidences) / len(confidences)
            return 0.0
        except:
            return 50.0  # Default confidence if we can't calculate
    
    def _is_valid_text(self, text: str) -> bool:
        """
        Validate if extracted text is likely real text (not gibberish)
        Checks for:
        - Reasonable word length patterns
        - Presence of common words/patterns
        - Not too many random capital letters
        - Reasonable character distribution
        """
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip()
        
        # Check 1: Too many random capital letters in middle of words (gibberish pattern)
        # Real text has mostly lowercase with occasional capitals
        words = text.split()
        if len(words) > 0:
            random_caps = sum(1 for w in words if len(w) > 2 and w[1:].isupper())
            if random_caps > len(words) * 0.3:  # More than 30% random caps = likely gibberish
                return False
        
        # Check 2: Too many consecutive consonants (unlikely in English) - STRICTER
        # Pattern like "preracicns" or "crcsyell" has too many consonant clusters
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', text.lower())
        if len(consonant_clusters) > 0:  # ANY 3+ consonant cluster is suspicious
            # But allow common safety words that might have clusters
            safety_words_with_clusters = ['hard', 'hat', 'hardhat', 'hard hat', 'required', 'protective']
            text_lower = text.lower()
            has_allowed_clusters = any(word in text_lower for word in safety_words_with_clusters)
            if not has_allowed_clusters:
                # If we have consonant clusters and no safety words, likely gibberish
                return False
        
        # Check 3: Check for common safety sign words (if present, likely valid)
        safety_keywords = [
            'danger', 'warning', 'caution', 'hard', 'hat', 'required', 'must', 'wear',
            'scaffolding', 'incomplete', 'complete', 'no', 'access', 'authorized',
            'personnel', 'footwear', 'visibility', 'protective', 'area', 'site'
        ]
        text_lower = text.lower()
        has_safety_words = any(keyword in text_lower for keyword in safety_keywords)
        
        # Check 4: Reasonable word length (most words 2-15 chars)
        word_lengths = [len(w) for w in words if w.isalpha()]
        if word_lengths:
            avg_length = sum(word_lengths) / len(word_lengths)
            if avg_length > 12:  # Average word too long = likely gibberish
                return False
        
        # Check 5: If no safety keywords and text looks random, be more strict
        if not has_safety_words:
            # Check for vowel-consonant balance (real words have vowels)
            vowels = sum(1 for c in text.lower() if c in 'aeiou')
            if len(text) > 10 and vowels < len(text) * 0.2:  # Less than 20% vowels = likely gibberish
                return False
        
        # Check 6: Too many random character patterns (like "HN EXIM Wek oe Neee") - STRICTER
        # Check for patterns of 2-4 letter words with random caps
        allowed_short_words = ['NO', 'HARD', 'HAT', 'PPE', 'CDM', 'HSE', 'WARNING', 'DANGER', 'CAUTION', 'HARD', 'HAT']
        short_random_words = sum(1 for w in words if len(w) <= 4 and w.isupper() and w not in allowed_short_words)
        if len(words) > 0 and short_random_words > len(words) * 0.3:  # 30% threshold (was 40%) - STRICTER
            return False
        
        # Check 7: Additional pattern check - words with weird mixed case (like "Wek", "Neee")
        weird_case_words = sum(1 for w in words if len(w) > 2 and (
            (w[0].islower() and w[1:].isupper()) or  # Like "wEK"
            (w.isupper() and len(w) > 4 and w not in allowed_short_words)  # Long all-caps that aren't safety words
        ))
        if len(words) > 0 and weird_case_words > len(words) * 0.3:  # More than 30% weird case = gibberish
            return False
        
        return True
    
    def _ai_validate_text(self, text: str) -> bool:
        """
        Use AI to validate if text is gibberish or real text
        Returns True if text is valid, False if gibberish
        """
        if not text or not self.openai_client:
            return True  # If no AI, assume valid (fallback to other checks)
        
        try:
            prompt = f"""Is this text from a construction safety sign real text or OCR gibberish?

Text: "{text}"

Respond with ONLY "valid" or "gibberish". 
- "valid" if it contains real words/phrases that could be on a safety sign
- "gibberish" if it's random characters, OCR errors, or meaningless text

Examples:
- "Danger | Scaffolding incomplete" → valid
- "preracicns HN EXIM Wek oe Neee" → gibberish
- "WARNING | Hard hat required" → valid
- "Recs iface Wee Paenbmars esa aie" → gibberish

Response:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a text validation expert. Determine if OCR text is real or gibberish."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            return "valid" in result
            
        except Exception as e:
            print(f"AI text validation error: {e}")
            return True  # If AI fails, assume valid (fallback to other checks)
    
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
2. Add proper spacing between words that got merged (e.g., "Dangeroussite" → "Dangerous site", "Egjacketsmust" → "e.g. jackets must")
3. Segment different parts of the sign with " | " separator
4. Capitalize appropriately for safety signs
5. Remove ALL artifacts:
   - Remove single letters that are OCR errors (like "a", ". AN", ". A")
   - Remove punctuation-only segments
   - Remove incomplete words
6. Only return complete, real words - no fragments or artifacts
7. Fix common patterns: "Eg" → "e.g.", "mustbeworn" → "must be worn"

Return ONLY the cleaned, segmented text with real words. Use " | " to separate sections.
Be aggressive about removing artifacts - only keep meaningful text.

Examples:
- "a Dangerr . AN Scaffoldingling, incomplete" → "Danger | Scaffolding incomplete"
- "WARNING Dangeroussite Nochildrenallowed Egjacketsmust" → "WARNING | Dangerous site | No children allowed | e.g. jackets must"

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
            # Apply artifact removal as final step
            cleaned_text = self._remove_artifacts(cleaned_text)
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
            'Egjacketsmust': 'e.g. jackets must',
            'Egjackets': 'e.g. jackets',
            'Scaffoldingling': 'Scaffolding',
            'Dangerr': 'Danger',
            # Fix "Eg" patterns
            'Eg ': 'e.g. ',
            'Eg': 'e.g.',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        # Add separators before common section markers
        text = re.sub(r'\s+(WARNING|DANGER|CAUTION|NOTICE|HARD HAT|PROTECTIVE|HIGH VISIBILITY|NO CHILDREN|NO ADMITTANCE|SCAFFOLDING)', r' | \1', text, flags=re.IGNORECASE)
        
        # Clean up multiple separators
        text = re.sub(r'\s*\|\s*\|\s*', ' | ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Post-processing: Remove artifacts
        text = self._remove_artifacts(text)
        
        return text.strip()
    
    def _remove_artifacts(self, text: str) -> str:
        """
        Remove OCR artifacts like single letters, punctuation-only segments, etc.
        """
        if not text:
            return text
        
        # Split by separator to process each segment
        segments = text.split(' | ')
        cleaned_segments = []
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Remove single-letter words at start/end (likely OCR artifacts)
            # But keep "I" and "A" if they're standalone words
            words = segment.split()
            if len(words) > 0:
                # Remove single-letter artifacts at start
                while len(words) > 0 and len(words[0]) == 1 and words[0].lower() not in ['i', 'a']:
                    words.pop(0)
                # Remove single-letter artifacts at end
                while len(words) > 0 and len(words[-1]) == 1 and words[-1].lower() not in ['i', 'a']:
                    words.pop()
            
            # Remove punctuation-only segments
            segment_clean = ' '.join(words).strip()
            if segment_clean and not re.match(r'^[.,;:!?\-\[\]()]+$', segment_clean):
                # Remove patterns like ". AN", ". A", etc.
                segment_clean = re.sub(r'^\.\s*[A-Z]{1,2}\s*$', '', segment_clean, flags=re.IGNORECASE)
                segment_clean = segment_clean.strip()
                
                if segment_clean and len(segment_clean) > 1:  # Keep segments with actual content
                    cleaned_segments.append(segment_clean)
        
        # Rejoin with separators
        result = ' | '.join(cleaned_segments)
        
        # Final cleanup: remove any remaining single-letter artifacts
        result = re.sub(r'\b[a-zA-Z]\b(?!\s*\|)', '', result)  # Remove single letters not before separator
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s*\|\s*\|', ' | ', result)
        
        return result.strip()
    
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
    """Draw bounding boxes on image for all detected objects"""
    img = cv2.imread(image_path)
    
    # Color mapping for different object types
    # People: red, Vehicles: blue, Equipment: green, Other: yellow
    def get_color(obj_type: str):
        obj_lower = obj_type.lower()
        if 'person' in obj_lower:
            return (0, 0, 255)  # Red for people
        elif any(v in obj_lower for v in ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train']):
            return (255, 0, 0)  # Blue for vehicles
        elif any(e in obj_lower for e in ['forklift', 'crane', 'excavator', 'backhoe']):
            return (0, 255, 0)  # Green for equipment
        else:
            return (0, 255, 255)  # Yellow for other objects
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        obj_type = detection['type']
        confidence = detection['confidence']
        color = get_color(obj_type)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Add label with object type and confidence
        label = f"{obj_type.capitalize()} {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(output_path, img)
    return output_path

