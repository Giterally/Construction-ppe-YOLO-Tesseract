"""
Document processing and cross-checking with UK construction regulations
"""
from typing import Dict, List
import PyPDF2
import pytesseract
from PIL import Image
from openai import OpenAI
import os
import json

# UK Construction Regulations Database
UK_CONSTRUCTION_DOCUMENTS = {
    "cdm_2015_general": {
        "name": "CDM 2015 - General Construction Work",
        "description": "Construction (Design and Management) Regulations 2015 - General site requirements",
        "official_link": "https://www.hse.gov.uk/construction/cdm/2015/index.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats (EN 397) mandatory for all personnel",
                "High-visibility clothing (EN ISO 20471) required",
                "Safety footwear (EN ISO 20345) with protective toe caps",
                "Eye protection where risk of flying particles"
            ],
            "access_requirements": [
                "Edge protection required at 2m or more from any edge where a person could fall",
                "Barriers, guardrails (950mm height) and toe boards required on all open edges",
                "Safe access routes must be maintained",
                "Warning signs for overhead work areas"
            ],
            "equipment_requirements": [
                "Scaffolds must be erected by competent persons",
                "All equipment must be inspected before use",
                "Machinery must have appropriate guards and safety devices"
            ],
            "personnel_limits": {
                "max_workers": None,
                "restrictions": "Only trained and competent personnel allowed on site"
            },
            "inspection_requirements": [
                "Daily visual inspections of work areas",
                "Weekly formal inspections of scaffolds and access equipment",
                "Inspection after adverse weather conditions"
            ],
            "zone_requirements": [
                "Exclusion zones around crane operations",
                "Restricted access to areas with overhead work",
                "Clear signage for hazardous areas"
            ]
        }
    },
    "cdm_2015_working_at_height": {
        "name": "CDM 2015 - Working at Height",
        "description": "Specific requirements for work at height activities",
        "official_link": "https://www.hse.gov.uk/work-at-height/index.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats mandatory",
                "High-visibility clothing required",
                "Safety harnesses and lanyards where fall risk exists",
                "Safety footwear with good grip"
            ],
            "access_requirements": [
                "Edge protection mandatory at 2m+ height",
                "Guardrails at 950mm with mid-rail at 470mm",
                "Toe boards minimum 150mm height",
                "Scaffold platforms minimum 600mm wide",
                "Ladders only for access, not as work platforms"
            ],
            "equipment_requirements": [
                "Scaffolds must be designed and erected by competent persons",
                "Mobile elevated work platforms (MEWPs) require operator certification",
                "All fall protection equipment must be inspected before use"
            ],
            "personnel_limits": {
                "max_workers": None,
                "restrictions": "Only persons trained in work at height allowed"
            },
            "inspection_requirements": [
                "Scaffolds inspected before first use",
                "Scaffolds inspected every 7 days",
                "Inspection after adverse weather or modifications",
                "MEWPs inspected daily before use"
            ],
            "zone_requirements": [
                "Exclusion zones below work at height",
                "Warning barriers at ground level",
                "No work below overhead operations"
            ]
        }
    },
    "hse_construction_ppe": {
        "name": "HSE - Construction Site PPE Requirements",
        "description": "Health and Safety Executive mandatory PPE for construction sites",
        "official_link": "https://www.hse.gov.uk/construction/healthrisks/personal-protective-equipment.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats (EN 397) - mandatory for all personnel",
                "High-visibility clothing (EN ISO 20471) Class 2 or 3",
                "Safety footwear (EN ISO 20345) with protective toe caps",
                "Eye protection (EN 166) where risk of impact or particles",
                "Hearing protection (EN 352) in high noise areas",
                "Gloves appropriate for task hazards"
            ],
            "access_requirements": [
                "Clear pedestrian routes separated from vehicle routes",
                "Barriers around excavations and openings",
                "Warning signs for overhead hazards"
            ],
            "equipment_requirements": [],
            "personnel_limits": {},
            "inspection_requirements": [
                "PPE inspected before each use",
                "Damaged PPE must be replaced immediately"
            ],
            "zone_requirements": [
                "PPE zones clearly marked",
                "No entry without appropriate PPE"
            ]
        }
    },
    "scaffold_safety_bs5973": {
        "name": "BS 5973 - Scaffold Safety Requirements",
        "description": "British Standard for scaffold design and safety",
        "official_link": "https://www.hse.gov.uk/construction/safetytopics/scaffolding.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats mandatory",
                "High-visibility clothing",
                "Safety footwear"
            ],
            "access_requirements": [
                "Guardrails at 950mm height with mid-rail at 470mm",
                "Toe boards minimum 150mm height",
                "Working platforms minimum 600mm wide",
                "Ladder access with handrails"
            ],
            "equipment_requirements": [
                "Scaffolds designed to BS 5973 standards",
                "Maximum loading not to exceed design limits",
                "Base plates and sole boards required",
                "Ties to structure as per design"
            ],
            "personnel_limits": {
                "max_workers": None,
                "restrictions": "Loading must not exceed design capacity"
            },
            "inspection_requirements": [
                "Inspection before first use",
                "Inspection every 7 days",
                "Inspection after adverse weather",
                "Inspection after modifications",
                "Inspection records must be maintained"
            ],
            "zone_requirements": [
                "Exclusion zones during erection/dismantling",
                "Warning signs for incomplete scaffolds"
            ]
        }
    },
    "excavation_safety_hse": {
        "name": "HSE - Excavation Safety",
        "description": "Requirements for safe excavation work",
        "official_link": "https://www.hse.gov.uk/construction/safetytopics/excavations.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats mandatory",
                "High-visibility clothing",
                "Safety footwear",
                "Eye protection if risk of flying debris"
            ],
            "access_requirements": [
                "Edge protection for excavations over 1.2m deep",
                "Barriers around excavation perimeters",
                "Safe access routes (ladders/steps) for deep excavations",
                "Warning signs and barriers"
            ],
            "equipment_requirements": [
                "Support systems for excavations over 1.2m",
                "Batter/benching for unsupported excavations",
                "Trench boxes where required"
            ],
            "personnel_limits": {
                "max_workers": None,
                "restrictions": "Only trained personnel in excavations"
            },
            "inspection_requirements": [
                "Daily inspection before work starts",
                "Inspection after adverse weather",
                "Inspection after any ground movement",
                "Inspection records maintained"
            ],
            "zone_requirements": [
                "Exclusion zones around excavations",
                "No vehicles within 2m of excavation edge",
                "Materials stored away from edges"
            ]
        }
    },
    "crane_operations_cdm": {
        "name": "CDM 2015 - Crane Operations",
        "description": "Requirements for crane and lifting operations",
        "official_link": "https://www.hse.gov.uk/construction/safetytopics/cranes.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats mandatory",
                "High-visibility clothing",
                "Safety footwear"
            ],
            "access_requirements": [
                "Exclusion zones around crane operations",
                "Barriers preventing access to lifting zones",
                "Warning signs for overhead operations",
                "Banksman/spotter required"
            ],
            "equipment_requirements": [
                "Cranes must have valid test certificates",
                "Lifting equipment inspected before use",
                "Load charts and capacity limits displayed",
                "Communication systems (radio/hand signals)"
            ],
            "personnel_limits": {
                "max_workers": None,
                "restrictions": "Only trained crane operators and slingers"
            },
            "inspection_requirements": [
                "Daily crane inspection before use",
                "Lifting equipment inspected before each lift",
                "Weekly thorough examination records"
            ],
            "zone_requirements": [
                "Exclusion zones around crane radius",
                "No personnel under suspended loads",
                "Clear signage for crane operating areas"
            ]
        }
    },
    "confined_spaces_hse": {
        "name": "HSE - Confined Space Entry",
        "description": "Requirements for work in confined spaces",
        "official_link": "https://www.hse.gov.uk/construction/healthrisks/confined-spaces.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats",
                "High-visibility clothing",
                "Appropriate respiratory protection if required",
                "Safety harnesses with retrieval systems"
            ],
            "access_requirements": [
                "Permit-to-work system required",
                "Barriers preventing unauthorized access",
                "Warning signs at entry points",
                "Emergency rescue equipment at access point"
            ],
            "equipment_requirements": [
                "Gas detection equipment",
                "Ventilation systems",
                "Communication equipment",
                "Rescue equipment and trained rescue team"
            ],
            "personnel_limits": {
                "max_workers": "As per permit and risk assessment",
                "restrictions": "Only trained personnel with valid permit"
            },
            "inspection_requirements": [
                "Pre-entry inspection and gas testing",
                "Continuous monitoring during entry",
                "Equipment inspection before each use"
            ],
            "zone_requirements": [
                "Restricted access - permit required",
                "Standby rescue team at access point",
                "Clear emergency procedures displayed"
            ]
        }
    },
    "demolition_work_cdm": {
        "name": "CDM 2015 - Demolition Work",
        "description": "Requirements for demolition operations",
        "official_link": "https://www.hse.gov.uk/construction/safetytopics/demolition.htm",
        "requirements": {
            "ppe_requirements": [
                "Hard hats mandatory",
                "High-visibility clothing",
                "Safety footwear",
                "Eye protection",
                "Hearing protection",
                "Dust masks/respiratory protection"
            ],
            "access_requirements": [
                "Exclusion zones around demolition areas",
                "Barriers preventing public access",
                "Warning signs and barriers",
                "Safe access routes for workers"
            ],
            "equipment_requirements": [
                "Demolition method statement required",
                "Structural survey before demolition",
                "Services isolated and marked",
                "Dust suppression measures"
            ],
            "personnel_limits": {
                "max_workers": None,
                "restrictions": "Only trained demolition operatives"
            },
            "inspection_requirements": [
                "Daily inspection of exclusion zones",
                "Structural inspection before each phase",
                "Equipment inspection before use"
            ],
            "zone_requirements": [
                "Exclusion zones clearly marked",
                "No access to demolition area during operations",
                "Public exclusion barriers"
            ]
        }
    }
}

# UK Construction Regulations Database (for cross-checking)
UK_REGULATIONS = {
    "CDM_2015": {
        "edge_protection": "Edge protection required at 2m or more from any edge where a person could fall",
        "scaffold_inspection": "Scaffolds must be inspected every 7 days and after adverse weather",
        "working_at_height": "Suitable and sufficient measures required to prevent falls",
        "barriers": "Barriers, guardrails and toe boards required on all open edges"
    },
    "HSE_CONSTRUCTION": {
        "ppe_requirements": "Hard hats, high-visibility clothing, and safety footwear mandatory on construction sites",
        "scaffold_loading": "Loading of scaffolds must not exceed design limits",
        "confined_spaces": "Permit required for confined space entry",
        "excavations": "Excavations over 1.2m deep require edge protection and safe access"
    },
    "SCAFFOLD_SAFETY": {
        "platform_width": "Working platforms must be at least 600mm wide",
        "guardrails": "Guardrails required at 950mm height with mid-rail and toe board",
        "inspection_frequency": "Inspected before first use, every 7 days, and after adverse weather"
    }
}

class DocumentProcessor:
    def __init__(self, openai_api_key: str = None):
        """Initialize with OpenAI API key"""
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print("⚠️  OpenAI API key not found. LLM features will use fallback methods.")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            # Fallback to OCR for scanned PDFs
            return self._ocr_pdf(pdf_path)
    
    def _ocr_pdf(self, pdf_path: str) -> str:
        """Fallback OCR for scanned PDFs"""
        # Convert PDF to images and OCR each page
        # For simplicity, return error message
        return "Error: Unable to extract text. PDF may be image-based."
    
    def get_document_requirements(self, document_id: str) -> Dict:
        """
        Get requirements for a selected UK construction document
        
        Args:
            document_id: ID from UK_CONSTRUCTION_DOCUMENTS
            
        Returns:
            Dictionary of requirements in same format as extract_safety_requirements
        """
        if document_id in UK_CONSTRUCTION_DOCUMENTS:
            return UK_CONSTRUCTION_DOCUMENTS[document_id]["requirements"]
        else:
            return {
                "ppe_requirements": [],
                "access_requirements": [],
                "equipment_requirements": [],
                "personnel_limits": {},
                "inspection_requirements": [],
                "zone_requirements": [],
                "other_requirements": []
            }
    
    def extract_safety_requirements(self, document_text: str) -> Dict:
        """
        Use LLM to extract structured safety requirements from document
        Returns: {
            'ppe_requirements': [...],
            'access_requirements': [...],
            'equipment_requirements': [...],
            'personnel_limits': {...},
            'inspection_requirements': [...],
            'other_requirements': [...]
        }
        """
        if not self.client:
            return self._fallback_extraction(document_text)
        
        prompt = f"""
You are a UK construction safety expert. Analyze this method statement or risk assessment and extract ALL safety requirements.

Document text:
{document_text[:3000]}  

Extract and categorize requirements into JSON format:
{{
  "ppe_requirements": ["list of PPE required"],
  "access_requirements": ["barriers, guardrails, edge protection, etc."],
  "equipment_requirements": ["scaffold specs, machinery, tools, etc."],
  "personnel_limits": {{"max_workers": number, "restrictions": "text"}},
  "inspection_requirements": ["inspection schedules, frequencies"],
  "zone_requirements": ["specific zone restrictions"],
  "other_requirements": ["anything else safety-related"]
}}

Be specific and extract exact requirements with numbers/measurements when mentioned.
Only include requirements actually stated in the document.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a UK construction safety compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            requirements = json.loads(response.choices[0].message.content)
            return requirements
            
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return self._fallback_extraction(document_text)
    
    def _fallback_extraction(self, text: str) -> Dict:
        """Simple keyword-based extraction if LLM fails"""
        requirements = {
            "ppe_requirements": [],
            "access_requirements": [],
            "equipment_requirements": [],
            "personnel_limits": {},
            "inspection_requirements": [],
            "zone_requirements": [],
            "other_requirements": []
        }
        
        text_upper = text.upper()
        
        # Simple keyword matching
        if any(word in text_upper for word in ['HARD HAT', 'HELMET']):
            requirements['ppe_requirements'].append("Hard hats required")
        if any(word in text_upper for word in ['HI-VIS', 'HIGH VISIBILITY']):
            requirements['ppe_requirements'].append("High-visibility clothing required")
        if any(word in text_upper for word in ['BARRIER', 'GUARDRAIL', 'EDGE PROTECTION']):
            requirements['access_requirements'].append("Edge protection/barriers required")
            
        return requirements
    
    def cross_check_with_photo_analysis(
        self, 
        photo_analysis: Dict, 
        document_requirements: Dict,
        document_name: str = "Method Statement"
    ) -> Dict:
        """
        Cross-check photo analysis results against document requirements
        
        Args:
            photo_analysis: Results from YOLO + Tesseract (existing analysis)
            document_requirements: Extracted requirements from document
            
        Returns: {
            'compliant': [...],
            'violations': [...],
            'warnings': [...],
            'regulatory_checks': {...}
        }
        """
        if not self.client:
            return self._fallback_cross_check(photo_analysis, document_requirements)
        
        prompt = f"""
You are a UK construction safety inspector. Cross-check this construction site photo analysis against the method statement requirements and UK regulations (CDM 2015, HSE).

PHOTO ANALYSIS:
- Workers detected: {photo_analysis.get('people_count', 0)}
- Signage text detected: "{photo_analysis.get('signage_text', 'None')}"
- Detections: {json.dumps(photo_analysis.get('detections', []))}

DOCUMENT REQUIREMENTS ({document_name}):
{json.dumps(document_requirements, indent=2)}

UK REGULATIONS TO CHECK:
{json.dumps(UK_REGULATIONS, indent=2)}

Perform a detailed cross-check and return JSON:
{{
  "compliant": [
    "Requirement met with evidence from photo"
  ],
  "violations": [
    {{
      "requirement": "What the document requires",
      "photo_evidence": "What the photo shows (or doesn't show)",
      "regulation": "CDM 2015 / HSE regulation violated",
      "severity": "high/medium/low"
    }}
  ],
  "warnings": [
    {{
      "issue": "Potential concern that needs manual verification",
      "reason": "Why this needs checking"
    }}
  ],
  "regulatory_checks": {{
    "cdm_2015_compliant": true/false,
    "hse_compliant": true/false,
    "issues_found": ["list of specific regulation violations"]
  }}
}}

Be precise. Only flag violations where there's clear evidence of non-compliance.
Note that YOLO can only detect people/objects, not specific PPE items, so flag PPE as "needs manual verification".
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a UK construction safety compliance inspector."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            cross_check = json.loads(response.choices[0].message.content)
            return cross_check
            
        except Exception as e:
            print(f"Cross-check LLM error: {e}")
            return self._fallback_cross_check(photo_analysis, document_requirements)
    
    def _fallback_cross_check(self, photo_analysis: Dict, doc_requirements: Dict) -> Dict:
        """Simple rule-based cross-checking if LLM fails"""
        violations = []
        compliant = []
        warnings = []
        
        people_count = photo_analysis.get('people_count', 0)
        signage = photo_analysis.get('signage_text', '').upper()
        
        # Check personnel limits
        if 'personnel_limits' in doc_requirements:
            limits = doc_requirements['personnel_limits']
            if 'max_workers' in limits:
                max_workers = limits['max_workers']
                if isinstance(max_workers, int) and people_count > max_workers:
                    violations.append({
                        "requirement": f"Maximum {max_workers} workers allowed",
                        "photo_evidence": f"{people_count} workers detected",
                        "regulation": "Method Statement",
                        "severity": "high"
                    })
        
        # Check PPE requirements
        if doc_requirements.get('ppe_requirements'):
            warnings.append({
                "issue": "PPE compliance requires manual verification",
                "reason": f"{people_count} workers detected. Method statement requires: {', '.join(doc_requirements['ppe_requirements'])}"
            })
        
        # Check access requirements vs signage
        if doc_requirements.get('access_requirements') and signage:
            compliant.append(f"Safety signage present: {signage[:50]}")
        
        return {
            "compliant": compliant,
            "violations": violations,
            "warnings": warnings,
            "regulatory_checks": {
                "cdm_2015_compliant": len(violations) == 0,
                "hse_compliant": len(violations) == 0,
                "issues_found": [v['requirement'] for v in violations]
            }
        }

