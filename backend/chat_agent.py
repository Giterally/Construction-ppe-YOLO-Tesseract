"""
Live Site Compliance Assistant
Intelligent chatbot for construction safety questions
"""
from openai import OpenAI
import os
import json
from typing import Dict, List, Optional
from document_processor import UK_CONSTRUCTION_DOCUMENTS

# Build comprehensive UK Regulations Knowledge Base from all 8 documents
UK_REGULATIONS_KNOWLEDGE = {}

for doc_id, doc_data in UK_CONSTRUCTION_DOCUMENTS.items():
    doc_name = doc_data["name"]
    requirements = doc_data.get("requirements", {})
    
    # Create a summary structure for the chat agent
    knowledge_entry = {
        "name": doc_name,
        "description": doc_data.get("description", ""),
        "official_link": doc_data.get("official_link", ""),
        "ppe_requirements": requirements.get("ppe_requirements", []),
        "access_requirements": requirements.get("access_requirements", []),
        "equipment_requirements": requirements.get("equipment_requirements", []),
        "personnel_limits": requirements.get("personnel_limits", {}),
        "inspection_requirements": requirements.get("inspection_requirements", []),
        "zone_requirements": requirements.get("zone_requirements", [])
    }
    
    # Use a clean key name
    clean_key = doc_id.upper().replace("_", " ")
    UK_REGULATIONS_KNOWLEDGE[clean_key] = knowledge_entry

class SafetyAssistant:
    def __init__(self, openai_api_key: str = None):
        """Initialize Safety Assistant with OpenAI"""
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client (handle proxy issues like in app.py)
        try:
            self.client = OpenAI(api_key=api_key, http_client=None)
        except Exception as e:
            print(f"⚠️  OpenAI client initialization failed: {e}")
            # Fallback
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e2:
                print(f"⚠️  OpenAI client initialization failed (fallback): {e2}")
                raise
        
    def build_context(
        self, 
        latest_analysis: Optional[Dict] = None,
        past_analyses: Optional[List[Dict]] = None
    ) -> str:
        """Build context string for LLM from available data"""
        
        context_parts = []
        
        # Latest photo analysis
        if latest_analysis:
            context_parts.append("=== LATEST SITE PHOTO ANALYSIS ===")
            context_parts.append(f"Timestamp: {latest_analysis.get('created_at', 'Unknown')}")
            context_parts.append(f"Compliance Score: {latest_analysis.get('compliance_score', 0)}/100")
            context_parts.append(f"Workers Detected: {latest_analysis.get('people_count', 0)}")
            
            detections = latest_analysis.get('detections', [])
            if detections:
                context_parts.append(f"Objects Detected: {', '.join([d.get('type', 'unknown') for d in detections])}")
            
            signage = latest_analysis.get('signage_text', '').strip()
            if signage:
                context_parts.append(f"Signage Text (OCR): {signage}")
            
            violations = latest_analysis.get('violations', [])
            if violations:
                context_parts.append(f"Violations Found: {len(violations)}")
                for i, v in enumerate(violations[:3], 1):  # Max 3 for brevity
                    context_parts.append(f"  {i}. {v}")
            
            context_parts.append("")
        
        # All past analyses - detailed context
        if past_analyses and len(past_analyses) > 0:
            context_parts.append("=== ALL SITE PHOTO ANALYSES ===")
            context_parts.append(f"Total Analyses: {len(past_analyses)}")
            
            # Include details from all analyses (most recent first)
            for idx, analysis in enumerate(past_analyses[:20], 1):  # Limit to 20 most recent for context size
                context_parts.append(f"\n--- Analysis #{idx} ({analysis.get('created_at', 'Unknown')[:10]}) ---")
                context_parts.append(f"Compliance Score: {analysis.get('compliance_score', 0)}/100")
                context_parts.append(f"Workers Detected: {analysis.get('people_count', 0)}")
                
                detections = analysis.get('detections', [])
                if detections:
                    detection_types = [d.get('type', 'unknown') for d in detections]
                    context_parts.append(f"Objects: {', '.join(detection_types)}")
                
                signage = analysis.get('signage_text', '').strip()
                if signage:
                    context_parts.append(f"Signage: {signage[:100]}")  # Truncate long text
                
                violations = analysis.get('violations', [])
                if violations:
                    context_parts.append(f"Violations ({len(violations)}):")
                    for v in violations[:2]:  # Max 2 per analysis
                        context_parts.append(f"  - {v}")
            
            # Summary statistics
            if len(past_analyses) > 1:
                avg_score = sum(a.get('compliance_score', 0) for a in past_analyses) / len(past_analyses)
                total_violations = sum(len(a.get('violations', [])) for a in past_analyses)
                context_parts.append(f"\n--- Summary Statistics ---")
                context_parts.append(f"Average Compliance Score: {avg_score:.1f}/100")
                context_parts.append(f"Total Violations Across All Analyses: {total_violations}")
            
            context_parts.append("")
        
        # UK Regulations - All 8 documents
        context_parts.append("=== UK CONSTRUCTION REGULATIONS (All 8 Documents) ===")
        for reg_key, reg_data in UK_REGULATIONS_KNOWLEDGE.items():
            context_parts.append(f"\n{reg_data['name']}:")
            context_parts.append(f"  Description: {reg_data.get('description', '')}")
            
            if reg_data.get('ppe_requirements'):
                context_parts.append(f"  PPE Requirements: {', '.join(reg_data['ppe_requirements'][:3])}")
            if reg_data.get('access_requirements'):
                context_parts.append(f"  Access Requirements: {', '.join(reg_data['access_requirements'][:3])}")
            if reg_data.get('equipment_requirements'):
                context_parts.append(f"  Equipment Requirements: {', '.join(reg_data['equipment_requirements'][:3])}")
            if reg_data.get('inspection_requirements'):
                context_parts.append(f"  Inspection Requirements: {', '.join(reg_data['inspection_requirements'][:2])}")
            if reg_data.get('personnel_limits'):
                limits = reg_data['personnel_limits']
                if limits.get('restrictions'):
                    context_parts.append(f"  Personnel Restrictions: {limits['restrictions']}")
        
        return "\n".join(context_parts)
    
    def answer_question(
        self,
        question: str,
        latest_analysis: Optional[Dict] = None,
        past_analyses: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        Answer a safety compliance question
        
        Returns: {
            'response': str,
            'confidence': str,
            'citations': List[Dict]
        }
        """
        
        # Build context from available data
        context = self.build_context(latest_analysis, past_analyses)
        
        # System prompt
        system_prompt = """You are a friendly and helpful UK construction safety compliance assistant with expertise in all 8 UK construction regulation documents:
1. CDM 2015 - General Construction Work
2. CDM 2015 - Working at Height
3. HSE - Construction Site PPE Requirements
4. BS 5973 - Scaffold Safety Requirements
5. HSE - Excavation Safety
6. CDM 2015 - Crane Operations
7. HSE - Confined Space Entry
8. CDM 2015 - Demolition Work

Your role:
- Answer questions about construction site safety and compliance in a warm, approachable manner
- Reference specific evidence from photo analysis when available
- Cite relevant UK regulations from the 8 documents above
- Give clear, actionable answers
- Format responses with sections and bullet points for readability
- Always prioritize worker safety
- Be conversational and friendly, not robotic or argumentative
- DO NOT use emojis in your responses - use text only (e.g., write "Compliant" instead of checkmarks, "Warning" instead of warning symbols, "Violation" instead of X symbols)

When answering:
1. For greetings or simple messages, respond warmly and offer to help with safety compliance questions
2. Avoid starting with "NO" or negative language - instead, be positive and helpful
3. Reference photo evidence (detections, OCR text, violations) when available
4. Cite specific regulations from the 8 documents when relevant
5. Provide actionable next steps
6. Keep responses concise but friendly (2-3 short paragraphs max)
7. Use plain text indicators: "Compliant", "Warning", "Violation" instead of emojis

Tone guidelines:
- Be welcoming and helpful, even for simple messages
- Use phrases like "I'd be happy to help", "Let me check", "Based on your analyses"
- For greetings, respond warmly and ask how you can help with safety compliance
- If photo data is unavailable, acknowledge this in a friendly way and provide general guidance based on regulations
- Never use emojis, symbols, or special characters - use plain text only"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ]
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract citations (simple regex for [View ...] style links)
            citations = []
            if latest_analysis:
                citations.append({
                    'type': 'photo',
                    'label': 'Latest Photo Analysis',
                    'data': latest_analysis
                })
            
            return {
                'response': ai_response,
                'confidence': 'high',
                'citations': citations
            }
            
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return {
                'response': f"I encountered an error processing your question. Please try again.\n\nError: {str(e)}",
                'confidence': 'error',
                'citations': []
            }

