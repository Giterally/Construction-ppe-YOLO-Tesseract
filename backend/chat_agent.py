"""
Live Site Compliance Assistant
Intelligent chatbot for construction safety questions
"""
from openai import OpenAI
import os
import json
from typing import Dict, List, Optional

# UK Regulations Knowledge Base (reuse from document_processor.py if exists)
UK_REGULATIONS_KNOWLEDGE = {
    "CDM_2015": {
        "working_at_height": "Edge protection required at 2m or more from any edge. Guardrails at 950mm height with mid-rail and toe board.",
        "scaffold_inspection": "Scaffolds must be inspected every 7 days and after adverse weather conditions.",
        "scaffold_loading": "Platform loading must not exceed design limits. Maximum workers per platform based on platform width.",
        "excavations": "Excavations over 1.2m deep require edge protection and safe access/egress.",
        "confined_spaces": "Permit to Work required. Gas testing mandatory. Emergency rescue equipment must be available.",
    },
    "HSE_CONSTRUCTION": {
        "ppe_requirements": "Hard hats, high-visibility clothing, and safety footwear mandatory on all construction sites.",
        "hot_work": "Hot work permits required for welding/cutting. Fire extinguisher within 5m. Fire watch required.",
        "lifting_operations": "Lifting equipment must be inspected every 6 months. Banksman required for crane operations.",
        "electrical_safety": "110V equipment on construction sites. Residual current devices (RCDs) required.",
    },
    "BS_5973": {
        "scaffold_width": "Working platforms must be at least 600mm wide for passage, 800mm for working.",
        "guardrails": "Guardrails required at 950mm height with mid-rail at 470mm and toe board at 150mm.",
        "inspection_frequency": "Inspect before first use, every 7 days, after adverse weather, after alterations.",
    }
}

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
        
        # UK Regulations
        context_parts.append("=== UK CONSTRUCTION REGULATIONS ===")
        for reg_name, reg_content in UK_REGULATIONS_KNOWLEDGE.items():
            context_parts.append(f"\n{reg_name}:")
            for key, value in reg_content.items():
                context_parts.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
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
        system_prompt = """You are a friendly and helpful UK construction safety compliance assistant with expertise in CDM 2015, HSE guidelines, and BS standards.

Your role:
- Answer questions about construction site safety and compliance in a warm, approachable manner
- Reference specific evidence from photo analysis when available
- Cite relevant UK regulations (CDM 2015, HSE, BS standards)
- Give clear, actionable answers
- Use ✓ for compliant items, ⚠️ for warnings, ❌ for violations
- Format responses with sections and bullet points for readability
- Always prioritize worker safety
- Be conversational and friendly, not robotic or argumentative

When answering:
1. For greetings or simple messages, respond warmly and offer to help with safety compliance questions
2. Avoid starting with "NO" or negative language - instead, be positive and helpful
3. Reference photo evidence (detections, OCR text, violations) when available
4. Cite specific regulations when relevant
5. Provide actionable next steps
6. Keep responses concise but friendly (2-3 short paragraphs max)

Tone guidelines:
- Be welcoming and helpful, even for simple messages
- Use phrases like "I'd be happy to help", "Let me check", "Based on your analyses"
- For greetings, respond warmly and ask how you can help with safety compliance
- If photo data is unavailable, acknowledge this in a friendly way and provide general guidance based on regulations"""

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

