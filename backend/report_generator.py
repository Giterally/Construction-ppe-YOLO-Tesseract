"""
PDF Report Generator using weasyprint
Generates Build UK Standard Format compliance reports
"""
from weasyprint import HTML
from datetime import datetime
import os
import base64

def generate_compliance_report(analysis_data: dict, output_path: str) -> str:
    """
    Generate PDF compliance report from analysis data
    
    Args:
        analysis_data: Analysis results from database
        output_path: Path to save PDF file
        
    Returns:
        Path to generated PDF file
    """
    
    # Extract data
    compliance_score = analysis_data.get('compliance_score', 0)
    people_count = analysis_data.get('people_count', 0)
    violations = analysis_data.get('violations', [])
    signage_text = analysis_data.get('signage_text', 'None detected')
    detections = analysis_data.get('detections', [])
    created_at = analysis_data.get('created_at', datetime.now().isoformat())
    
    # Optional user-provided fields
    site_name = analysis_data.get('site_name', 'Not specified')
    contractor_name = analysis_data.get('contractor_name', 'Not specified')
    submitted_to = analysis_data.get('submitted_to', 'Not specified')
    
    # Get images and convert to base64 for embedding
    original_image_path = analysis_data.get('original_image_path', '')
    annotated_image_path = analysis_data.get('annotated_image_path', '')
    
    original_image_base64 = ''
    annotated_image_base64 = ''
    
    if original_image_path and os.path.exists(original_image_path):
        with open(original_image_path, 'rb') as f:
            original_image_base64 = base64.b64encode(f.read()).decode()
    
    if annotated_image_path and os.path.exists(annotated_image_path):
        with open(annotated_image_path, 'rb') as f:
            annotated_image_base64 = base64.b64encode(f.read()).decode()
    
    # Generate report ID (first 12 chars of analysis ID)
    report_id = analysis_data.get('id', 'UNKNOWN')[:12]
    
    # Determine score color
    if compliance_score >= 80:
        score_color = '#2ecc71'
    elif compliance_score >= 60:
        score_color = '#f39c12'
    else:
        score_color = '#e74c3c'
    
    # Format created_at date
    try:
        if isinstance(created_at, str):
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%d %B %Y, %H:%M')
        else:
            formatted_date = str(created_at)
    except:
        formatted_date = str(created_at)
    
    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #1a1a1a;
            }}
            
            .header {{
                border-bottom: 3px solid #1a1a1a;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 24pt;
                font-weight: 700;
                margin: 0 0 5px 0;
                letter-spacing: -0.02em;
            }}
            
            .header .subtitle {{
                font-size: 10pt;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }}
            
            .section {{
                margin-bottom: 25px;
                page-break-inside: avoid;
            }}
            
            .section-title {{
                font-size: 14pt;
                font-weight: 700;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }}
            
            .info-item {{
                border: 1px solid #e0e0e0;
                padding: 12px;
            }}
            
            .info-label {{
                font-size: 9pt;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #666;
                margin-bottom: 5px;
            }}
            
            .info-value {{
                font-size: 12pt;
                font-weight: 600;
            }}
            
            .score-box {{
                border-left: 8px solid {score_color};
                background: #fafafa;
                padding: 20px;
                margin-bottom: 25px;
            }}
            
            .score-number {{
                font-size: 48pt;
                font-weight: 700;
                color: {score_color};
                line-height: 1;
            }}
            
            .violations-list {{
                list-style: none;
                padding: 0;
            }}
            
            .violation-item {{
                background: #fff5f5;
                border-left: 4px solid #e74c3c;
                padding: 15px;
                margin-bottom: 10px;
            }}
            
            .violation-text {{
                font-size: 11pt;
                line-height: 1.5;
            }}
            
            .no-violations {{
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                color: #155724;
            }}
            
            .detections-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            
            .detections-table th,
            .detections-table td {{
                border: 1px solid #e0e0e0;
                padding: 8px;
                text-align: left;
            }}
            
            .detections-table th {{
                background: #f8f9fa;
                font-weight: 600;
                font-size: 10pt;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            .images-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 15px;
            }}
            
            .image-container {{
                text-align: center;
            }}
            
            .image-container img {{
                max-width: 100%;
                border: 1px solid #e0e0e0;
                margin-top: 10px;
            }}
            
            .image-label {{
                font-size: 10pt;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            
            .signage-box {{
                background: #fafafa;
                border: 1px solid #e0e0e0;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                white-space: pre-wrap;
                margin-top: 10px;
            }}
            
            .footer {{
                border-top: 1px solid #e0e0e0;
                padding-top: 20px;
                margin-top: 40px;
                font-size: 9pt;
                color: #666;
            }}
            
            .signature-line {{
                margin-top: 30px;
                display: inline-block;
                width: 200px;
                border-bottom: 1px solid #1a1a1a;
                padding-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>CONSTRUCTION SAFETY COMPLIANCE REPORT</h1>
            <div class="subtitle">Build UK Standard Format v1.0</div>
        </div>
        
        <div class="section">
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Site</div>
                    <div class="info-value">{site_name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Date</div>
                    <div class="info-value">{formatted_date}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Contractor</div>
                    <div class="info-value">{contractor_name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Submitted To</div>
                    <div class="info-value">{submitted_to}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Executive Summary</div>
            <div class="score-box">
                <div class="score-number">{compliance_score}/100</div>
                <div style="margin-top: 10px; font-size: 11pt;">
                    <strong>Workers Detected:</strong> {people_count}<br>
                    <strong>Total Detections:</strong> {len(detections)}<br>
                    <strong>Violations Found:</strong> {len(violations)}
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Violations & Actions Required</div>
            {''.join([f'<div class="violation-item"><div class="violation-text">{i+1}. {v}</div></div>' for i, v in enumerate(violations)]) if violations else '<div class="no-violations">✓ No violations detected. Site appears compliant with selected regulations.</div>'}
        </div>
        
        <div class="section">
            <div class="section-title">Detected Signage Text</div>
            <div class="signage-box">{signage_text}</div>
        </div>
        
        <div class="section">
            <div class="section-title">Detection Details</div>
            <table class="detections-table">
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{d.get("type", "Unknown").capitalize()}</td><td>{d.get("confidence", 0)*100:.1f}%</td></tr>' for d in detections]) if detections else '<tr><td colspan="2">No detections</td></tr>'}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">Visual Evidence</div>
            <div class="images-grid">
                <div class="image-container">
                    <div class="image-label">Original Image</div>
                    {f'<img src="data:image/jpeg;base64,{original_image_base64}" />' if original_image_base64 else '<p>Image not available</p>'}
                </div>
                <div class="image-container">
                    <div class="image-label">Detected Workers</div>
                    {f'<img src="data:image/jpeg;base64,{annotated_image_base64}" />' if annotated_image_base64 else '<p>Image not available</p>'}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Report ID:</strong> {report_id}</p>
            <p><strong>Generated by:</strong> Build UK Standard Safety Compliance Checker</p>
            <p><strong>Verification:</strong> This report can be verified at safetycompliance.uk/verify/{report_id}</p>
            
            <div style="margin-top: 20px;">
                <p><strong>Inspector Signature:</strong></p>
                <div class="signature-line"></div>
                <p style="margin-top: 5px;">Date: _______________</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate PDF using weasyprint
    HTML(string=html_content).write_pdf(output_path)
    
    return output_path

