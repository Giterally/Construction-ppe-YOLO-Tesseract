from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from detector import SafetyComplianceDetector, create_annotated_image
from document_processor import DocumentProcessor
import os
import uuid
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = None

if supabase_url and supabase_key:
    try:
        # Create client without proxy to avoid compatibility issues
        supabase = create_client(
            supabase_url, 
            supabase_key,
            options={"db": {"schema": "public"}}
        )
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"⚠️  Supabase connection failed: {e}")
        # Try without options if that fails
        try:
            supabase = create_client(supabase_url, supabase_key)
            print("✅ Connected to Supabase (fallback method)")
        except Exception as e2:
            print(f"⚠️  Supabase connection failed (fallback): {e2}")
            supabase = None
else:
    print("⚠️  Supabase credentials not found. Results will not be saved to database.")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize OpenAI client for AI-powered OCR text cleanup (optional)
openai_client = None
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    try:
        from openai import OpenAI
        # Initialize without proxy to avoid Railway proxy issues
        openai_client = OpenAI(
            api_key=openai_api_key,
            http_client=None  # Let OpenAI handle HTTP client internally
        )
    except Exception as e:
        print(f"⚠️  OpenAI client initialization failed for detector: {e}")
        # Try with minimal config
        try:
            openai_client = OpenAI(api_key=openai_api_key)
        except Exception as e2:
            print(f"⚠️  OpenAI client initialization failed (fallback): {e2}")
            openai_client = None

# Initialize detector with optional OpenAI client for text cleanup
detector = SafetyComplianceDetector(openai_client=openai_client)

# Initialize document processor
doc_processor = DocumentProcessor()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Safety PPE Compliance Checker'})

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for safety compliance"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    try:
        # Save uploaded file with unique name
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        file_ext = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{unique_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Analyze image (pass unique_id for OCR visualization)
        results = detector.analyze_image(filepath, unique_id=unique_id)
        
        # Create annotated image
        annotated_filename = f"{unique_id}_annotated.{file_ext}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        create_annotated_image(filepath, results['detections'], annotated_path)
        
        # Add image URLs to results
        original_image_url = f"http://localhost:5001/api/images/{saved_filename}"
        annotated_image_url = f"http://localhost:5001/api/images/{annotated_filename}"
        results['original_image'] = f"/api/images/{saved_filename}"
        results['annotated_image'] = f"/api/images/{annotated_filename}"
        
        # Save to Supabase if connected
        results['document_provided'] = False
        save_analysis_to_db(results)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    """Serve uploaded/annotated images"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    """Get past safety compliance analyses"""
    if not supabase:
        return jsonify({'error': 'Supabase not configured'}), 503
    
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        response = supabase.table('safety_analyses')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(limit)\
            .offset(offset)\
            .execute()
        
        # Convert image URLs from absolute to relative for frontend
        analyses = response.data
        for analysis in analyses:
            # Convert original_image_url to original_image
            if 'original_image_url' in analysis and analysis['original_image_url']:
                analysis['original_image'] = analysis['original_image_url'].replace('http://localhost:5001', '')
            # Convert annotated_image_url to annotated_image
            if 'annotated_image_url' in analysis and analysis['annotated_image_url']:
                analysis['annotated_image'] = analysis['annotated_image_url'].replace('http://localhost:5001', '')
            # Convert image URLs in ocr_processing_steps from absolute to relative
            if 'ocr_processing_steps' in analysis and analysis['ocr_processing_steps']:
                for step in analysis['ocr_processing_steps']:
                    if 'image' in step and step['image']:
                        step['image'] = step['image'].replace('http://localhost:5001', '')
                    if 'highlighted_image' in step and step['highlighted_image']:
                        step['highlighted_image'] = step['highlighted_image'].replace('http://localhost:5001', '')
        
        return jsonify({
            'analyses': analyses,
            'count': len(analyses)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-all-analyses', methods=['DELETE'])
def delete_all_analyses():
    """Delete all safety compliance analyses"""
    if not supabase:
        return jsonify({'error': 'Supabase not configured'}), 503
    
    try:
        # Delete all analyses by fetching all IDs and deleting them individually
        # This works better with RLS policies
        all_analyses = supabase.table('safety_analyses')\
            .select('id')\
            .execute()
        
        count = len(all_analyses.data) if all_analyses.data else 0
        deleted_count = 0
        
        if all_analyses.data:
            for analysis in all_analyses.data:
                try:
                    supabase.table('safety_analyses')\
                        .delete()\
                        .eq('id', analysis['id'])\
                        .execute()
                    deleted_count += 1
                except Exception as e2:
                    print(f"Error deleting analysis {analysis['id']}: {e2}")
                    continue
        
        return jsonify({
            'success': True, 
            'message': 'All analyses deleted successfully', 
            'count': count,
            'deleted': deleted_count
        })
    except Exception as e:
        print(f"Error deleting all analyses: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to delete all analyses: {str(e)}'}), 500

@app.route('/api/analyses/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get a specific safety compliance analysis by ID"""
    if not supabase:
        return jsonify({'error': 'Supabase not configured'}), 503
    
    try:
        response = supabase.table('safety_analyses')\
            .select('*')\
            .eq('id', analysis_id)\
            .single()\
            .execute()
        
        return jsonify(response.data)
    except Exception as e:
        return jsonify({'error': 'Analysis not found'}), 404

@app.route('/api/analyses/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete a specific safety compliance analysis by ID"""
    # Prevent "all" from being treated as an ID
    if analysis_id == 'all':
        return jsonify({'error': 'Use /api/analyses/all endpoint to delete all analyses'}), 400
    
    if not supabase:
        return jsonify({'error': 'Supabase not configured'}), 503
    
    try:
        # Delete the analysis
        response = supabase.table('safety_analyses')\
            .delete()\
            .eq('id', analysis_id)\
            .execute()
        
        return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
    except Exception as e:
        print(f"Error deleting analysis: {e}")
        return jsonify({'error': 'Failed to delete analysis'}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of available UK construction documents for selection"""
    from document_processor import UK_CONSTRUCTION_DOCUMENTS
    
    documents = []
    for doc_id, doc_info in UK_CONSTRUCTION_DOCUMENTS.items():
        documents.append({
            'id': doc_id,
            'name': doc_info['name'],
            'description': doc_info['description'],
            'official_link': doc_info.get('official_link', '')
        })
    
    return jsonify({'documents': documents})

@app.route('/api/documents/<document_id>/requirements', methods=['GET'])
def get_document_requirements(document_id):
    """Get full requirements for a specific document"""
    from document_processor import UK_CONSTRUCTION_DOCUMENTS, DocumentProcessor
    
    if document_id not in UK_CONSTRUCTION_DOCUMENTS:
        return jsonify({'error': 'Document not found'}), 404
    
    doc_info = UK_CONSTRUCTION_DOCUMENTS[document_id]
    processor = DocumentProcessor()
    requirements = processor.get_document_requirements(document_id)
    
    return jsonify({
        'document_id': document_id,
        'document_name': doc_info['name'],
        'official_link': doc_info.get('official_link', ''),
        'requirements': requirements
    })

def save_analysis_to_db(results: dict):
    """Save analysis to Supabase - handles both standard and enhanced results"""
    if not supabase:
        return
    
    try:
        # Handle image URLs - convert relative to absolute
        original_image = results.get('original_image', '')
        annotated_image = results.get('annotated_image', '')
        
        if original_image.startswith('/api/images/'):
            original_image_url = f"http://localhost:5001{original_image}"
        elif original_image.startswith('http://'):
            original_image_url = original_image
        else:
            original_image_url = f"http://localhost:5001/api/images/{original_image}"
        
        if annotated_image.startswith('/api/images/'):
            annotated_image_url = f"http://localhost:5001{annotated_image}"
        elif annotated_image.startswith('http://'):
            annotated_image_url = annotated_image
        else:
            annotated_image_url = f"http://localhost:5001/api/images/{annotated_image}"
        
            # Get OCR processing steps from results
            ocr_steps = results.get('ocr_processing_steps')
            print(f"🔍 DEBUG: ocr_processing_steps in results: {ocr_steps is not None}, type: {type(ocr_steps)}")
            if ocr_steps:
                print(f"🔍 DEBUG: ocr_steps length: {len(ocr_steps) if isinstance(ocr_steps, list) else 'not a list'}")
            if ocr_steps is None:
                ocr_steps = []
            
            data = {
                'original_image_url': original_image_url,
                'annotated_image_url': annotated_image_url,
                'people_count': results.get('people_count', 0),
                'signage_text': results.get('signage_text', ''),
                'violations': results.get('violations', []),
                'detections': results.get('detections', []),
                'compliance_score': results.get('compliance_score', 0),
                'document_provided': results.get('document_provided', False),
                'document_id': results.get('document_id'),
                'document_name': results.get('document_name'),
                'document_requirements': results.get('document_requirements'),
                'cross_check': results.get('cross_check'),
                'ocr_processing_steps': ocr_steps if ocr_steps else None
            }
            
            # Debug: Print OCR steps to verify they're being saved
            print(f"🔍 DEBUG: data['ocr_processing_steps'] = {data.get('ocr_processing_steps')}")
            if ocr_steps:
                print(f"💾 Saving {len(ocr_steps)} OCR processing steps to database")
            else:
                print(f"⚠️  No OCR processing steps found in results")
        
        result = supabase.table('safety_analyses').insert(data).execute()
        print(f"✅ Saved analysis to Supabase, inserted: {result.data[0]['id'] if result.data else 'unknown'}")
    except Exception as e:
        print(f"⚠️  Failed to save to Supabase: {e}")

@app.route('/api/analyze-with-document', methods=['POST'])
def analyze_with_document():
    """
    Enhanced analysis that cross-checks photo against selected UK construction document
    Accepts: image file + optional document_id (from predefined list)
    """
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid image file type'}), 400
    
    try:
        # Save image
        filename = secure_filename(image_file.filename)
        unique_id = str(uuid.uuid4())
        file_ext = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{unique_id}.{file_ext}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        image_file.save(image_path)
        
        # Standard photo analysis (existing functionality)
        photo_results = detector.analyze_image(image_path, unique_id=unique_id)
        
        # Create annotated image
        annotated_filename = f"{unique_id}_annotated.{file_ext}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        create_annotated_image(image_path, photo_results['detections'], annotated_path)
        
        photo_results['original_image'] = f"/api/images/{saved_filename}"
        photo_results['annotated_image'] = f"/api/images/{annotated_filename}"
        
        # If document selected, do cross-checking
        document_id = request.form.get('document_id')  # Get selected document ID
        
        if document_id:
            from document_processor import UK_CONSTRUCTION_DOCUMENTS
            
            if document_id not in UK_CONSTRUCTION_DOCUMENTS:
                return jsonify({'error': 'Invalid document selection'}), 400
            
            # Get requirements for selected document
            doc_info = UK_CONSTRUCTION_DOCUMENTS[document_id]
            doc_requirements = doc_processor.get_document_requirements(document_id)
            
            # Cross-check photo vs document
            cross_check = doc_processor.cross_check_with_photo_analysis(
                photo_results, 
                doc_requirements,
                doc_info['name']
            )
            
            # Enhanced results
            enhanced_results = {
                **photo_results,
                'document_provided': True,
                'document_id': document_id,
                'document_name': doc_info['name'],
                'document_description': doc_info['description'],
                'document_requirements': doc_requirements,
                'cross_check': cross_check
            }
            
            # Save to Supabase with document data
            save_analysis_to_db(enhanced_results)
            
            return jsonify(enhanced_results)
        
        else:
            # No document - return standard analysis
            photo_results['document_provided'] = False
            save_analysis_to_db(photo_results)
            return jsonify(photo_results)
    
    except Exception as e:
        print(f"Error in analyze_with_document: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("🚀 Starting Safety PPE Compliance Checker API...")
    print(f"📍 Server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

