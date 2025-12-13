# Safety PPE Compliance Checker

Automated construction site safety compliance checker using YOLOv8 and Tesseract OCR.

## Features

- **Worker Detection**: YOLOv8 detects people in construction site images
- **Signage Reading**: Tesseract OCR extracts text from safety signs
- **Compliance Analysis**: Automated checking for safety violations
- **Visual Results**: Annotated images with bounding boxes
- **Swiss Design UI**: Clean, minimal interface
- **Database Storage**: Supabase integration to store past analyses

## Tech Stack

### Backend
- Python 3.11
- Flask
- YOLOv8 (ultralytics)
- Tesseract OCR
- OpenCV
- Supabase (database storage)

### Frontend
- React 18
- TypeScript
- Vite
- Pure CSS (Swiss Design)

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OR Python 3.11 + Node.js 20

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to project
cd safety-ppe-checker

# Start all services
docker-compose up --build

# Frontend: http://localhost:3000
# Backend: http://localhost:5000
```

### Option 2: Local Development

**Backend:**
```bash
cd backend

# Install Tesseract (Mac)
brew install tesseract

# Install Python dependencies
pip install -r requirements.txt

# Set up Supabase credentials
# Copy env.example to .env and add your Supabase URL and API key
cp env.example .env
# Edit .env with your Supabase credentials

# Run server
python app.py
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

## Usage

1. Open http://localhost:3000
2. Upload construction site image (PNG/JPG/WEBP)
3. Wait for analysis (~2-5 seconds)
4. Review compliance report with:
   - Worker count
   - Detected signage
   - Potential violations
   - Annotated images

## API Endpoints

### `POST /api/analyze`
Upload image for analysis

**Request:**
- `multipart/form-data`
- Field: `image` (file)

**Response:**
```json
{
  "people_count": 3,
  "signage_text": "HARD HAT REQUIRED",
  "violations": ["Hard hat required zone: 3 worker(s) detected"],
  "compliance_score": 70,
  "detections": [...],
  "original_image": "/api/images/...",
  "annotated_image": "/api/images/..."
}
```

### `GET /api/images/<filename>`
Retrieve uploaded/annotated images

### `GET /api/analyses`
Get past safety compliance analyses

**Query Parameters:**
- `limit` (optional): Number of results to return (default: 20)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "analyses": [...],
  "count": 10
}
```

### `GET /api/analyses/<analysis_id>`
Get a specific analysis by ID

## Technical Details

### YOLO Detection
- Model: YOLOv8 Nano (yolov8n.pt)
- Detects: People (COCO class 0)
- Confidence threshold: 0.5

### OCR Processing
- Engine: Tesseract 5.x
- Extracts all visible text
- Matches against safety keywords

### Compliance Rules
- Hard hat requirements
- High-visibility clothing
- Restricted area access
- Hazard zone presence

## Project Structure

```
project-root/
├── backend/
│   ├── app.py              # Flask API
│   ├── detector.py         # YOLO + Tesseract logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── styles/
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Built For

Interface.ai technical demonstration  
Uses their production stack: YOLO, Tesseract, Python, TypeScript, Docker

## License

MIT

