FROM python:3.11-slim

# Install system dependencies for Tesseract and OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Uninstall opencv-python if present, then install opencv-python-headless
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (.dockerignore will exclude large files)
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Use PORT environment variable (Railway provides this)
ENV PORT=5000
EXPOSE $PORT

CMD python app.py

