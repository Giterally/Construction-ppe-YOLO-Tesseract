#!/bin/bash
# Start script for backend with WeasyPrint library paths

cd "$(dirname "$0")"
source venv/bin/activate

# Set library paths for WeasyPrint
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"

# Start the Flask app
python app.py

