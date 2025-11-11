#!/bin/bash

# Digit Classifier API Startup Script

echo "Starting Digit Classifier API..."
echo "================================"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Start uvicorn with auto-reload
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
