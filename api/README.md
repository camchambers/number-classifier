# Digit Classifier API

FastAPI backend for serving the digit classification model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the trained model exists:
```bash
# Should have model_best.pth in parent directory
ls ../model_best.pth
```

## Running the API

### Development mode (with auto-reload):
```bash
uvicorn main:app --reload
```

### Production mode:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Health check endpoint
```bash
curl http://localhost:8000/
```

### POST `/predict`
Upload an image file for prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### POST `/predict/base64`
Send base64 encoded image for prediction
```bash
curl -X POST "http://localhost:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KG..."}'
```

### GET `/health`
Detailed health check
```bash
curl http://localhost:8000/health
```

## Response Format

```json
{
  "predicted_digit": 5,
  "confidence": 0.9987,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.0003,
    "3": 0.0001,
    "4": 0.0004,
    "5": 0.9987,
    "6": 0.0001,
    "7": 0.0000,
    "8": 0.0001,
    "9": 0.0000
  }
}
```

## Interactive Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## CORS Configuration

The API currently allows all origins (`*`). For production, update the CORS middleware in `main.py` to specify your frontend URL:

```python
allow_origins=["http://localhost:5173"]  # Your frontend URL
```
