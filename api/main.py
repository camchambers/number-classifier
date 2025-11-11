from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
from pathlib import Path

# Import model from local file
from model import DigitClassifier

app = FastAPI(title="Digit Classifier API")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: dict[int, float]


class Base64ImageRequest(BaseModel):
    image: str  # base64 encoded image


def load_model():
    """Load the trained model"""
    global model
    model_path = Path(__file__).parent.parent / "model_best.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = DigitClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully from {model_path}")


def predict_image(image: Image.Image) -> PredictionResponse:
    """Make prediction on a PIL Image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Convert to dictionary
    probs_dict = {i: float(probabilities[0][i].item()) for i in range(10)}
    
    return PredictionResponse(
        predicted_digit=int(predicted.item()),
        confidence=float(confidence.item()),
        probabilities=probs_dict
    )


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Digit Classifier API",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image file
    
    Args:
        file: Image file (PNG, JPG, etc.)
    
    Returns:
        PredictionResponse with predicted digit, confidence, and probabilities
    """
    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        result = predict_image(image)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64ImageRequest):
    """
    Predict digit from base64 encoded image
    
    Args:
        request: JSON with base64 encoded image string
    
    Returns:
        PredictionResponse with predicted digit, confidence, and probabilities
    """
    try:
        # Decode base64 image
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Decode and open image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_image(image)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": ["/", "/predict", "/predict/base64", "/health"]
    }
