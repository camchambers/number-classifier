import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to import model from api
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from api.model import DigitClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_path=None):
    if model_path is None:
        # Default to model_best.pth in project root
        model_path = Path(__file__).parent.parent / 'model_best.pth'
    
    model = DigitClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(model, image_path):
    """Predict digit from image file"""
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item() * 100, probabilities[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
        print("Example: python test_model.py data/numbers/0/img_1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print(f"Loading model...")
    model = load_model()
    
    print(f"Testing image: {image_path}")
    predicted_digit, confidence, probabilities = predict_image(model, image_path)
    
    print(f"\n{'='*50}")
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"{'='*50}")
    
    print(f"\nAll class probabilities:")
    for digit in range(10):
        prob = probabilities[digit].item() * 100
        bar = '█' * int(prob / 2)
        print(f"  {digit}: {prob:5.2f}% {bar}")

if __name__ == "__main__":
    main()
