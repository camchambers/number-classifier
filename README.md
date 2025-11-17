# Number Classifier

A handwritten digit classification system with a PyTorch CNN model, FastAPI backend, and React frontend.

## Features

- PyTorch CNN trained on handwritten digits
- FastAPI backend for model inference
- React UI with canvas for drawing digits
- ~98% validation accuracy

## Project Status

- Model training and testing complete
- FastAPI backend operational
- React frontend with canvas drawing
- End-to-end integration working

## Dataset

Uses the [Handwritten Digits 0-9 dataset from Kaggle](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download).

- ~21,555 grayscale images
- 10 classes (digits 0-9)
- Located in `data/numbers/`

## Project Structure

```
number-classifier/
├── training/             # Training scripts and utilities
│   ├── train.py         # Model training script
│   ├── test_model.py    # Model testing script
│   └── README.md        # Training documentation
├── api/                  # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── model.py         # Model architecture
│   ├── requirements.txt # API dependencies
│   ├── run.sh           # API startup script
│   └── README.md        # API documentation
├── ui/                   # React frontend (Vite)
│   ├── src/
│   │   ├── components/  # React components
│   │   │   ├── App.tsx
│   │   │   ├── Canvas.tsx
│   │   │   └── ClassificationResult.tsx
│   │   ├── pages/       # Page components
│   │   │   └── Home.tsx
│   │   ├── api/         # API client
│   │   │   └── classifier.ts
│   │   ├── utils/       # Utility functions
│   │   │   └── imageUtils.ts
│   │   └── styles/      # CSS styles
│   └── package.json
├── data/
│   └── numbers/         # Training dataset (0-9 folders)
├── models/
│   ├── model_best.pth   # Best trained model checkpoint
│   └── model_final.pth  # Final model after all epochs
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+ (for UI)
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/camchambers/number-classifier.git
   cd number-classifier
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download)
   - Extract to `data/numbers/` directory

## Training

Train the CNN model on the digit dataset:

```bash
python3 training/train.py
```

**Training Details:**
- Architecture: 3 Conv layers + 3 FC layers (defined in `api/model.py`)
- Input: 28x28 grayscale images
- Batch size: 64
- Epochs: 10
- Optimizer: Adam (lr=0.001)
- Validation split: 80/20

**Output:**
- `models/model_best.pth` - Best model checkpoint (highest validation accuracy during training)
- `models/model_final.pth` - Final model state after all epochs complete

**Note:** The API uses `model_best.pth` for inference as it typically generalizes better than the final model.

## Testing

Test the trained model on individual images:

```bash
python3 training/test_model.py data/numbers/5/img_1.jpg
```

**Output:**
```
==================================================
Predicted Digit: 5
Confidence: 99.87%
==================================================

All class probabilities:
  0:  0.01% 
  1:  0.02% 
  2:  0.03% 
  3:  0.01% 
  4:  0.04% 
  5: 99.87% ██████████████████████████████████████████████████
  6:  0.01% 
  7:  0.00% 
  8:  0.01% 
  9:  0.00% 
```

## Running the Application

### 1. Start the API (FastAPI)

```bash
cd api
./run.sh
```

Or manually:
```bash
cd api
python3 -m uvicorn main:app --reload
```

API will be available at `http://localhost:8000`  
API Documentation: `http://localhost:8000/docs`

### 2. Start the UI (React + Vite)

```bash
cd ui
npm install
npm run dev
```

UI will be available at `http://localhost:5173`

### 3. Use the Application

1. Open `http://localhost:5173` in your browser
2. Draw a digit (0-9) on the canvas using your mouse
3. Click "Classify" to send the drawing to the API
4. View the predicted digit with confidence score and probability distribution
5. Click "Clear" to reset and try another digit

## Model Architecture

```
DigitClassifier(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), padding=1)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=1)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), padding=1)
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (dropout): Dropout(p=0.5)
  (fc1): Linear(in_features=1152, out_features=256)
  (fc2): Linear(in_features=256, out_features=128)
  (fc3): Linear(in_features=128, out_features=10)
)
```

**Total Parameters**: ~550,000

## Dependencies

### Python
- torch >= 2.0.0
- torchvision >= 0.15.0
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- pillow >= 10.0.0
- numpy >= 1.24.0

### JavaScript
- React 18
- TypeScript
- Vite
- (See `ui/package.json` for full list)

## UI Features

- Canvas for drawing digits
- Real-time classification
- Confidence scores and probability distribution
- Error handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Cam Chambers**
- GitHub: [@camchambers](https://github.com/camchambers)

## Acknowledgments

Dataset: [Olaf Krastovski's Handwritten Digits 0-9](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9)