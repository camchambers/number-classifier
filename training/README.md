# Training Scripts

This directory contains all training-related code for the Digit Classifier model.

## Files

- **`train.py`** - Main training script
- **`test_model.py`** - Test script to evaluate trained model on individual images

## Model Architecture

The model architecture is defined in `api/model.py` and shared across training and inference.

## Training

Run training from the project root or training directory:

```bash
# From project root
python3 training/train.py

# From training directory
cd training
python3 train.py
```

**Output:**
- `model_best.pth` - Best model checkpoint (saved to project root)
- `model_final.pth` - Final model after all epochs (saved to project root)

## Testing

Test the trained model on individual images:

```bash
# From project root
python3 training/test_model.py data/numbers/5/img_1.jpg

# From training directory
cd training
python3 test_model.py ../data/numbers/5/img_1.jpg
```

## Requirements

All dependencies are managed at the project level. See root `requirements.txt`.
