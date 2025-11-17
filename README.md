# Handwritten Digit Recognition using OpenCV and CNN

## Project Overview

This is an **Advanced Handwritten Digit Recognition System** built with **Convolutional Neural Networks (CNN)** and **OpenCV**, featuring an interactive **Tkinter GUI** for real-time digit recognition.

### Key Features
✓ **Advanced CNN Architecture** with 99.5%+ accuracy on MNIST dataset  
✓ **Batch Normalization** for stable training  
✓ **Data Augmentation** for robust model  
✓ **Dropout Layers** to prevent overfitting  
✓ **Interactive Tkinter GUI** with canvas drawing  
✓ **Real-time Prediction** with confidence percentage  
✓ **Automatic MNIST Dataset Download** (60,000 training images)  
✓ **Model Persistence** - Save and load trained models  

---

## System Requirements

- **OS:** Windows 11 (or any Windows version with Python 3.10+)
- **Python Version:** 3.10 or higher
- **IDE:** Visual Studio Code (recommended)
- **RAM:** Minimum 4GB (8GB recommended)
- **Disk Space:** Minimum 2GB free space
- **GPU:** Optional (CPU training is also supported)

---

## Installation Guide

### Quick Installation (Recommended)

1. **Open Command Prompt** as Administrator

2. **Navigate to project directory:**
```bash
cd path/to/project
```

3. **Install all packages at once:**
```bash
pip install -r requirements.txt
```

### Manual Step-by-Step Installation

If you prefer to install packages individually:

```bash
# Step 1: Upgrade pip
python -m pip install --upgrade pip

# Step 2: Install dependencies in order
pip install numpy==2.2.6
pip install Pillow==12.0.0
pip install matplotlib==3.10.7
pip install scikit-learn==1.7.2
pip install opencv-python
pip install tensorflow==2.20.0
pip install Flask==3.1.2

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

---

## Project Structure

```
project-folder/
│
├── digit_recognition.py       # Main application (FULL SOURCE CODE)
├── requirements.txt            # Package dependencies
├── installation_guide.md       # Detailed installation steps
├── quick_start.md             # Quick start guide
└── digit_model.h5             # Trained model (created after first run)
```

---

## How to Run

### Method 1: From VS Code

1. Open the project folder in **Visual Studio Code**
2. Open `digit_recognition.py`
3. Press **Ctrl + F5** or click "Run"

### Method 2: From Command Prompt

```bash
python digit_recognition.py
```

### Method 3: From Python Interpreter

```python
import digit_recognition
```

---

## First Run Behavior

**When you run the application for the first time:**

1. **MNIST Dataset Download** (50-100MB)
   - Automatic download from TensorFlow/Keras
   - Downloaded to `~/.keras/datasets/`

2. **Model Training** (2-5 minutes depending on CPU/GPU)
   - Advanced CNN training on 60,000 MNIST images
   - 50 epochs with data augmentation
   - Real-time accuracy reporting

3. **Model Saving**
   - Trained model saved as `digit_model.h5`
   - Next runs will load pre-trained model instantly

4. **GUI Launch**
   - Interactive Tkinter window appears
   - Ready for digit drawing and recognition

---

## User Interface Guide

### Drawing Canvas (Left Side)
- **Draw** any digit (0-9) using your mouse
- **Line Width:** 8 pixels for optimal recognition
- **Background:** White canvas

### Recognition Panel (Right Side)
- **Prediction Result:** Large display of recognized digit
- **Confidence Score:** Percentage of model confidence (0-100%)
- **Prediction Details:** Additional information and statistics

### Controls
- **Recognize Button (Green):** Predict the drawn digit
- **Clear Button (Red):** Clear canvas and start fresh

### Status Bar
- Shows current system status and messages
- Indicates when processing is complete

---

## Model Architecture

### CNN Layers
```
Input (28x28x1)
    ↓
Conv2D (32 filters) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Conv2D (64 filters) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Conv2D (128 filters) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Flatten
    ↓
Dense (256) + BatchNorm + ReLU + Dropout
    ↓
Dense (128) + BatchNorm + ReLU + Dropout
    ↓
Output Dense (10 classes) + Softmax
```

### Hyperparameters
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 128
- **Epochs:** 50
- **Dropout Rate:** 0.25-0.5
- **Data Augmentation:** Rotation, Shift, Zoom, Shear

---

## Expected Accuracy

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.8%+ |
| Validation Accuracy | 99.6%+ |
| Test Accuracy | 99.5%+ |
| GUI Prediction Accuracy | ~95-98%* |

*GUI accuracy depends on drawing clarity and precision

---

## Dataset Information

### MNIST Dataset
- **Training Images:** 60,000 (28x28 grayscale)
- **Test Images:** 10,000 (28x28 grayscale)
- **Classes:** 10 (digits 0-9)
- **Source:** Automatic download from TensorFlow/Keras
- **Download Size:** ~50-100MB
- **Extracted Size:** ~200-300MB

### Dataset Preprocessing
- Normalization (0-255 → 0-1)
- Channel dimension addition (28x28 → 28x28x1)
- One-hot encoding for labels
- Data augmentation (rotation, shift, zoom)

---

## Troubleshooting

### Issue: "Module not found" error
**Solution:** Reinstall the missing package
```bash
pip install --no-cache-dir tensorflow==2.20.0
```

### Issue: GUI doesn't appear
**Solution:** Check if tkinter is installed (comes with Python)
```bash
python -m tkinter
```

### Issue: Slow prediction
**Solution:** This is normal for first prediction (model initialization)

### Issue: MNIST download fails
**Solution:** Check internet connection or download manually
```bash
import tensorflow as tf
tf.keras.datasets.mnist.load_data()
```

### Issue: Out of memory error
**Solution:** Reduce batch size in code or close other applications

---

## Advanced Usage

### Train with Custom Parameters

Edit `digit_recognition.py` line where model is trained:
```python
model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=64)
```

### Load Pre-trained Model

```python
from digit_recognition import AdvancedDigitRecognitionModel
model = AdvancedDigitRecognitionModel()
model.load_model()
```

### Make Predictions Programmatically

```python
import numpy as np
from digit_recognition import AdvancedDigitRecognitionModel

model = AdvancedDigitRecognitionModel()
model.load_model()

# Your image array (28x28 normalized)
image = np.random.rand(28, 28)
digit, confidence = model.predict(image)
print(f"Predicted: {digit}, Confidence: {confidence:.2f}%")
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `digit_recognition.py` | Main application with complete source code |
| `requirements.txt` | All package dependencies for easy installation |
| `installation_guide.md` | Step-by-step installation instructions |
| `digit_model.h5` | Trained model (created after first run) |

---

## Performance Tips

1. **First Run:** Takes 2-5 minutes for training (normal)
2. **Subsequent Runs:** Loads pre-trained model instantly
3. **Drawing:** Draw clearly and fill the digit well
4. **GPU:** TensorFlow will auto-detect and use GPU if available
5. **Batch Size:** Larger batch size uses more memory but trains faster

---

## Requirements Verification

After installation, verify everything is working:

```bash
# Open Python interactive shell
python

# Run verification
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tkinter

print("✓ All packages installed successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
```

---

## Support & Documentation

### Code Comments
- Every function has detailed docstrings
- Inline comments explain complex logic
- Class documentation provided

### References
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras API: https://keras.io/
- OpenCV Documentation: https://docs.opencv.org/

---

## License & Credits

**Project:** Advanced Handwritten Digit Recognition System  
**Version:** 1.0  
**Date:** November 2025  
**Target Accuracy:** 99.5%+  

---

## Important Notes

⚠️ **First Run Only:**
- MNIST dataset auto-downloads (~50-100MB)
- Model training takes 2-5 minutes
- Be patient - this is one-time setup

✓ **After First Run:**
- Pre-trained model loads instantly
- Predictions are immediate
- Full GUI responsiveness

📊 **Expected Output:**
- Recognized digit (0-9)
- Confidence percentage (0-100%)
- Status messages for user feedback

---

## Questions & Issues

If you encounter any issues:
1. Check all packages are installed correctly
2. Verify Python version (should be 3.10+)
3. Ensure sufficient disk space (2GB minimum)
4. Check internet connection (for dataset download)
5. Review error messages in terminal

---

**Enjoy your Handwritten Digit Recognition System!** 🎉
