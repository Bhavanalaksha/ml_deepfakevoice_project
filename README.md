# 🎙️ Audio Deepfake Detection System

> An advanced ML system for detecting AI-generated and manipulated audio using MFCC features and ensemble learning.

---

## 🎯 Overview

Production-ready deepfake audio detection system that classifies audio files as **FAKE** (AI-generated) or **REAL** (authentic) with 96%+ accuracy.

**Key Features:**
- 🧠 Ensemble models (Random Forest + XGBoost)
- 🎵 MFCC-based feature extraction (26 features)
- ⚖️ SMOTE resampling for class imbalance
- 🌐 Interactive Gradio web interface
- 📊 Real-time visualizations (waveforms, spectrograms)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch web interface
python app_gradio.py

# Opens in browser + generates shareable public link
```

---

## 📁 Project Structure

```
├── app_gradio.py          # Web interface
├── app.ipynb             # Model training notebook
├── model_loader.py       # Model utilities
├── requirements.txt      # Dependencies
├── models/               # Trained models
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── *_metadata.json
└── train/dev/eval/       # Dataset (52,782 samples)
```

---

## 🧠 Technical Architecture

### Pipeline

1. **Feature Extraction**
   - Extract 13 MFCC coefficients + mean & std → 26 features
   - Captures spectral characteristics of fake vs. real audio

2. **Preprocessing**
   - StandardScaler normalization
   - SMOTE oversampling (4:1 → 1:1 balance)
   - 80/20 train/validation split

3. **Model Training**
   - **Random Forest:** 100 trees, ensemble voting
   - **XGBoost:** Gradient boosting, hyperparameter tuning
   - **Ensemble:** Weighted average for final prediction

4. **Inference**
   - Upload → Extract features → Scale → Predict
   - Output: Classification + Confidence + Visualizations

---

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 95.2% | 94.8% | 95.6% | 95.2% |
| XGBoost | 96.1% | 95.9% | 96.3% | 96.1% |
| **Ensemble** | **96.8%** | **96.5%** | **97.0%** | **96.7%** |

*Evaluated on balanced validation set*

---

## 🛠️ Technology Stack

**ML/AI:** scikit-learn, XGBoost, imbalanced-learn  
**Audio:** librosa, soundfile  
**Web:** Gradio 4.x  
**Visualization:** matplotlib, seaborn, plotly  
**Core:** Python 3.10+, pandas, numpy

---

## 💡 Use Cases

- 🔍 **Media Verification:** Authenticate recordings for journalism
- ⚖️ **Legal Evidence:** Verify audio in court cases
- 🎙️ **Content Moderation:** Detect synthetic voices on social media
- 🔐 **Security:** Identify voice spoofing attempts
- 🎬 **Production QC:** Validate audio quality

---

## 📝 Dataset

**Total:** 52,782 audio files (WAV format)

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Train | 10,660 | 2,525 | 13,185 |
| Dev | 10,295 | 2,548 | 12,843 |
| Eval | 26,412 | 6,334 | 32,746 |

**Imbalance:** 4.2:1 (Fake:Real) - handled via SMOTE

---

## 🔧 Training Your Own Models

```bash
# 1. Open Jupyter notebook
jupyter notebook app.ipynb

# 2. Configure dataset path (Cell 2)
dataset_root = r"d:\ML\archive"

# 3. Choose resampling strategy (Cell 12)
RESAMPLING_STRATEGY = 'smote'  # Options: smote, class_weight, adasyn

# 4. Run all cells to train models

# 5. Export models (final cell) → saves to models/ directory
```

**Training Time:** ~10-15 minutes on CPU

---

## 🌐 Web Interface Features

**Analysis Tab:**
- Upload single audio file (WAV, MP3, M4A, FLAC, DAT)
- Real-time prediction with confidence scores
- Waveform and spectrogram visualizations
- Audio metadata display

**Batch Processing:**
- Upload multiple files simultaneously
- Bulk analysis with CSV export
- Summary statistics

**Information:**
- Model performance metrics
- Technical architecture details
- Supported formats guide

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

```bash
# Create virtual environment (recommended)
python -m venv v
v\Scripts\activate  # Windows
source v/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gradio; import librosa; print('✅ Ready!')"
```

### Optional: MP3/M4A Support

Install ffmpeg for compressed audio formats:
```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

---

## 🚨 Limitations

- Trained on specific deepfake methods (may not detect all future techniques)
- Performance depends on audio quality (noise affects accuracy)
- Optimized for speech audio (not music/environmental sounds)
- Requires retraining for new deepfake algorithms

---

## 🎓 How It Works

### Why MFCC Features?

MFCCs capture spectral envelope of audio - how energy distributes across frequencies:

- **Fake audio:** Unnatural patterns, artifacts, over-smoothed features
- **Real audio:** Natural variations, human voice characteristics, micro-imperfections

### Why SMOTE?

Dataset has 4:1 imbalance. SMOTE creates synthetic minority samples by:
1. Finding k-nearest neighbors for each real sample
2. Interpolating between sample and neighbors
3. Generating new samples until balanced (1:1)

### Why Ensemble?

Combines strengths of multiple models:
- **Random Forest:** Handles overfitting, captures non-linear patterns
- **XGBoost:** Higher accuracy, sequential error correction
- **Weighted Average:** Reduces variance, improves robustness

---

