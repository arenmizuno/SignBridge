# SignBridge

## ADSP 31018 – Machine Learning II
### Final Project – University of Chicago  

**Instructor:** Batuhan Gundogdu, Gregory Green  
**Term:** Spring 2026  

---

## Team Members

- Aren Mizuno  
- Arthur Acker
- Jaysen Jensen 
- Lawrence Lin  

---

# Overview

SignBridge is a deep learning project focused on American Sign Language (ASL) translation using landmark-based sequence modeling. The project explores multiple neural network architectures for temporal sign classification using MediaPipe landmark sequences extracted from video.

The repository includes:
- Full preprocessing and dataset generation pipelines
- Exploratory data analysis workflows
- Partial and full dataset model training
- Transformer-based sequence modeling
- ONNX export and deployment preparation
- Real-time inference preparation using MediaPipe

The project was developed using PyTorch, Transformers, CNNs, GRUs, TCNs, and hybrid sequence architectures.

---

# Repository Structure

```text
SignBridge/
│
├── src/
│   ├── 00_Preprocessing.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_Partial_Model_Training.ipynb
│   ├── 03_Full_Model_Training.ipynb
│   └── 04_Transformer_Full_Training.ipynb
│
├── data/
│   ├── external/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   ├── logs/
│   └── onnx/
│
├── results/
│   ├── graphs/
│   ├── reports/
│   └── tables/
│
└── README.md
```

---

# Dataset

This project uses the Kaggle ASL Signs competition dataset.

Dataset:
- 250 ASL sign classes
- Landmark-based temporal sequences
- Hand, pose, and facial keypoints
- Variable-length sign sequences

Kaggle Competition:
- ASL Signs Dataset
- https://www.kaggle.com/competitions/asl-signs

---

# Important Repository Notes

Large datasets and trained model checkpoints are **not included** in this GitHub repository due to GitHub file size limitations.

The following folders are excluded from version control:

```text
data/
models/checkpoints/
models/onnx/
```

These directories contain:
- Processed `.npz` and `.npy` arrays
- Full PyTorch checkpoints (`.pt`)
- ONNX deployment exports
- Quantized inference models

---

# Reproducing the Dataset

To regenerate the processed dataset:

1. Download the Kaggle ASL Signs dataset
2. Place the zip of the dataset into:

```text
data/external/
```

3. Run:

```text
src/00_Preprocessing.ipynb
```

The preprocessing notebook will:
- Clean landmark sequences
- Normalize features
- Generate train/validation/test splits
- Save processed arrays
- Create metadata and label mappings

Processed outputs will be saved into:

```text
data/processed/
```

---

# Pipeline Overview

## 1. Preprocessing

Notebook:
```text
src/00_Preprocessing.ipynb
```

The preprocessing pipeline:
- Loads raw landmark sequences
- Cleans invalid samples
- Pads variable-length sequences
- Normalizes landmark features
- Creates participant-aware splits
- Saves processed training arrays

---

## 2. Exploratory Data Analysis

Notebook:
src/01_EDA.ipynb

EDA includes:
- Dataset dimensionality and split verification
- Class balance analysis across 250 ASL signs
- Sequence length distribution visualization
- Feature mean and standard deviation analysis
- Missing value and infinity checks
- Normalization validation
- Participant-aware split validation

Dataset summary:

| Split | Samples | Sequence Length | Features |
|-------|---------:|----------------:|----------:|
| Train | 55,642 | 96 | 708 |
| Validation | 9,291 | 96 | 708 |
| Test | 11,826 | 96 | 708 |

Additional statistics:
- Total classes: 250
- No NaN values detected across any split
- No infinite values detected across any split
- Landmark sequences were padded/truncated to 96 frames
- Each frame contains 708 landmark features from pose, hand, and facial keypoints

Key findings:
- Sequence lengths are highly right-skewed, with most signs occurring in shorter sequences while a large number of samples reach the 96-frame padding limit
- Train, validation, and test distributions remain visually consistent, indicating stable participant-aware splitting
- Feature standard deviation analysis revealed multiple feature groups with different variability ranges, reflecting differences between hand, pose, and facial landmark movement
- Normalization successfully standardized the majority of landmark features while preserving temporal variation between signs
- No missing or corrupted numerical values were detected after preprocessing, confirming dataset integrity before model training

EDA visualizations:
- Sequence length histograms
- Feature standard deviation distributions
- Split distribution comparisons
- Landmark variability analysis

Outputs are saved into:

results/graphs/
results/tables/

---

## 3. Partial Dataset Training

Notebook:
```text
src/02_Partial_Model_Training.ipynb
```

Used for:
- Faster experimentation
- Architecture comparison
- Hyperparameter tuning
- Stability testing

Models explored:
- MLP
- CNN
- LSTM
- GRU
- CNN + GRU
- Transformer
- CNN + Transformer
- TCN

---

## 4. Full Dataset Training

Notebook:
```text
src/03_Full_Model_Training.ipynb
```

Final large-scale training on the complete 250-sign dataset.

Primary architectures:
- CNN
- CNN + GRU
- TCN
- Transformer

---

## 5. Transformer Full Training

Notebook:
```text
src/04_Transformer_Full_Training.ipynb
```

Focused on:
- Transformer encoder architectures
- Attention-based temporal modeling
- CNN + Transformer hybrids
- ONNX export and deployment preparation

---

# Data Augmentation

The project uses several augmentation methods:

- Temporal stretching
- Gaussian landmark noise
- Random frame dropout
- Sequence masking

These augmentations improve robustness to signer variation and temporal inconsistencies.

---

# Evaluation

Models are evaluated using:
- Top-1 Accuracy
- Top-5 Accuracy
- Precision / Recall / F1
- Confusion matrices
- Per-class accuracy analysis

Outputs are saved into:

```text
results/reports/
results/tables/
results/graphs/
```

---

# Model Exporting

The repository supports multiple deployment formats.

## ONNX
Used for optimized inference and cross-platform deployment.

## TorchScript
Used for PyTorch deployment compatibility.

## Quantization
Dynamic quantization was explored for:
- Smaller model size
- Faster CPU inference
- Mobile deployment support

---

# Planned Real-Time Pipeline

```text
Webcam
   ↓
OpenCV Frames
   ↓
MediaPipe Landmark Extraction
   ↓
Landmark Sequence Buffer
   ↓
Neural Network Inference
   ↓
Predicted Sign
   ↓
Live Translation UI
```

---

# Technologies Used

## Deep Learning
- PyTorch
- Transformers
- Temporal CNNs
- GRUs / LSTMs

## Data Processing
- NumPy
- Pandas
- Scikit-learn

## Visualization
- Matplotlib

## Computer Vision
- OpenCV
- MediaPipe

## Deployment
- ONNX Runtime
- TorchScript

---
