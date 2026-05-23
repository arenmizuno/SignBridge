# SignBridge — ASL Translation Using Deep Learning

## Overview

SignBridge is a deep learning project focused on real-time American Sign Language (ASL) translation using hand and body landmark sequences extracted from video. The project explores multiple neural network architectures for sequence classification and aims to build a deployable live translation system using MediaPipe landmarks and optimized neural network inference.

The project was developed using PyTorch, Transformer architectures, temporal convolutional networks (TCNs), recurrent models, and hybrid CNN-attention systems.

---

# Project Goals

The primary objectives of this project were:

- Build a scalable ASL sequence classification pipeline
- Process landmark-based sign language data into trainable tensors
- Experiment with multiple sequence modeling architectures
- Compare CNNs, Transformers, GRUs, and TCNs
- Train models on both partial and full datasets
- Export deployable models for real-time inference
- Build the foundation for a live translation application using MediaPipe

---

# Dataset

The project uses the ASL Signs dataset containing:

- 250 sign classes
- Landmark sequences extracted from sign videos
- Hand, face, and pose keypoints
- Variable-length temporal sequences

The dataset was preprocessed into fixed-length tensors suitable for sequence modeling.

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
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   ├── onnx/
│   └── torchscript/
│
├── results/
│   ├── graphs/
│   ├── metrics/
│   ├── reports/
│   └── tables/
│
└── README.md
```

---

# Pipeline Overview

## 1. Preprocessing

Notebook: `00_Preprocessing.ipynb`

The preprocessing pipeline:

- Loads raw landmark sequence data
- Cleans invalid samples
- Handles variable sequence lengths
- Pads sequences to fixed length
- Generates train/validation/test splits
- Saves processed `.npz` and `.npy` arrays
- Stores metadata and label mappings

### Key Features

- Participant-aware splitting
- Removal of labels missing in validation/test sets
- Sequence normalization
- Length tracking for masking and pooling

---

## 2. Exploratory Data Analysis (EDA)

Notebook: `01_EDA.ipynb`

EDA focused on understanding:

- Class balance
- Sequence length distributions
- Missing values
- Landmark feature distributions
- Sign frequency
- Dataset split quality

### Outputs

- Histograms
- Distribution plots
- Sequence statistics
- Class frequency graphs
- Summary tables

All outputs are saved into:

```text
results/graphs/
results/tables/
```

---

# Modeling Approaches

The project explored several deep learning architectures for sequence classification.

---

## CNN Models

Convolutional models were used to capture local temporal motion patterns.

### Features

- Depthwise separable convolutions
- Residual blocks
- Large kernel temporal convolutions
- Dilated convolutions
- Temporal pooling

### Motivation

CNNs performed extremely well because sign language contains strong local temporal structure and repetitive movement patterns.

---

## GRU / Recurrent Models

GRU-based models were used to capture long-term temporal dependencies.

### Features

- Bidirectional GRUs
- Sequence pooling
- Hybrid CNN + GRU models

---

## Transformer Models

Transformers were introduced to model global temporal relationships.

### Features

- Multi-head self-attention
- Positional embeddings
- Layer normalization
- Feedforward attention blocks
- Hybrid CNN + Transformer architectures

---

## Temporal Convolutional Networks (TCNs)

TCNs provided an alternative sequence modeling approach using:

- Dilated convolutions
- Residual temporal blocks
- Large receptive fields
- Efficient parallel training

---

# Partial Dataset Experiments

Notebook: `02_Partial_Model_Training.ipynb`

Before training on all 250 signs, models were tested on smaller subsets.

### Purpose

- Faster experimentation
- Architecture comparison
- Hyperparameter tuning
- Stability checks

### Models Tested

- MLP
- CNN
- LSTM
- GRU
- CNN + GRU
- Transformer
- CNN + Transformer
- TCN

### Evaluation Metrics

- Top-1 Accuracy
- Top-5 Accuracy
- Top-10 Accuracy
- Classification reports
- Per-class F1 scores

---

# Full Dataset Training

Notebook: `03_Full_Model_Training.ipynb`

The strongest architectures from partial experiments were scaled to the full 250-sign dataset.

### Final Models

- Strong CNN
- Strong CNN + GRU
- Strong Transformer
- Strong TCN

### Improvements

- Larger hidden dimensions
- Stronger augmentation
- Label smoothing
- Cosine learning rate scheduling
- Larger receptive fields
- Better regularization

---

# Advanced Transformer Training

Notebook: `04_Transformer_Full_Training.ipynb`

This notebook focused on advanced attention-based architectures.

### Focus Areas

- CNN + Transformer hybrids
- Larger Transformer encoders
- Improved positional embeddings
- Hybrid local/global temporal modeling

---

# Data Augmentation

The project used several augmentation techniques:

## Time Stretching

Randomly compressing or expanding temporal sequences.

## Gaussian Noise

Small random perturbations added to landmark coordinates.

## Frame Dropout

Random frame masking to improve robustness.

---

# Evaluation

Models were evaluated using:

- Validation Top-1 Accuracy
- Test Top-1 Accuracy
- Test Top-5 Accuracy
- Test Top-10 Accuracy
- Precision / Recall / F1
- Confusion matrices

---

# Saved Outputs

## Graphs

Saved to:

```text
results/graphs/
```

Examples:

- Loss curves
- Accuracy curves
- Confusion matrices
- Confidence distributions
- Per-class F1 charts

---

## Metrics and Reports

Saved to:

```text
results/reports/
results/metrics/
results/tables/
```

Examples:

- Classification reports
- Per-class metrics
- Ablation tables
- Training summaries

---

# Model Exporting

The project supports multiple export formats for deployment.

## TorchScript

Used for PyTorch deployment and mobile compatibility.

## ONNX

Used for cross-platform inference and optimized runtimes.

## Quantization

Dynamic quantization was explored to:

- Reduce model size
- Improve CPU inference speed
- Enable mobile deployment

---

# Live Translation System

The long-term objective is a real-time sign language translation application.

## Planned Pipeline

```text
Webcam
   ↓
OpenCV Video Frames
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

# MediaPipe Integration

MediaPipe is used to extract:

- Hand landmarks
- Pose landmarks
- Face landmarks

These landmarks are converted into tensors matching the training format used by the models.

---

# Technologies Used

## Core Libraries

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Computer Vision

- OpenCV
- MediaPipe

## Deployment

- ONNX Runtime
- TorchScript

---

# Current Status

The repository currently includes:

- Full preprocessing pipeline
- EDA workflow
- Multiple model architectures
- Full training pipelines
- Export utilities
- Evaluation framework
- Deployment preparation

Additional official results and finalized metrics will be added after retraining and standardized evaluation.

---

# Future Improvements

Planned future work includes:

- Official participant-level cross-validation
- Real-time inference application
- Sentence-level translation
- Language modeling
- Beam search decoding
- Mobile deployment
- Transformer scaling
- Landmark feature engineering
- Faster inference optimization

---

# Notes

Some experimental results and metrics are still preliminary and will be updated after standardized retraining and evaluation. The repository structure and pipelines are finalized, but official benchmark numbers are still being regenerated.
