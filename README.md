# SENet Ultrasound Classification

A comprehensive deep learning implementation using Squeeze-and-Excitation Networks (SENet) for automated classification of ultrasound images. This project demonstrates the application of attention mechanisms in medical image analysis to improve diagnostic accuracy and efficiency.

## Table of Contents

- [Overview](#overview)
- [What is SENet?](#what-is-senet)
- [Implementation Details](#implementation-details)
- [Dataset Requirements](#dataset-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [How This Helps](#how-this-helps)
- [Contributing](#contributing)

## Overview

This project implements a state-of-the-art deep learning solution for classifying ultrasound images using Squeeze-and-Excitation Networks (SENet). The implementation leverages transfer learning from a pre-trained SE-ResNeXt-50 model to achieve high accuracy in medical image classification tasks.

**Key Features:**
- 🎯 **High Accuracy**: Achieves 91.45% accuracy on test data
- 🔄 **Transfer Learning**: Uses pre-trained SE-ResNeXt-50 from ImageNet
- 🎨 **Comprehensive Pipeline**: End-to-end solution from data preprocessing to evaluation
- 📊 **Detailed Metrics**: Includes accuracy, precision, recall, and F1-score analysis
- 🛡️ **Robust Training**: Implements early stopping and learning rate scheduling
- 📈 **Visualization**: Training progress tracking and prediction visualization

## What is SENet?

Squeeze-and-Excitation Networks (SENet) are a revolutionary architecture that introduces attention mechanisms to convolutional neural networks. SENet enhances the representational power of networks by explicitly modeling the interdependencies between channels.

### Key Components:

1. **Squeeze Operation**: Global average pooling to compress spatial dimensions
2. **Excitation Operation**: Simple gating mechanism with sigmoid activation
3. **Channel Attention**: Learns to emphasize important features and suppress less useful ones

### Why SENet for Medical Imaging?

- **Feature Enhancement**: Automatically focuses on diagnostically relevant features
- **Improved Accuracy**: Attention mechanisms lead to better classification performance
- **Interpretability**: Channel attention weights provide insights into model decisions
- **Efficiency**: Minimal computational overhead while maximizing performance gains

## Implementation Details

### Technical Architecture

```
Input Ultrasound Images → Data Preprocessing → SE-ResNeXt-50 → Classification Head → Prediction
                             ↓
                    Data Augmentation & Normalization
                             ↓
                    Channel Attention Mechanism (SENet)
                             ↓
                    Multi-class Classification Output
```

### Core Components

1. **Custom Dataset Class**: `UltrasoundDataset` for efficient data loading
2. **Pre-trained Backbone**: SE-ResNeXt-50-32x4d with ImageNet weights
3. **Transfer Learning**: Modified final layer for ultrasound-specific classification
4. **Training Pipeline**: Custom training loop with validation and early stopping
5. **Evaluation Framework**: Comprehensive metrics calculation and visualization

### Training Strategy

- **Optimizer**: SGD with momentum (0.9) and learning rate (0.001)
- **Scheduler**: StepLR with step size 7 and gamma 0.1
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Early Stopping**: Prevents overfitting with patience of 5 epochs
- **Data Split**: 64% training, 16% validation, 20% testing

## Dataset Requirements

### Expected Data Structure

Your dataset should be organized as a CSV file with the following structure:

```csv
Image_path,Plane
/path/to/image1.jpg,0
/path/to/image2.jpg,1
/path/to/image3.jpg,2
...
```

### Data Specifications

- **Image Format**: RGB images (automatically converted from grayscale if needed)
- **Image Size**: Automatically resized to 224x224 pixels
- **Labels**: Integer-encoded class labels (automatically handled by LabelEncoder)
- **Supported Formats**: JPEG, PNG, and other PIL-supported formats

### Data Preprocessing Pipeline

1. **Label Encoding**: Automatic conversion of categorical labels to integers
2. **Image Resizing**: Standardized to 224x224 pixels
3. **Normalization**: ImageNet mean and standard deviation
4. **Center Cropping**: Ensures consistent input dimensions
5. **Tensor Conversion**: PyTorch-compatible format

## Installation

### Prerequisites

Ensure you have the following installed:

- **Python**: 3.9.0 or higher
- **CUDA**: Compatible version for GPU acceleration (optional but recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tanjim-Islam/SENet-Ultrasound-Classification.git
cd SENet-Ultrasound-Classification
```

### Step 2: Install Dependencies

#### Core Dependencies
```bash
pip install pandas numpy matplotlib Pillow tqdm scikit-learn pretrainedmodels
```

#### PyTorch Installation
```bash
# For CUDA support (recommended)
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Verify Installation

```python
import torch
import torchvision
import pretrainedmodels

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

### Quick Start

1. **Prepare Your Data**: Ensure your CSV file follows the required format
2. **Update Data Path**: Modify the CSV path in the notebook
3. **Run Training**: Execute the Jupyter notebook cells sequentially
4. **Monitor Progress**: Watch training/validation metrics in real-time
5. **Evaluate Results**: Review final performance metrics and visualizations

### Detailed Workflow

#### 1. Data Loading and Preprocessing
```python
# Load dataset
data = pd.read_csv('path_to_csv_file.csv')

# Encode labels
le = LabelEncoder()
data['Plane'] = le.fit_transform(data['Plane'])

# Create data splits
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
```

#### 2. Model Configuration
```python
# Load pre-trained SENet model
model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')

# Modify for your classification task
num_classes = len(data['Plane'].unique())
model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
```

#### 3. Training
```python
# Training loop with early stopping
for epoch in range(n_epochs):
    # Training phase
    model.train()
    # ... training code ...
    
    # Validation phase
    model.eval()
    # ... validation code ...
    
    # Early stopping check
    if val_loss < min_val_loss:
        torch.save(model.state_dict(), 'best_model.pt')
```

#### 4. Evaluation
```python
# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pt'))
accuracy, precision, recall, f1 = calculate_metrics(test_loader, model, device)
```

## Model Architecture

### SE-ResNeXt-50-32x4d Overview

The implementation uses SE-ResNeXt-50-32x4d as the backbone architecture, which combines:

- **ResNeXt Architecture**: Group convolutions for improved efficiency
- **Squeeze-and-Excitation Blocks**: Channel attention mechanisms
- **Deep Residual Learning**: Skip connections for gradient flow

### Architecture Details

```
Input (3, 224, 224)
       ↓
Conv2d + BatchNorm + ReLU
       ↓
MaxPool2d
       ↓
SE-ResNeXt Blocks (Layer 1) - 64 channels
       ↓
SE-ResNeXt Blocks (Layer 2) - 128 channels
       ↓
SE-ResNeXt Blocks (Layer 3) - 256 channels
       ↓
SE-ResNeXt Blocks (Layer 4) - 512 channels
       ↓
Global Average Pooling
       ↓
Fully Connected Layer (num_classes)
       ↓
Output Predictions
```

### SE Module Details

Each SE block contains:
1. **Global Average Pooling**: Squeeze spatial information
2. **FC → ReLU → FC → Sigmoid**: Channel excitation
3. **Channel-wise Multiplication**: Apply attention weights

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.45% |
| **Precision** | 91.70% |
| **Recall** | 91.45% |
| **F1-Score** | 91.36% |

### Training Characteristics

- **Convergence**: Stable training with early stopping
- **Generalization**: Minimal overfitting observed
- **Efficiency**: Fast convergence due to transfer learning
- **Robustness**: Consistent performance across validation runs

### Key Insights

1. **Transfer Learning Effectiveness**: Pre-trained ImageNet features transfer well to ultrasound images
2. **Attention Benefits**: SE modules significantly improve feature discrimination
3. **Data Efficiency**: High performance achieved with limited medical imaging data
4. **Computational Efficiency**: Reasonable training time with modern GPUs

## How This Helps

### Medical Applications

#### 1. **Diagnostic Assistance**
- **Automated Screening**: Rapid initial assessment of ultrasound images
- **Quality Control**: Identify improperly acquired images
- **Standardization**: Consistent classification across different operators

#### 2. **Clinical Workflow Enhancement**
- **Time Savings**: Reduce manual review time for routine cases
- **Accuracy Improvement**: Minimize human error in classification
- **Training Aid**: Educational tool for medical students and residents

#### 3. **Research Applications**
- **Large-scale Studies**: Automated analysis of extensive image databases
- **Reproducibility**: Consistent classification criteria across studies
- **Data Mining**: Discovery of imaging patterns and correlations

### Technical Benefits

#### 1. **Attention Mechanisms**
- **Interpretability**: Understand which image regions drive decisions
- **Feature Enhancement**: Automatically focus on relevant anatomical structures
- **Noise Reduction**: Suppress irrelevant background information

#### 2. **Transfer Learning Advantages**
- **Reduced Training Time**: Leverage pre-trained features
- **Data Efficiency**: Achieve high performance with limited medical data
- **Generalization**: Better performance on unseen data

#### 3. **Scalability**
- **Batch Processing**: Handle large volumes of images efficiently
- **Cloud Deployment**: Easy integration with telemedicine platforms
- **Real-time Processing**: Fast inference for clinical applications

### Impact Areas

1. **Healthcare Accessibility**: Enable expert-level analysis in resource-limited settings
2. **Quality Assurance**: Standardize image interpretation across institutions
3. **Research Acceleration**: Facilitate large-scale medical imaging studies
4. **Education**: Provide learning tools for medical professionals
5. **Cost Reduction**: Decrease manual review requirements

## Contributing

We welcome contributions to improve this project! Here's how you can help:

### Areas for Contribution

1. **Model Improvements**: Experiment with different architectures
2. **Data Augmentation**: Implement advanced augmentation techniques
3. **Evaluation Metrics**: Add specialized medical imaging metrics
4. **Documentation**: Improve code documentation and examples
5. **Testing**: Add unit tests and integration tests

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write comprehensive comments for complex logic

---

**Note**: This implementation is designed for research and educational purposes. For clinical applications, ensure proper validation and regulatory compliance.
