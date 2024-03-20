# SENet Ultrasound Classification

Implementation of Squeeze-and-Excitation Networks (SENet) for classifying ultrasound images.

## Prerequisites

Ensure you have the following installed:

- Python 3.9.0
- PyTorch 2.1.0+cu118
- Torchvision 0.16.0+cu118
- Additional Python libraries: pandas, numpy, matplotlib, Pillow, tqdm, scikit-learn, pretrainedmodels

## Installation

First, clone the repository:

```
git clone https://github.com/Tanjim-Islam/SENet-Ultrasound-Classification.git
cd SENet-Ultrasound-Classification
```

Then, install the required dependencies:

```
pip install pandas numpy matplotlib Pillow tqdm scikit-learn pretrainedmodels
```

```
pip install torch==2.1.1 torchvision==0.16.1
```

## Structure

- **Data Preprocessing**: Label encoding, train-test split, data loaders and transformations setup.
- **Model Setup**: Loading the pre-trained SENet model, modifying it for the dataset, setting up the loss function, optimizer, and scheduler.
- **Training and Validation**: Custom loop for training with early stopping, validation accuracy and loss monitoring.
- **Evaluation**: Accuracy, precision, recall, and F1-score calculations on the test set.
- **Prediction**: Visualizing model predictions on test images.

