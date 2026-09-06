# Ship Classification — CNN

A deep learning project for **multi-class ship image classification**, built with **PyTorch**.

The project focuses on building a complete computer vision pipeline, from dataset preprocessing and augmentation to CNN training and model evaluation.

## Project Overview

The objective is to classify ship images into **10 different classes** using a custom Convolutional Neural Network.

The project covers:

* Image preprocessing and normalization
* Data augmentation
* Train / validation / test splitting
* Class imbalance handling
* CNN architecture design
* GPU-accelerated training
* Overfitting tests
* Confusion matrix and classification report

## Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

## Pipeline

```text
Raw Images
    │
    ▼
Grayscale Conversion
    │
    ▼
Train / Validation / Test Split
    │
    ├── Training
    │     ├── Random Horizontal Flip
    │     ├── Random Rotation
    │     └── Normalization
    │
    └── Validation / Test
          └── Normalization
    │
    ▼
CNN
    │
    ├── Conv Block 1: 32 channels
    ├── Conv Block 2: 64 channels
    ├── Conv Block 3: 128 channels
    └── Conv Block 4: 256 channels
    │
    ▼
Global Average Pooling
    │
    ▼
Fully Connected Classifier
    │
    ▼
10 Ship Classes
```

## CNN Architecture

The model is a custom CNN composed of four convolutional blocks.

Each block contains:

* `Conv2D`
* `BatchNorm`
* `ReLU`
* `Conv2D`
* `BatchNorm`
* `ReLU`
* `MaxPooling`

The final feature maps are reduced using **Adaptive Average Pooling**, followed by a fully connected classifier.

Dropout is used in the final classifier to reduce overfitting.

## Training

The model is trained using:

* **Optimizer:** Adam
* **Loss:** Weighted Cross-Entropy
* **Learning rate:** `1e-3`
* **Scheduler:** Cosine Annealing
* **Epochs:** 50
* **Batch size:** 64
* **GPU acceleration:** CUDA when available

Because the dataset is imbalanced, class weights are applied to the loss function so that minority classes have a stronger influence during training.

## Model Validation

Before the full training phase, a small subset of the dataset is used to verify that the model can successfully overfit a limited number of samples.

This acts as a basic **training sanity check** and helps detect issues in:

* Model architecture
* Labels
* Data preprocessing
* Loss computation
* Gradient propagation

## Evaluation

The trained model is evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

Example:

```python
classification_report(
    all_labels,
    all_preds,
    target_names=classes
)
```

The confusion matrix provides a visual representation of which ship classes are most frequently confused by the model.

## Key Engineering Decisions

### Normalization

Dataset-wide mean and standard deviation are computed before training rather than using arbitrary normalization values.

### Data Augmentation

Training images are randomly:

* Horizontally flipped
* Rotated by up to ±10°

This increases visual variability and helps improve generalization.

### Class Imbalance

The dataset contains significantly different numbers of samples per class.

Weighted Cross-Entropy is therefore used instead of treating every class equally.

### Global Average Pooling

`AdaptiveAvgPool2d(1, 1)` reduces the spatial dimensions before classification, limiting the number of parameters in the classifier and reducing the risk of overfitting.

## Project Structure

```text
.
├── data/
│   └── ships_gray/
├── notebooks/
├── src/
├── README.md
└── requirements.txt
```

## Results

Training and evaluation results are reported through the generated classification report and confusion matrix.

> Add your final test accuracy and F1-score here once the final experiment is complete.

```text
Test Accuracy: XX.XX%
Macro F1-Score: XX.XX%
```

## Future Improvements

Potential improvements include:

* Transfer learning with pretrained architectures
* Hyperparameter optimization
* More advanced data augmentation
* Early stopping
* Experiment tracking
* Model checkpointing
* Precision / Recall analysis per class
* Comparison with ResNet / EfficientNet
* Deployment as an inference API

## Skills Demonstrated

This project demonstrates practical experience with:

**Computer Vision • Deep Learning • PyTorch • CNNs • Data Preprocessing • Data Augmentation • Model Evaluation • Class Imbalance • GPU Training • Machine Learning Engineering**

## Author

**Mustapha Oumeziane**

AI / ML Engineering Student — EPITA

Interested in **Software Engineering, AI/ML and Computer Vision**.
