# Self-Supervised Learning Models for Solar Flare Analysis

## 📚 Table of Contents
- [Introduction](#introduction)
- [What is Self-Supervised Learning?](#what-is-self-supervised-learning)
- [Contrastive Learning Fundamentals](#contrastive-learning-fundamentals)
- [Why Use Self-Supervised Learning for Solar Flares?](#why-use-self-supervised-learning-for-solar-flares)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Key Concepts Explained](#key-concepts-explained)
- [Practical Usage](#practical-usage)
- [Benefits and Limitations](#benefits-and-limitations)
- [References and Further Reading](#references-and-further-reading)

## 🌟 Introduction

This module implements **self-supervised contrastive learning models** specifically designed for solar flare analysis using GOES X-ray data. Self-supervised learning is a powerful machine learning paradigm that learns meaningful representations from unlabeled data, making it particularly valuable for solar physics where labeled data is often scarce or expensive to obtain.

## 🔍 What is Self-Supervised Learning?

### Basic Definition
Self-supervised learning is a type of machine learning where the model learns to understand data by creating its own supervision signal from the data itself, without requiring human-labeled examples.

### Key Principles
1. **No Manual Labels Required**: The model creates its own learning tasks from the raw data
2. **Representation Learning**: Focuses on learning meaningful features that capture the underlying structure of the data
3. **Transfer Learning**: Pre-trained representations can be fine-tuned for specific downstream tasks

### How It Differs from Other Learning Types

| Learning Type | Supervision Source | Data Requirements | Example |
|---------------|-------------------|-------------------|---------|
| **Supervised** | Human labels | Large labeled datasets | "This X-ray pattern shows a C-class flare" |
| **Unsupervised** | None | Unlabeled data only | Clustering similar X-ray patterns |
| **Self-Supervised** | Data-derived tasks | Unlabeled data | "Predict if two X-ray signals are from the same time period" |

## 🎯 Contrastive Learning Fundamentals

### Core Concept
Contrastive learning teaches a model to distinguish between **similar** (positive) and **dissimilar** (negative) examples by learning representations where:
- Similar samples have representations that are close together in the learned feature space
- Dissimilar samples have representations that are far apart

### The Contrastive Learning Process

```
1. Take an X-ray time series sample
2. Create two different "views" through data augmentation
   └── View 1: Original + noise
   └── View 2: Original + time masking
3. Pass both views through an encoder network
4. The model learns to make the representations of these two views similar
5. While making them different from representations of other samples
```

### Mathematical Foundation

The model optimizes the **NT-Xent (Normalized Temperature-scaled Cross Entropy)** loss:

```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

Where:
- `z_i, z_j` are representations of positive pairs (same sample, different augmentations)
- `z_k` are representations of negative pairs (different samples)
- `sim()` is cosine similarity
- `τ` is the temperature parameter

## 🌞 Why Use Self-Supervised Learning for Solar Flares?

### Challenges in Solar Flare Analysis
1. **Limited Labeled Data**: Solar flare classifications require expert knowledge
2. **Class Imbalance**: Major flares (X-class) are rare compared to minor ones
3. **Temporal Complexity**: Flare patterns evolve over different timescales
4. **Feature Engineering**: Traditional methods require manual feature selection

### Advantages of Self-Supervised Approach
1. **Leverages Abundant Unlabeled Data**: GOES provides continuous X-ray measurements
2. **Learns Temporal Patterns**: Captures complex time-series relationships automatically
3. **Domain-Agnostic Features**: Learns representations without human bias
4. **Transfer Learning**: Pre-trained models can be adapted for various tasks

### Real-World Application
```python
# Example workflow
1. Pre-train on 10 years of GOES X-ray data (unlabeled)
2. Learn general solar activity representations
3. Fine-tune on small labeled dataset for flare classification
4. Achieve better performance than training from scratch
```

## 🏗️ Model Architecture

### Overall Structure
```
Input X-ray Time Series
         ↓
    Data Augmentation (creates positive pairs)
         ↓
    Encoder Network (CNN-based)
         ↓
    Projection Head (Dense layers)
         ↓
    Contrastive Loss Computation
```

### Component Details

#### 1. **Encoder Network**
- **Purpose**: Extract meaningful features from X-ray time series
- **Architecture**: 1D Convolutional Neural Network
- **Layers**:
  ```
  Conv1D(64) → BatchNorm → MaxPool
  Conv1D(128) → BatchNorm → MaxPool
  Conv1D(256) → BatchNorm → GlobalAvgPool
  Dense(512) → Dropout → Dense(256)
  ```

#### 2. **Data Augmentation**
- **Purpose**: Create different "views" of the same data
- **Techniques**:
  - **Noise Addition**: `x_aug = x + noise * 0.1`
  - **Time Masking**: Randomly set some time steps to zero
  - **Amplitude Scaling**: Multiply by random factor (0.8-1.2)
  - **Time Shifting**: Circular shift of the time series

#### 3. **Projection Head**
- **Purpose**: Map encoder outputs to contrastive learning space
- **Architecture**: `Dense(256) → Dropout → Dense(projection_dim)`
- **Note**: Discarded after pre-training, only encoder is kept

#### 4. **Classifier Head** (Fine-tuning phase)
- **Purpose**: Perform downstream task (flare classification)
- **Architecture**: `Dense(128) → Dropout → Dense(n_classes)`

## 🎓 Training Process

### Phase 1: Contrastive Pre-training
```python
# Pseudo-code for pre-training
for batch in unlabeled_data:
    # Create augmented pairs
    view1 = augment(batch)
    view2 = augment(batch)
    
    # Get representations
    repr1 = encoder(view1)
    repr2 = encoder(view2)
    
    # Project to contrastive space
    proj1 = projection_head(repr1)
    proj2 = projection_head(repr2)
    
    # Compute contrastive loss
    loss = contrastive_loss(proj1, proj2)
    
    # Update model
    optimizer.step(loss)
```

### Phase 2: Supervised Fine-tuning
```python
# Pseudo-code for fine-tuning
# Freeze encoder weights (optional)
encoder.freeze()

for batch, labels in labeled_data:
    # Get pre-trained representations
    representations = encoder(batch)
    
    # Classify
    predictions = classifier(representations)
    
    # Standard supervised loss
    loss = cross_entropy(predictions, labels)
    
    # Update classifier only
    optimizer.step(loss)
```

## 🔑 Key Concepts Explained

### 1. **Temperature Parameter (τ)**
- **Purpose**: Controls the concentration of the learned representations
- **Effect**: 
  - Low temperature (0.1): Sharp distinctions, hard negatives
  - High temperature (1.0): Softer distinctions, easier learning
- **Analogy**: Like adjusting the "strictness" of the similarity judgment

### 2. **Positive vs Negative Pairs**
- **Positive Pairs**: Different augmentations of the same X-ray sample
  ```
  Original signal: [1.2, 1.5, 2.1, ...]
  Augmented:      [1.3, 1.4, 2.0, ...]  ← Should be similar
  ```
- **Negative Pairs**: Augmentations from different samples
  ```
  Sample A: [1.2, 1.5, 2.1, ...]
  Sample B: [0.8, 0.9, 1.1, ...]  ← Should be dissimilar
  ```

### 3. **Representation Space**
- **Goal**: Learn a space where similar solar activity patterns cluster together
- **Visualization**: Imagine a 256-dimensional space where:
  - Quiet sun periods cluster in one region
  - C-class flares cluster in another region
  - X-class flares form their own cluster

### 4. **Data Augmentation Philosophy**
- **Principle**: Create variations that preserve the essential physics
- **Solar Flare Context**: 
  - Small noise → Instrument calibration variations
  - Time masking → Data gaps or missing measurements
  - Amplitude scaling → Different solar cycle phases

## 💻 Practical Usage

### Basic Example
```python
from self_supervised_models import ContrastiveLearningModel

# Initialize model
model = ContrastiveLearningModel(
    sequence_length=128,    # 128 time steps
    n_features=2,          # XRSA and XRSB channels
    projection_dim=128,    # Contrastive space dimension
    temperature=0.1        # Temperature parameter
)

# Pre-training phase
model.build_contrastive_model()
pretrain_history = model.pretrain(
    X_unlabeled, 
    epochs=100, 
    batch_size=32
)

# Fine-tuning phase
model.build_classifier(n_classes=6)  # A, B, C, M, X, background
finetune_history = model.fine_tune(
    X_train, y_train, 
    X_val, y_val,
    epochs=50
)

# Analysis and visualization
model.plot_contrastive_analysis(X_sample)
model.plot_training_comparison(pretrain_history, finetune_history)
```

### Step-by-Step Workflow

1. **Data Preparation**
   ```python
   # Load GOES X-ray data
   X_unlabeled = load_goes_data()  # Shape: (samples, time_steps, features)
   
   # Optional: Load labeled subset for fine-tuning
   X_labeled, y_labeled = load_labeled_flares()
   ```

2. **Pre-training**
   ```python
   # Learn general solar activity representations
   model.pretrain(X_unlabeled, epochs=100)
   ```

3. **Fine-tuning**
   ```python
   # Adapt for specific classification task
   model.fine_tune(X_labeled, y_labeled, epochs=50)
   ```

4. **Inference**
   ```python
   # Use trained model for predictions
   predictions = model.classifier.predict(new_data)
   ```

## ⚖️ Benefits and Limitations

### ✅ Benefits

1. **Data Efficiency**
   - Requires less labeled data compared to fully supervised approaches
   - Can leverage years of unlabeled GOES measurements

2. **Robust Representations**
   - Learns features that generalize across different solar conditions
   - Less sensitive to specific instrument characteristics

3. **Transfer Learning**
   - Pre-trained encoder can be adapted for multiple tasks:
     - Flare classification
     - Intensity prediction
     - Anomaly detection

4. **Automatic Feature Discovery**
   - No need for manual feature engineering
   - Discovers temporal patterns automatically

### ❌ Limitations

1. **Computational Requirements**
   - Pre-training phase requires significant computation
   - Needs careful hyperparameter tuning

2. **Architecture Sensitivity**
   - Performance depends on augmentation strategies
   - Temperature parameter requires tuning

3. **Interpretability**
   - Learned representations are not easily interpretable
   - Black-box nature of deep learning

4. **Domain Knowledge**
   - Augmentation strategies should respect physical constraints
   - May not capture all relevant physics without careful design

## 📖 References and Further Reading

### Foundational Papers
1. **SimCLR**: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
2. **MoCo**: He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
3. **SwAV**: Caron et al. "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments" (2020)

### Solar Physics Applications
1. **Solar Flare Prediction**: Time series analysis and machine learning approaches
2. **GOES Data Analysis**: X-ray solar flare detection and classification
3. **Space Weather**: Machine learning for space weather prediction

### Technical Resources
1. **Contrastive Learning Survey**: "Self-supervised Learning: Generative or Contrastive" (2021)
2. **Time Series SSL**: "Self-supervised learning for time series analysis: Taxonomy, progress, and prospects" (2023)
3. **TensorFlow SSL Guide**: Official TensorFlow self-supervised learning tutorials

### Implementation Details
- **Framework**: TensorFlow/Keras
- **Hardware**: GPU recommended for pre-training
- **Memory**: Batch size depends on available GPU memory
- **Storage**: Consider data pipeline optimization for large datasets

---

## 🚀 Getting Started

To start using the self-supervised models:

1. **Install Dependencies**
   ```bash
   pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
   ```

2. **Import and Initialize**
   ```python
   from self_supervised_models import ContrastiveLearningModel
   model = ContrastiveLearningModel()
   ```

3. **Load Your Data**
   ```python
   # Ensure data shape: (n_samples, sequence_length, n_features)
   X = your_goes_data.reshape(-1, 128, 2)
   ```

4. **Start Pre-training**
   ```python
   model.pretrain(X, epochs=50)
   ```

For detailed examples and advanced usage, see the test scripts in the `model_test/` directory.

---

*This README provides a comprehensive introduction to self-supervised contrastive learning for solar flare analysis. For questions or contributions, please refer to the main project documentation.*
