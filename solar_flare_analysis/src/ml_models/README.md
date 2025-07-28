# 🌟 Transformer-Based Solar Flare Analysis Models

## Overview

This module implements advanced deep learning models specifically designed for solar flare detection, classification, and analysis using state-of-the-art Transformer architectures. The models leverage the power of self-attention mechanisms to capture complex temporal dependencies in solar X-ray flux data from GOES satellites.

## 🔬 Scientific Background

### Solar Flares: Physical Phenomena

Solar flares are sudden releases of electromagnetic energy from the Sun's atmosphere, primarily in the corona. These events are among the most energetic phenomena in our solar system and can have significant impacts on Earth's technological infrastructure.

#### Key Physical Characteristics:

1. **Energy Release**: Solar flares can release up to 10³² ergs of energy in minutes to hours
2. **Electromagnetic Spectrum**: Emit across the entire electromagnetic spectrum, from radio waves to gamma rays
3. **Magnetic Reconnection**: Caused by the sudden reconfiguration of magnetic field lines in the solar corona
4. **Temporal Evolution**: Exhibit characteristic rise and decay phases with complex temporal patterns

### X-Ray Emission Physics

Solar flares are primarily detected through their X-ray emissions, measured by GOES satellites in two energy bands:

- **GOES-XRSA (0.05-0.4 nm)**: Higher-energy soft X-rays (3.1-24.8 keV)
- **GOES-XRSB (0.1-0.8 nm)**: Lower-energy soft X-rays (1.55-12.4 keV)

The X-ray flux follows the relationship:
```
F(λ) = ∫ N(T) × σ(λ,T) × dT
```
Where:
- `F(λ)` is the observed flux at wavelength λ
- `N(T)` is the differential emission measure
- `σ(λ,T)` is the emission cross-section

### Flare Classification

Solar flares are classified based on their peak X-ray flux in the GOES-B channel:

| Class | Peak Flux Range (W/m²) | Physical Significance |
|-------|------------------------|----------------------|
| A | < 10⁻⁷ | Background level |
| B | 10⁻⁷ to 10⁻⁶ | Minor events |
| C | 10⁻⁶ to 10⁻⁵ | Small flares |
| M | 10⁻⁵ to 10⁻⁴ | Medium flares |
| X | > 10⁻⁴ | Major flares |

## 🧠 Deep Learning Architecture

### Why Transformers for Solar Flare Analysis?

Traditional RNNs and CNNs have limitations in capturing long-range temporal dependencies in time series data. Solar flares exhibit complex multi-scale temporal patterns:

1. **Pre-flare Phase**: Gradual increase over minutes to hours
2. **Impulsive Phase**: Rapid rise to peak (minutes)
3. **Decay Phase**: Exponential or power-law decay (minutes to hours)
4. **Post-flare Loops**: Secondary structures and oscillations

Transformers excel at modeling these patterns because:

#### Self-Attention Mechanism
The attention mechanism computes relationships between all pairs of time steps:

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

Where:
- `Q` (Query): Current time step seeking information
- `K` (Key): All time steps being evaluated
- `V` (Value): Information content at each time step
- `d_k`: Dimension scaling factor

#### Multi-Head Attention
Multiple attention heads capture different aspects of temporal relationships:

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

This allows the model to simultaneously focus on:
- **Local patterns**: Sharp rises and falls
- **Global trends**: Background variations
- **Periodicities**: Solar rotation and activity cycles
- **Cross-channel correlations**: GOES-A vs GOES-B relationships

### Model Architectures

#### 1. TransformerFlareModel

A pure transformer architecture optimized for multi-task learning:

**Input Processing:**
```
X ∈ ℝ^(batch_size × sequence_length × n_features)
```

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

**Multi-Task Outputs:**
- **Flare Classification**: 6-class categorical (A, B, C, M, X, Non-flare)
- **Intensity Prediction**: Continuous peak flux value
- **Duration Estimation**: Flare duration in minutes

#### 2. ConvolutionalTransformerModel

A hybrid CNN-Transformer architecture that combines:

**Convolutional Feature Extraction:**
```
h_conv = Conv1D(X) → BatchNorm → ReLU → MaxPool
```

**Transformer Processing:**
```
h_trans = MultiHeadAttention(h_conv) + FFN(h_conv)
```

This hybrid approach captures:
- **Local features** via convolution (sudden changes, noise filtering)
- **Global dependencies** via attention (long-term correlations)

## 🔧 Technical Implementation

### Key Features

#### 1. Dynamic Sequence Length Support
```python
seq_len = tf.shape(x)[1]  # Dynamic sequence length
positions = tf.range(start=0, limit=seq_len, delta=1)
```

#### 2. Multi-Scale Loss Function
```python
L_total = w1×L_classification + w2×L_intensity + w3×L_duration
```

Where:
- `L_classification`: Sparse categorical crossentropy
- `L_intensity`: Mean squared error for flux prediction
- `L_duration`: Mean squared error for temporal prediction

#### 3. Advanced Training Strategies
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive rate reduction
- **Model Checkpointing**: Saves best performing weights

### Data Preprocessing

#### Robust Scaling
```python
X_scaled = (X - median(X)) / IQR(X)
```

This approach is robust to outliers common in solar data due to:
- Instrument calibration drifts
- Solar particle events
- Data gaps and interpolation artifacts

#### Temporal Windowing
Fixed-length sequences (default: 128 time steps) ensure:
- Consistent model input dimensions
- Capture of complete flare evolution
- Efficient batch processing

## 📊 Model Interpretability

### Attention Visualization

The models provide several interpretability tools:

#### 1. Attention Weight Analysis
```python
attention_weights = model.extract_attention_weights(X_sample)
```

Reveals which time steps the model focuses on during prediction.

#### 2. Attention Entropy
```python
entropy = -Σ(attention × log(attention + ε))
```

Measures attention concentration:
- **Low entropy**: Focused attention on specific time steps
- **High entropy**: Distributed attention across sequence

#### 3. Multi-Head Comparison
Different attention heads learn to focus on:
- **Head 1**: Peak detection
- **Head 2**: Background trends  
- **Head 3**: Rise phase patterns
- **Head 4**: Decay characteristics

## 🚀 Usage Examples

### Basic Model Training
```python
from transformer_flare_model import TransformerFlareModel

# Initialize model
model = TransformerFlareModel(
    sequence_length=128,
    n_features=2,  # GOES-A and GOES-B channels
    n_classes=6,   # A, B, C, M, X, Non-flare
    d_model=64,
    num_heads=8,
    num_transformer_blocks=4
)

# Build and train
model.build_model()
history = model.train(X_train, y_train, X_val, y_val)

# Save model and scaler
model.save_model('solar_flare_transformer.h5')
```

### Advanced Configuration
```python
# Custom loss weights for imbalanced classes
loss_weights = {
    'flare_class': 1.0,      # Standard classification weight
    'flare_intensity': 0.5,  # Reduced intensity weight
    'flare_duration': 0.3    # Lower duration weight
}

model = TransformerFlareModel(
    sequence_length=256,      # Longer sequences for better context
    d_model=128,             # Higher model capacity
    num_heads=16,            # More attention heads
    dropout_rate=0.15,       # Increased regularization
    learning_rate=0.0005,    # Lower learning rate
    loss_weights=loss_weights
)
```

### Attention Analysis
```python
# Load trained model
model.load_model('solar_flare_transformer.h5')

# Visualize attention patterns
model.visualize_attention(
    X_sample=test_sequences[:10],
    sample_idx=0,
    save_path='attention_analysis.png'
)

# Plot training progress
model.plot_training_history(
    history=training_history,
    save_path='training_dashboard.png'
)
```

## 📈 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic

### Regression Metrics
- **MAE**: Mean Absolute Error for intensity/duration
- **MSE**: Mean Squared Error for continuous predictions
- **R²**: Coefficient of determination

### Temporal Metrics
- **Time-to-Peak Error**: Accuracy in peak time prediction
- **Duration Error**: Accuracy in flare duration estimation
- **Rise Time Error**: Accuracy in rise phase duration

## 🔬 Scientific Applications

### Space Weather Prediction
- **Real-time Classification**: Automated flare detection from GOES data
- **Intensity Forecasting**: Predicting peak flux values
- **Duration Estimation**: Forecasting event timescales

### Solar Physics Research
- **Pattern Recognition**: Identifying pre-flare signatures
- **Multi-wavelength Correlation**: Cross-referencing with other instruments
- **Statistical Studies**: Large-scale flare occurrence analysis

### Operational Space Weather
- **Satellite Protection**: Automated anomaly detection
- **Communication Systems**: HF radio blackout prediction
- **Aviation Safety**: Polar route radiation exposure assessment

## 🔧 Model Customization

### Architecture Modifications
```python
# Increase model capacity for complex patterns
model = TransformerFlareModel(
    d_model=256,                    # Larger embedding dimension
    num_transformer_blocks=8,       # More transformer layers
    ff_dim=512,                    # Larger feed-forward network
    num_heads=16                   # More attention heads
)
```

### Multi-Instrument Support
```python
# Extend for additional data sources
model = TransformerFlareModel(
    n_features=5,  # GOES-A, GOES-B, RHESSI, SDO/EVE, STEREO
    sequence_length=512,  # Longer context for multi-instrument data
)
```

### Transfer Learning
```python
# Load pre-trained weights
base_model.load_model('pretrained_solar_transformer.h5')

# Fine-tune for specific mission
fine_tuned_model = TransformerFlareModel(n_classes=custom_classes)
fine_tuned_model.model.set_weights(base_model.model.get_weights()[:-2])
```

## 📚 References

### Solar Physics
1. Priest, E. R. (2014). *Magnetohydrodynamics of the Sun*. Cambridge University Press.
2. Aschwanden, M. J. (2005). *Physics of the Solar Corona*. Springer-Verlag.
3. Fletcher, L., et al. (2011). "An observational overview of solar flares." *Space Science Reviews*, 159(1-4), 19-106.

### Machine Learning
1. Vaswani, A., et al. (2017). "Attention is all you need." *Advances in Neural Information Processing Systems*.
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers." *arXiv preprint arXiv:1810.04805*.

### Space Weather Applications
1. Bobra, M. G., & Couvidat, S. (2015). "Solar flare prediction using SDO/HMI vector magnetic field data." *The Astrophysical Journal*, 798(2), 135.
2. Liu, C., et al. (2017). "Deep learning for solar flare prediction." *Research in Astronomy and Astrophysics*, 17(11), 111.

## 🤝 Contributing

We welcome contributions to improve the models and add new features:

1. **Bug Reports**: Submit issues with detailed error descriptions
2. **Feature Requests**: Propose new capabilities or improvements  
3. **Code Contributions**: Submit pull requests with tests and documentation
4. **Scientific Validation**: Provide feedback on model physics and performance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Citation

If you use these models in your research, please cite:

```bibtex
@software{solar_flare_transformer,
  title={Transformer-Based Solar Flare Analysis Models},
  author={Solar Flare Research Team},
  year={2025},
  url={https://github.com/pretamdas007/solar_flare_analysis}
}
```

---

*For questions, support, or collaboration opportunities, please contact the development team or submit an issue on GitHub.*
