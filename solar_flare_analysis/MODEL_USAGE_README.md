# Using Trained .h5 Models in Solar Flare Analysis

## Overview

Your `/solar_flare_analysis/models/` folder contains several trained neural network models saved in HDF5 (`.h5`) format. These models can be used to predict solar flares from GOES XRS data.

## Available Models

1. **`binary_flare_classifier.h5`** - Binary classification (flare/no-flare)
2. **`multiclass_flare_classifier.h5`** - Multi-class classification (C, M, X class flares)
3. **`energy_regression_model.h5`** - Energy estimation regression
4. **`cnn_flare_detector.h5`** - CNN-based flare detection
5. **`minimal_flare_model.h5`** - Minimal/lightweight flare detection
6. **`test_model.h5`** - Test/experimental model

## Quick Start

### 1. Basic Model Loading

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load a specific model
model = keras.models.load_model("models/binary_flare_classifier.h5")

# Check model details
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Parameters: {model.count_params():,}")

# Make a prediction (with sample data)
sample_data = np.random.random((1, model.input_shape[1]))
prediction = model.predict(sample_data)
print(f"Prediction: {prediction[0, 0]:.4f}")
```

### 2. Load All Models

```python
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def load_all_models():
    models = {}
    model_files = {
        'binary': 'binary_flare_classifier.h5',
        'multiclass': 'multiclass_flare_classifier.h5',
        'energy': 'energy_regression_model.h5',
        'cnn': 'cnn_flare_detector.h5',
        'minimal': 'minimal_flare_model.h5'
    }
    
    for name, filename in model_files.items():
        filepath = Path("models") / filename
        if filepath.exists():
            models[name] = keras.models.load_model(str(filepath))
            print(f"✅ Loaded {name}")
        else:
            print(f"❌ Not found: {filename}")
    
    return models

models = load_all_models()
```

### 3. Making Predictions

```python
def predict_flare(models, input_data):
    results = {}
    
    # Binary classification
    if 'binary' in models:
        pred = models['binary'].predict(input_data, verbose=0)
        probability = pred[0, 0]
        results['binary'] = {
            'probability': probability,
            'classification': 'flare' if probability > 0.5 else 'no_flare'
        }
    
    # Multiclass classification
    if 'multiclass' in models:
        pred = models['multiclass'].predict(input_data, verbose=0)
        classes = ['no_flare', 'C_class', 'M_class', 'X_class']
        class_idx = np.argmax(pred[0])
        results['multiclass'] = {
            'predicted_class': classes[class_idx],
            'confidence': float(np.max(pred[0]))
        }
    
    # Energy estimation
    if 'energy' in models:
        pred = models['energy'].predict(input_data, verbose=0)
        results['energy'] = {
            'estimated_energy': float(pred[0, 0])
        }
    
    return results

# Example usage
sample_input = np.random.random((1, 180))  # Adjust size as needed
predictions = predict_flare(models, sample_input)
print(predictions)
```

## Data Preprocessing

Your models expect preprocessed GOES XRS data. Here's the typical preprocessing pipeline:

### 1. Load GOES Data

```python
import pandas as pd
import numpy as np
import netCDF4 as nc

def load_goes_data(file_path):
    with nc.Dataset(file_path, 'r') as dataset:
        times = dataset.variables['time'][:]
        xrs_a = dataset.variables['xrsa_flux'][:]  # 0.05-0.4 nm
        xrs_b = dataset.variables['xrsb_flux'][:]  # 0.1-0.8 nm
        
        # Create DataFrame
        data = pd.DataFrame({
            'xrs_a': xrs_a,
            'xrs_b': xrs_b,
            'ratio': xrs_a / (xrs_b + 1e-12)
        })
        
        return data.dropna()
```

### 2. Create Features

```python
def preprocess_data(data, window_size=60):
    # Log transform XRS values
    data['xrs_a_log'] = np.log10(data['xrs_a'] + 1e-12)
    data['xrs_b_log'] = np.log10(data['xrs_b'] + 1e-12)
    
    # Select features
    features = ['xrs_a_log', 'xrs_b_log', 'ratio']
    feature_data = data[features].values
    
    # Create sliding window (for most recent data)
    if len(feature_data) >= window_size:
        window = feature_data[-window_size:]
    else:
        # Pad if insufficient data
        padding = np.zeros((window_size - len(feature_data), len(features)))
        window = np.vstack([padding, feature_data])
    
    # Flatten for model input
    return window.flatten().reshape(1, -1)
```

### 3. Complete Example

```python
# Load data
data = load_goes_data("your_goes_file.nc")

# Preprocess
model_input = preprocess_data(data)

# Load models
models = load_all_models()

# Make predictions
results = predict_flare(models, model_input)

# Display results
for model_type, result in results.items():
    print(f"{model_type}: {result}")
```

## Model Input Requirements

- **Data Shape**: Most models expect flattened time series data
- **Window Size**: Typically 60 time steps
- **Features**: Usually 3 features (xrs_a_log, xrs_b_log, ratio)
- **Input Size**: 60 × 3 = 180 features (flattened)
- **Normalization**: XRS values should be log-transformed

## Model Output Interpretation

### Binary Classifier
- **Output**: Single probability (0-1)
- **Threshold**: 0.5 (>0.5 = flare detected)

### Multiclass Classifier
- **Output**: 4 probabilities [no_flare, C_class, M_class, X_class]
- **Prediction**: Class with highest probability

### Energy Estimator
- **Output**: Estimated energy in W/m²
- **Scale**: Usually in log scale

### CNN Detector
- **Input**: May need 3D shape (batch, timesteps, features)
- **Output**: Detection score (0-1)

## Practical Scripts

I've created several helper scripts for you:

1. **`SIMPLE_MODEL_USAGE.py`** - Absolute simplest example
2. **`quick_start_models.py`** - Quick start with all models
3. **`model_usage_guide.py`** - Comprehensive guide with classes
4. **`practical_example.py`** - Complete workflow with real data

## Running the Examples

```bash
# Activate your environment
.\Scripts\activate

# Run simple example
python solar_flare_analysis\SIMPLE_MODEL_USAGE.py

# Run comprehensive example
python solar_flare_analysis\model_usage_guide.py

# Run practical example with data loading
python solar_flare_analysis\practical_example.py
```

## Troubleshooting

### Common Issues

1. **Model not found**: Check file paths and ensure models exist
2. **Shape mismatch**: Verify input data shape matches model.input_shape
3. **Import errors**: Ensure TensorFlow/Keras is installed
4. **Memory errors**: Process data in smaller batches

### Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn netcdf4
```

### Checking Model Details

```python
model = keras.models.load_model("models/binary_flare_classifier.h5")
model.summary()  # Shows architecture
print(model.input_shape)  # Shows expected input shape
print(model.output_shape)  # Shows output shape
```

## Integration with Existing Code

Your models are already integrated in several places:

- **`backend_server.py`**: Web API endpoints
- **`main.py`**: Main analysis pipeline
- **`train_models.py`**: Training and saving models

Look at these files for more advanced usage patterns.

## Next Steps

1. **Real-time Processing**: Set up continuous data monitoring
2. **Alert System**: Add threshold-based alerting
3. **Model Ensemble**: Combine multiple model predictions
4. **Performance Monitoring**: Track prediction accuracy
5. **Model Updates**: Retrain with new data periodically

## Support

If you encounter issues:
1. Check the model file exists and is readable
2. Verify your input data shape and preprocessing
3. Ensure all dependencies are installed
4. Look at the example scripts for reference patterns
