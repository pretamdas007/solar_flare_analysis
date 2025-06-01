# Solar Flare Analysis User Guide

This guide provides step-by-step instructions for using the solar flare analysis package to detect, analyze, and decompose solar flares from GOES XRS data.

## Table of Contents

1. [Installation](#installation)
2. [Data Acquisition](#data-acquisition)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Validation](#validation)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

Before installing the package, ensure you have:

- Python 3.7 or higher
- pip package manager
- (Optional) A virtual environment for isolated dependencies

### Package Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/solar_flare_analysis.git
   cd solar_flare_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Acquisition

### Using the Download Script

The package includes a convenient script for downloading GOES XRS data from NOAA NCEI:

```bash
python scripts/download_goes_data.py --start-date 2022-01-01 --end-date 2022-01-31 --output-dir data
```

Parameters:
- `--start-date`: Start date for data download (YYYY-MM-DD)
- `--end-date`: End date for data download (YYYY-MM-DD)
- `--output-dir`: Directory to save downloaded data
- `--satellite`: GOES satellite number (default: 16)

### Manual Download

You can also manually download GOES XRS data from:
- [NOAA NCEI GOES archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/)

Download netCDF files and place them in your data directory.

## Basic Usage

### Command Line Usage

The package can be run from the command line to analyze GOES XRS data files:

```bash
python main.py --data data/goes16_xrsf-l2-avg1m_s20220101_e20220102_v1-0-0.nc --output output/results_20220101.csv
```

Required parameters:
- `--data`: Path to NetCDF data file
- `--output`: Path to save results

Optional parameters:
- `--channel`: XRS channel to analyze (A or B, default: B)
- `--plot`: Whether to generate plots (default: False)
- `--plot-dir`: Directory to save plots

### Basic Python Usage

You can also use the package as a Python module:

```python
import os
from solar_flare_analysis.src.data_processing.data_loader import load_goes_data, preprocess_xrs_data, remove_background
from solar_flare_analysis.src.flare_detection.traditional_detection import detect_flare_peaks, define_flare_bounds

# Load data
data_file = 'data/goes16_xrsf-l2-avg1m_s20220101_e20220102_v1-0-0.nc'
data = load_goes_data(data_file)

# Preprocess data
df = preprocess_xrs_data(data, channel='B', remove_bad_data=True, interpolate_gaps=True)

# Remove background
df_bg = remove_background(df, window_size=60, quantile=0.1)

# Detect flare peaks
peaks = detect_flare_peaks(df_bg, 'xrsb', threshold_factor=3, window_size=5)

# Define flare boundaries
flares = define_flare_bounds(df_bg, 'xrsb', peaks['peak_index'].values,
                            start_threshold=0.5, end_threshold=0.5,
                            min_duration=1, max_duration=60)

# Print detected flares
print(f"Detected {len(flares)} flares")
print(flares[['start_time', 'peak_time', 'end_time', 'peak_flux']])
```

## Advanced Features

### ML-based Flare Decomposition

For handling overlapping flares, the package includes an ML-based decomposition method:

```python
from solar_flare_analysis.src.ml_models.flare_decomposition import FlareDecompositionModel, reconstruct_flares

# Detect overlapping flares
overlapping = detect_overlapping_flares(flares, min_overlap='1min')
print(f"Detected {len(overlapping)} overlapping flare pairs")

# Initialize and build the model
model = FlareDecompositionModel(sequence_length=64, n_features=1, max_flares=3)
model.build_model()

# Load a pre-trained model or train a new one
try:
    model.load_model('models/flare_decomposition_model')
    print("Loaded pre-trained model")
except:
    print("Training a new model with synthetic data...")
    X_train, y_train = model.generate_synthetic_data(n_samples=2000, noise_level=0.05)
    X_val, y_val = model.generate_synthetic_data(n_samples=500, noise_level=0.05)
    model.train(X_train, y_train, validation_data=(X_val, y_val), 
               epochs=50, batch_size=32, save_path='models/flare_decomposition_model')

# Process an overlapping flare segment
if overlapping:
    # Get indices of overlapping flares
    i, j, duration = overlapping[0]
    
    # Extract the segment with overlapping flares
    start_idx = min(flares.iloc[i]['start_index'], flares.iloc[j]['start_index'])
    end_idx = max(flares.iloc[i]['end_index'], flares.iloc[j]['end_index'])
    
    # Get the time series segment
    segment = df_bg.iloc[start_idx:end_idx]['xrsb_bg_subtracted'].values
    
    # Prepare the segment for the model
    segment = segment.reshape(1, -1, 1)
    
    # Decompose the flares
    original, individual_flares, combined = reconstruct_flares(model, segment, window_size=64, plot=True)
    print(f"Decomposed into {individual_flares.shape[1]} components")
```

### Power Law Analysis

To analyze flare energy distributions and fit power laws:

```python
from solar_flare_analysis.src.analysis.power_law import calculate_flare_energy, fit_power_law

# Calculate flare energies
flares_with_energy = calculate_flare_energy(flares, flux_column='xrsb_bg_subtracted')

# Fit power law to the energy distribution
energy_values = flares_with_energy['energy'].values
power_law_params = fit_power_law(energy_values, xmin=np.percentile(energy_values, 10))

print(f"Power law index: α = {power_law_params['alpha']:.2f} ± {power_law_params['alpha_err']:.2f}")
```

## Validation

### Validating Against Known Catalogs

To validate your detection method against known flare catalogs:

```python
from solar_flare_analysis.src.validation.catalog_validation import (
    download_noaa_flare_catalog, compare_detected_flares, calculate_detection_quality
)

# Download NOAA flare catalog
start_date = '2022-01-01'
end_date = '2022-01-31'
catalog_flares = download_noaa_flare_catalog(start_date, end_date)

# Compare detected flares with catalog
comparison = compare_detected_flares(
    flares, catalog_flares, 
    time_tolerance='5min', 
    flux_ratio_threshold=3.0
)

# Print comparison metrics
metrics = comparison['metrics']
print("\nValidation Metrics:")
print(f"True Positives: {metrics['true_positives']}")
print(f"False Positives: {metrics['false_positives']}")
print(f"False Negatives: {metrics['false_negatives']}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

# Calculate detailed quality metrics
quality_metrics = calculate_detection_quality(comparison)
print(f"\nMean time difference: {quality_metrics['mean_time_diff']:.2f} minutes")
print(f"Mean flux ratio: {quality_metrics['mean_flux_ratio']:.2f}")
```

### Evaluating ML Model Performance

To evaluate the performance of the ML model:

```python
from solar_flare_analysis.src.evaluation.model_evaluation import (
    evaluate_flare_reconstruction, evaluate_flare_segmentation, plot_learning_curves
)

# Evaluate reconstruction quality for a test segment
metrics = evaluate_flare_reconstruction(original.flatten(), combined.flatten(), individual_flares)

print("\nReconstruction Metrics:")
print(f"Mean Squared Error: {metrics['mse']:.4e}")
print(f"R² Score: {metrics['r2']:.4f}")
print(f"Peak Error: {metrics['peak_error']:.4e} ({metrics['relative_peak_error']:.2%})")
print(f"Energy Error: {metrics['energy_error']:.4e} ({metrics['relative_energy_error']:.2%})")

# Plot learning curves from model training
if 'history' in locals():
    fig = plot_learning_curves(history)
    fig.savefig('output/learning_curves.png')
```

## Customization

### Adjusting Detection Parameters

The flare detection parameters can be customized to suit different needs:

```python
# Custom detection parameters
detection_params = {
    'threshold_factor': 5,        # Higher value for more prominent flares
    'window_size': 10,            # Wider window for smoother detection
    'start_threshold': 0.25,      # Lower values extend flare duration
    'end_threshold': 0.25,
    'min_duration': 2,            # Minimum flare duration in minutes
    'max_duration': 120           # Maximum flare duration in minutes
}

# Apply custom parameters
peaks = detect_flare_peaks(df_bg, 'xrsb', 
                          threshold_factor=detection_params['threshold_factor'], 
                          window_size=detection_params['window_size'])

flares = define_flare_bounds(df_bg, 'xrsb', peaks['peak_index'].values,
                           start_threshold=detection_params['start_threshold'], 
                           end_threshold=detection_params['end_threshold'],
                           min_duration=detection_params['min_duration'], 
                           max_duration=detection_params['max_duration'])
```

### Customizing ML Model Architecture

You can customize the ML model architecture by extending the `FlareDecompositionModel` class:

```python
from solar_flare_analysis.src.ml_models.flare_decomposition import FlareDecompositionModel

class CustomFlareModel(FlareDecompositionModel):
    def build_model(self):
        """Build a custom model architecture."""
        input_layer = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder - more complex with residual connections
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(input_layer)
        x_shortcut = x
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.add([x, x_shortcut])  # Residual connection
        
        # Add more layers as needed
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        
        # Decoder - separate branches for each component
        components = []
        for i in range(self.max_flares):
            component = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
            component = tf.keras.layers.UpSampling1D(2)(component)
            component = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(component)
            component = tf.keras.layers.Conv1D(1, 3, activation='relu', padding='same', name=f'component_{i+1}')(component)
            components.append(component)
        
        # Combine components into a single output
        output_layer = tf.keras.layers.Concatenate(axis=2)(components)
        
        # Create and compile the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        return model

# Initialize and use the custom model
custom_model = CustomFlareModel(sequence_length=64, n_features=1, max_flares=3)
custom_model.build_model()
```

## Troubleshooting

### Common Issues and Solutions

1. **Missing Data Files**

   Error: `FileNotFoundError: [Errno 2] No such file or directory`
   
   Solution: Ensure you've downloaded the GOES XRS data files and specified the correct path.

2. **Memory Issues with Large Files**

   Error: `MemoryError` during data loading or processing
   
   Solution: Process files in smaller time chunks:
   ```python
   # Process one day at a time
   for file in data_files:
       data = load_goes_data(file)
       # Process and then clear memory
       del data
       import gc
       gc.collect()
   ```

3. **ML Model Training Issues**

   Error: `Loss is not decreasing` or poor convergence
   
   Solution: Try adjusting learning rate, batch size, or model complexity:
   ```python
   # Use custom training parameters
   optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
   model.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
   ```

4. **No Flares Detected**

   Issue: `Detected 0 flares` despite known flares in the data
   
   Solution: Adjust detection parameters for more sensitivity:
   ```python
   # More sensitive parameters
   peaks = detect_flare_peaks(df_bg, 'xrsb', threshold_factor=2, window_size=3)
   ```

### Getting Help

If you encounter issues not covered here, please:

1. Check the [API Reference](api_reference.md) for detailed function documentation
2. Look at example notebooks in the `notebooks` directory
3. Open an issue on the project's GitHub page
