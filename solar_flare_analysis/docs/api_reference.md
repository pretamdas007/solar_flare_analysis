# Solar Flare Analysis API Reference

This document provides a comprehensive API reference for the solar flare analysis package.

## Data Processing Module

### `data_loader` Module

#### `load_goes_data(filepath)`
Loads GOES XRS data from a NetCDF file.

**Parameters:**
- `filepath` (str): Path to the NetCDF file.

**Returns:**
- `dict`: Dictionary containing time, XRS-A, and XRS-B data arrays.

#### `preprocess_xrs_data(data, channel='B', remove_bad_data=True, interpolate_gaps=True)`
Preprocesses GOES XRS data.

**Parameters:**
- `data` (dict): Data dictionary from `load_goes_data()`.
- `channel` (str): XRS channel to process ('A' or 'B').
- `remove_bad_data` (bool): Whether to remove data with bad quality flags.
- `interpolate_gaps` (bool): Whether to interpolate gaps in the data.

**Returns:**
- `pandas.DataFrame`: Preprocessed data with datetime index.

#### `remove_background(df, window_size=60, quantile=0.1)`
Removes background from XRS flux data.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with XRS flux data.
- `window_size` (int): Window size for background calculation (in minutes).
- `quantile` (float): Quantile to use as background estimate.

**Returns:**
- `pandas.DataFrame`: DataFrame with background and background-subtracted flux.

#### `calculate_derivative(df, column, window=3)`
Calculates time derivative of a data column.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with data.
- `column` (str): Column name to calculate derivative for.
- `window` (int): Window size for smoothing the derivative.

**Returns:**
- `pandas.DataFrame`: DataFrame with the derivative column added.

#### `interpolate_gaps(df, column, max_gap=30)`
Interpolates gaps in data.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with data.
- `column` (str): Column name to interpolate.
- `max_gap` (int): Maximum gap length to interpolate (in minutes).

**Returns:**
- `pandas.DataFrame`: DataFrame with interpolated data.

## Flare Detection Module

### `traditional_detection` Module

#### `detect_flare_peaks(df, flux_column, threshold_factor=3, window_size=5)`
Detects peaks in XRS flux data.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with XRS flux data.
- `flux_column` (str): Name of the flux column.
- `threshold_factor` (float): Factor above background to consider as peak.
- `window_size` (int): Window size for peak detection (in minutes).

**Returns:**
- `pandas.DataFrame`: DataFrame with detected peak indices and properties.

#### `define_flare_bounds(df, flux_column, peak_indices, start_threshold=0.5, end_threshold=0.5, min_duration=1, max_duration=60)`
Determines start and end times of flares.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with XRS flux data.
- `flux_column` (str): Name of the flux column.
- `peak_indices` (list): List of peak indices.
- `start_threshold` (float): Threshold for determining start time.
- `end_threshold` (float): Threshold for determining end time.
- `min_duration` (float): Minimum flare duration (in minutes).
- `max_duration` (float): Maximum flare duration (in minutes).

**Returns:**
- `pandas.DataFrame`: DataFrame with flare properties.

#### `detect_overlapping_flares(flares, min_overlap='1min')`
Detects overlapping flares.

**Parameters:**
- `flares` (pandas.DataFrame): DataFrame with flare properties.
- `min_overlap` (str): Minimum overlap duration.

**Returns:**
- `list`: List of tuples containing overlapping flare indices and overlap duration.

## ML Models Module

### `flare_decomposition` Module

#### `class FlareDecompositionModel`
Neural network model for separating overlapping flares.

**Methods:**

##### `__init__(sequence_length=64, n_features=1, max_flares=3)`
Initializes the model.

**Parameters:**
- `sequence_length` (int): Length of input time series.
- `n_features` (int): Number of input features.
- `max_flares` (int): Maximum number of overlapping flares to separate.

##### `build_model()`
Builds the model architecture.

**Returns:**
- `tensorflow.keras.Model`: Built model.

##### `train(X_train, y_train, validation_data=None, epochs=100, batch_size=32, save_path=None)`
Trains the model.

**Parameters:**
- `X_train` (numpy.ndarray): Training input data.
- `y_train` (numpy.ndarray): Training target data.
- `validation_data` (tuple): Validation data (X_val, y_val).
- `epochs` (int): Number of training epochs.
- `batch_size` (int): Batch size for training.
- `save_path` (str): Path to save the trained model.

**Returns:**
- `tensorflow.keras.callbacks.History`: Training history.

##### `load_model(model_path)`
Loads a trained model.

**Parameters:**
- `model_path` (str): Path to the saved model.

##### `predict(X)`
Makes predictions with the model.

**Parameters:**
- `X` (numpy.ndarray): Input data.

**Returns:**
- `numpy.ndarray`: Model predictions.

##### `generate_synthetic_data(n_samples=1000, noise_level=0.05)`
Generates synthetic training data.

**Parameters:**
- `n_samples` (int): Number of samples to generate.
- `noise_level` (float): Level of noise to add.

**Returns:**
- `tuple`: (X, y) tuple of input and target data.

#### `reconstruct_flares(model, segment, window_size=64, plot=False)`
Reconstructs individual flares from a combined signal.

**Parameters:**
- `model` (FlareDecompositionModel): Trained model.
- `segment` (numpy.ndarray): Input signal segment.
- `window_size` (int): Window size for processing.
- `plot` (bool): Whether to plot the results.

**Returns:**
- `tuple`: (original, individual_flares, combined) tuple of signals.

## Analysis Module

### `power_law` Module

#### `calculate_flare_energy(flares, flux_column='xrsb_bg_subtracted')`
Calculates the energy of flares.

**Parameters:**
- `flares` (pandas.DataFrame): DataFrame with flare properties.
- `flux_column` (str): Name of the flux column.

**Returns:**
- `pandas.DataFrame`: DataFrame with energy column added.

#### `fit_power_law(values, xmin=None, xmax=None)`
Fits a power-law distribution to data.

**Parameters:**
- `values` (numpy.ndarray): Values to fit.
- `xmin` (float): Minimum value to include in fit.
- `xmax` (float): Maximum value to include in fit.

**Returns:**
- `dict`: Dictionary with fit parameters.

#### `compare_flare_populations(population1, population2, parameter='energy')`
Compares two populations of flares.

**Parameters:**
- `population1` (pandas.DataFrame): First flare population.
- `population2` (pandas.DataFrame): Second flare population.
- `parameter` (str): Parameter to compare.

**Returns:**
- `dict`: Dictionary with comparison results.

## Validation Module

### `catalog_validation` Module

#### `download_noaa_flare_catalog(start_date, end_date, output_file=None)`
Downloads solar flare data from NOAA SWPC.

**Parameters:**
- `start_date` (str or datetime): Start date.
- `end_date` (str or datetime): End date.
- `output_file` (str): Path to save the downloaded data.

**Returns:**
- `pandas.DataFrame`: DataFrame with flare catalog data.

#### `parse_noaa_event_list(start_date, end_date, output_file=None)`
Parses the NOAA SWPC event list webpage.

**Parameters:**
- `start_date` (str or datetime): Start date.
- `end_date` (str or datetime): End date.
- `output_file` (str): Path to save the output.

**Returns:**
- `pandas.DataFrame`: DataFrame with flare catalog data.

#### `compare_detected_flares(detected_flares, catalog_flares, time_tolerance='5min', flux_ratio_threshold=3.0)`
Compares detected flares with catalog flares.

**Parameters:**
- `detected_flares` (pandas.DataFrame): DataFrame with detected flares.
- `catalog_flares` (pandas.DataFrame): DataFrame with catalog flares.
- `time_tolerance` (str or timedelta): Time tolerance for matching.
- `flux_ratio_threshold` (float): Maximum flux ratio for matching.

**Returns:**
- `dict`: Dictionary with comparison results.

#### `calculate_detection_quality(comparison_result)`
Calculates quality metrics for flare detection.

**Parameters:**
- `comparison_result` (dict): Output from `compare_detected_flares()`.

**Returns:**
- `dict`: Dictionary with quality metrics.

#### `get_flare_class_distribution(catalog_flares)`
Gets the distribution of flare classes in a catalog.

**Parameters:**
- `catalog_flares` (pandas.DataFrame): DataFrame with catalog flares.

**Returns:**
- `pandas.Series`: Series with flare class counts.

## Evaluation Module

### `model_evaluation` Module

#### `evaluate_flare_reconstruction(original, reconstructed, individual=None)`
Evaluates reconstruction quality.

**Parameters:**
- `original` (numpy.ndarray): Original signal.
- `reconstructed` (numpy.ndarray): Reconstructed signal.
- `individual` (numpy.ndarray): Individual components.

**Returns:**
- `dict`: Dictionary with evaluation metrics.

#### `evaluate_flare_segmentation(model, test_data, ground_truth=None, threshold=0.5)`
Evaluates model performance for flare segmentation.

**Parameters:**
- `model` (tensorflow.keras.Model): Trained model.
- `test_data` (numpy.ndarray): Test data.
- `ground_truth` (numpy.ndarray): Ground truth segmentations.
- `threshold` (float): Threshold for binary segmentation.

**Returns:**
- `dict`: Dictionary with evaluation metrics.

#### `plot_learning_curves(history)`
Plots learning curves from training history.

**Parameters:**
- `history` (tensorflow.keras.callbacks.History): Training history.

**Returns:**
- `matplotlib.figure.Figure`: Figure with learning curves.

#### `plot_flare_segmentation_results(test_data, predictions, indices=None)`
Plots flare segmentation results.

**Parameters:**
- `test_data` (numpy.ndarray): Test data.
- `predictions` (numpy.ndarray): Model predictions.
- `indices` (list): Indices to plot.

**Returns:**
- `matplotlib.figure.Figure`: Figure with segmentation results.

#### `calculate_flare_separation_metrics(true_individual, predicted_individual, threshold=0.2)`
Calculates metrics for flare separation.

**Parameters:**
- `true_individual` (numpy.ndarray): Ground truth individual flares.
- `predicted_individual` (numpy.ndarray): Predicted individual flares.
- `threshold` (float): Threshold for significance.

**Returns:**
- `dict`: Dictionary with flare separation metrics.

## Visualization Module

### `plotting` Module

#### `plot_xrs_time_series(df, flux_col, background_col=None, log_scale=True, figsize=(12, 6))`
Plots XRS time series data.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with XRS data.
- `flux_col` (str): Name of the flux column.
- `background_col` (str): Name of the background column.
- `log_scale` (bool): Whether to use logarithmic scale.
- `figsize` (tuple): Figure size.

**Returns:**
- `tuple`: (fig, ax) tuple.

#### `plot_detected_flares(df, flares, flux_col, figsize=(12, 6))`
Plots detected flares on flux time series.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with XRS data.
- `flares` (pandas.DataFrame): DataFrame with flare properties.
- `flux_col` (str): Name of the flux column.
- `figsize` (tuple): Figure size.

**Returns:**
- `tuple`: (fig, ax) tuple.

#### `plot_flare_decomposition(original, components, reconstructed, figsize=(12, 6))`
Plots flare decomposition results.

**Parameters:**
- `original` (numpy.ndarray): Original signal.
- `components` (numpy.ndarray): Decomposed components.
- `reconstructed` (numpy.ndarray): Reconstructed signal.
- `figsize` (tuple): Figure size.

**Returns:**
- `tuple`: (fig, ax) tuple.

#### `plot_power_law_comparison(populations, labels, parameter='energy', figsize=(10, 6))`
Plots power-law comparison between populations.

**Parameters:**
- `populations` (list): List of DataFrames with populations.
- `labels` (list): List of labels for populations.
- `parameter` (str): Parameter to compare.
- `figsize` (tuple): Figure size.

**Returns:**
- `tuple`: (fig, ax) tuple.
