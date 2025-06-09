# Enhanced API Reference - Solar Flare Analysis System

Comprehensive API documentation for the Solar Flare Analysis system, covering all modules, classes, functions, and endpoints.

---

## 🏗️ System Architecture

### **Package Structure**
```python
solar_flare_analysis/
├── src/                           # Core library
│   ├── data_processing/           # Data loading and preprocessing
│   ├── flare_detection/          # Traditional and ML detection
│   ├── ml_models/                # Machine learning models
│   ├── analysis/                 # Statistical analysis
│   ├── visualization/            # Plotting and visualization
│   ├── validation/               # Model validation
│   └── evaluation/               # Performance evaluation
├── config/                       # Configuration settings
├── backend/                      # API server
└── ml_app/                       # Web application
```

---

## 📊 1. Data Processing Module (`src.data_processing`)

### **1.1 GOESDataLoader Class**
```python
class GOESDataLoader:
    """Enhanced GOES data loader with multiple data source support and caching"""
    
    def __init__(self, cache_dir='data_cache'):
        """Initialize the GOES data loader
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for caching downloaded/processed data, default 'data_cache'
        """
```

**Core Methods:**

#### **`load_goes_data()`**
```python
def load_goes_data(file_path, channel='B', start_time=None, end_time=None):
    """Load GOES XRS data from NetCDF file with intelligent caching
    
    Parameters
    ----------
    file_path : str
        Path to NetCDF file or directory containing multiple files
    channel : str, optional
        XRS channel ('A' or 'B'), default 'B'
        - Channel A: 0.05-0.4 nm (higher-energy soft X-rays)
        - Channel B: 0.1-0.8 nm (lower-energy soft X-rays)
    start_time : datetime or str, optional
        Start time for data extraction (ISO format)
    end_time : datetime or str, optional
        End time for data extraction (ISO format)
        
    Returns
    -------
    xarray.Dataset
        Loaded GOES data with time series and metadata
        
    Raises
    ------
    DataLoadingError
        If file cannot be loaded or is corrupted
        
    Examples
    --------
    >>> data = load_goes_data('data/goes16_2017.nc', channel='B')
    >>> filtered_data = load_goes_data('data/', start_time='2017-01-01', end_time='2017-01-02')
    """
```

#### **`preprocess_xrs_data()`**
```python
def preprocess_xrs_data(data, remove_spikes=True, interpolate_gaps=True, 
                       quality_filter=True, time_resolution='1min'):
    """Comprehensive preprocessing of XRS data for analysis
    
    Parameters
    ----------
    data : xarray.Dataset
        Raw GOES data from load_goes_data()
    remove_spikes : bool, optional
        Remove cosmic ray spikes using statistical filtering, default True
    interpolate_gaps : bool, optional
        Interpolate data gaps up to specified maximum, default True
    quality_filter : bool, optional
        Filter data based on quality flags, default True
    time_resolution : str, optional
        Target time resolution ('1min', '1s', '12s'), default '1min'
        
    Returns
    -------
    xarray.Dataset
        Preprocessed data with cleaned time series
        
    Notes
    -----
    Preprocessing steps include:
    1. Quality flag filtering
    2. Spike removal (3-sigma clipping)
    3. Gap interpolation (linear/spline)
    4. Time resampling to target resolution
    5. Data validation and integrity checks
    """
```

#### **`remove_background()`**
```python
def remove_background(flux, method='polynomial', window_size=3600, 
                     polynomial_degree=3, robust=True):
    """Remove background flux from time series using multiple methods
    
    Parameters
    ----------
    flux : array-like
        Flux time series (W/m²)
    method : str, optional
        Background removal method, default 'polynomial'
        Options: 'polynomial', 'moving_median', 'wavelet', 'spline'
    window_size : int, optional
        Window size for background estimation (seconds), default 3600
    polynomial_degree : int, optional
        Degree for polynomial fitting, default 3
    robust : bool, optional
        Use robust fitting methods, default True
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'background_corrected': Background-subtracted flux
        - 'background': Estimated background
        - 'residual': Fitting residual
        - 'quality_metrics': Background estimation quality
        
    Examples
    --------
    >>> result = remove_background(flux_data, method='polynomial', window_size=7200)
    >>> clean_flux = result['background_corrected']
    >>> background = result['background']
    """
```

---

## 🔍 2. Flare Detection Module (`src.flare_detection`)

### **2.1 Traditional Detection Functions**

#### **`detect_flare_peaks()`**
```python
def detect_flare_peaks(flux, times, threshold_factor=2.0, window_size=10, 
                      min_prominence=None, adaptive_threshold=True):
    """Detect flare peaks using enhanced traditional threshold methods
    
    Parameters
    ----------
    flux : array-like
        XRS flux time series (W/m²)
    times : array-like
        Time stamps corresponding to flux measurements
    threshold_factor : float, optional
        Factor above moving median for detection, default 2.0
    window_size : int, optional
        Window size for moving median calculation (minutes), default 10
    min_prominence : float, optional
        Minimum peak prominence (automatic if None)
    adaptive_threshold : bool, optional
        Use adaptive thresholding based on local variance, default True
        
    Returns
    -------
    dict
        Detection results containing:
        - 'peak_indices': Array of detected peak indices
        - 'peak_times': Corresponding timestamps
        - 'peak_fluxes': Peak flux values
        - 'prominences': Peak prominences
        - 'detection_quality': Quality metrics
        
    Examples
    --------
    >>> peaks = detect_flare_peaks(flux, times, threshold_factor=2.5)
    >>> peak_times = peaks['peak_times']
    >>> peak_fluxes = peaks['peak_fluxes']
    """
```

#### **`define_flare_bounds()`**
```python
def define_flare_bounds(flux, peak_indices, start_threshold=1.5, end_threshold=1.2,
                       min_duration='2min', max_duration='6h', method='adaptive'):
    """Define precise start and end times for detected flares
    
    Parameters
    ----------
    flux : array-like
        XRS flux time series
    peak_indices : array-like
        Indices of detected peaks from detect_flare_peaks()
    start_threshold : float, optional
        Fraction of peak flux for start detection, default 1.5
    end_threshold : float, optional
        Fraction of peak flux for end detection, default 1.2
    min_duration : str or timedelta, optional
        Minimum flare duration, default '2min'
    max_duration : str or timedelta, optional
        Maximum flare duration, default '6h'
    method : str, optional
        Boundary detection method ('adaptive', 'fixed', 'derivative'), default 'adaptive'
        
    Returns
    -------
    list of dict
        List of flare events, each containing:
        - 'start_time': Flare start timestamp
        - 'peak_time': Peak timestamp
        - 'end_time': Flare end timestamp
        - 'duration': Total duration
        - 'peak_flux': Maximum flux value
        - 'integrated_flux': Time-integrated flux
        - 'confidence': Detection confidence score
    """
```

#### **`detect_overlapping_flares()`**
```python
def detect_overlapping_flares(flux, times, ml_model=None, separation_threshold=0.3,
                             use_ml_separation=True):
    """Detect and separate temporally overlapping solar flares
    
    Parameters
    ----------
    flux : array-like
        XRS flux time series
    times : array-like
        Time stamps
    ml_model : object, optional
        Trained ML model for enhanced separation
    separation_threshold : float, optional
        Threshold for identifying overlapping events, default 0.3
    use_ml_separation : bool, optional
        Use ML-based separation when available, default True
        
    Returns
    -------
    dict
        Overlapping flare analysis results:
        - 'overlapping_events': List of overlapping flare groups
        - 'separated_components': Individual flare components
        - 'separation_quality': Quality metrics for separation
        - 'reconstruction_error': Reconstruction accuracy
        
    Notes
    -----
    This function combines traditional peak detection with advanced ML-based
    separation techniques to identify and decompose overlapping solar flares.
    """
```

---

## 🧠 3. Machine Learning Models (`src.ml_models`)

### **3.1 FlareDecompositionModel**
```python
class FlareDecompositionModel:
    """Base neural network model for flare decomposition and parameter estimation"""
    
    def __init__(self, input_shape=(256,), output_dim=10, model_type='dense'):
        """Initialize flare decomposition model
        
        Parameters
        ----------
        input_shape : tuple, optional
            Input time series shape, default (256,)
        output_dim : int, optional
            Output dimension for flare parameters, default 10
        model_type : str, optional
            Model architecture type ('dense', 'cnn', 'lstm'), default 'dense'
        """
```

**Core Methods:**

#### **`build_model()`**
```python
def build_model(self, learning_rate=0.001, dropout_rate=0.2, batch_norm=True):
    """Build the neural network architecture with configurable parameters
    
    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for optimizer, default 0.001
    dropout_rate : float, optional
        Dropout rate for regularization, default 0.2
    batch_norm : bool, optional
        Use batch normalization, default True
        
    Returns
    -------
    tensorflow.keras.Model
        Compiled model ready for training with:
        - Optimized architecture for time series analysis
        - Appropriate loss functions and metrics
        - Regularization for robust training
    """
```

#### **`train()`**
```python
def train(self, X_train, y_train, validation_split=0.2, epochs=100, 
          batch_size=32, early_stopping=True, save_best=True):
    """Train the model with comprehensive monitoring and callbacks
    
    Parameters
    ----------
    X_train : numpy.ndarray
        Training input data (time series windows)
    y_train : numpy.ndarray
        Training target data (flare parameters)
    validation_split : float, optional
        Fraction of data for validation, default 0.2
    epochs : int, optional
        Maximum number of training epochs, default 100
    batch_size : int, optional
        Training batch size, default 32
    early_stopping : bool, optional
        Use early stopping callback, default True
    save_best : bool, optional
        Save best model during training, default True
        
    Returns
    -------
    tensorflow.keras.callbacks.History
        Training history with metrics and validation performance
        
    Notes
    -----
    Training includes:
    - Early stopping to prevent overfitting
    - Learning rate scheduling
    - Model checkpointing
    - Comprehensive validation monitoring
    """
```

### **3.2 EnhancedFlareDecompositionModel**
```python
class EnhancedFlareDecompositionModel(FlareDecompositionModel):
    """Enhanced model with attention mechanisms and advanced architectures
    
    Features:
    - Multi-head attention for temporal modeling
    - Residual connections for deep networks
    - 3.3M parameters for complex pattern recognition
    - Advanced regularization techniques
    """
    
    def __init__(self, input_shape=(256,), output_dim=10, attention_heads=8,
                 model_depth=6, feature_dim=128):
        """Initialize enhanced model architecture
        
        Parameters
        ----------
        input_shape : tuple, optional
            Input shape, default (256,)
        output_dim : int, optional
            Output dimension, default 10
        attention_heads : int, optional
            Number of attention heads, default 8
        model_depth : int, optional
            Number of model layers, default 6
        feature_dim : int, optional
            Feature dimension for attention, default 128
        """
```

#### **`build_enhanced_architecture()`**
```python
def build_enhanced_architecture(self, use_attention=True, use_residual=True,
                               use_multi_scale=True):
    """Build enhanced model with state-of-the-art architectures
    
    Parameters
    ----------
    use_attention : bool, optional
        Include multi-head attention layers, default True
    use_residual : bool, optional
        Use residual connections, default True
    use_multi_scale : bool, optional
        Multi-scale feature extraction, default True
        
    Returns
    -------
    tensorflow.keras.Model
        Enhanced model with 3.3M parameters including:
        - Multi-head self-attention for temporal dependencies
        - Residual connections for gradient flow
        - Multi-scale convolutional layers
        - Advanced normalization and regularization
    """
```

### **3.3 NanoflareDetector**
```python
class NanoflareDetector:
    """Specialized detector for identifying nanoflares and assessing corona heating"""
    
    def __init__(self, min_energy_threshold=1e-9, alpha_threshold=2.0,
                 statistical_significance=0.95):
        """Initialize nanoflare detector with physics-based parameters
        
        Parameters
        ----------
        min_energy_threshold : float, optional
            Minimum energy for nanoflare classification (W/m²), default 1e-9
        alpha_threshold : float, optional
            Power-law slope threshold for corona heating (α > 2), default 2.0
        statistical_significance : float, optional
            Required significance level for detection, default 0.95
            
        Notes
        -----
        Nanoflares with α > 2 are considered significant contributors to
        coronal heating based on theoretical models.
        """
```

#### **`detect_nanoflares()`**
```python
def detect_nanoflares(self, flux, times, background=None, noise_level=None):
    """Detect nanoflares using enhanced sensitivity algorithms
    
    Parameters
    ----------
    flux : array-like
        XRS flux measurements (W/m²)
    times : array-like
        Time stamps corresponding to flux
    background : array-like, optional
        Pre-computed background flux
    noise_level : float, optional
        Instrument noise level (automatic if None)
        
    Returns
    -------
    dict
        Nanoflare detection results:
        - 'detected_nanoflares': List of nanoflare events
        - 'energy_distribution': Energy distribution analysis
        - 'temporal_clustering': Temporal clustering analysis
        - 'detection_statistics': Statistical summary
        
    Examples
    --------
    >>> detector = NanoflareDetector(min_energy_threshold=5e-10)
    >>> nanoflares = detector.detect_nanoflares(flux, times)
    >>> print(f\"Detected {len(nanoflares['detected_nanoflares'])} nanoflares\")
    """
```

#### **`assess_corona_heating()`**
```python
def assess_corona_heating(self, nanoflare_population, coronal_volume=None):
    """Assess corona heating contribution from nanoflare population
    
    Parameters
    ----------
    nanoflare_population : dict
        Detected nanoflare population from detect_nanoflares()
    coronal_volume : float, optional
        Coronal volume for heating rate calculation (m³)
        
    Returns
    -------
    dict
        Corona heating assessment:
        - 'total_heating_rate': Total heating rate (W)
        - 'heating_per_unit_volume': Volumetric heating rate (W/m³)
        - 'power_law_analysis': Power-law slope analysis
        - 'energy_budget': Energy budget comparison
        - 'heating_efficiency': Efficiency assessment
        
    Notes
    -----
    Calculates the contribution of nanoflares to coronal heating
    based on energy release rates and power-law distributions.
    """
```

### **3.4 BayesianFlareAnalyzer**
```python
class BayesianFlareAnalyzer:
    """Bayesian inference for flare analysis with uncertainty quantification"""
    
    def __init__(self, n_samples=1000, mcmc_chains=4, burn_in=500):
        """Initialize Bayesian analyzer with MCMC parameters
        
        Parameters
        ----------
        n_samples : int, optional
            Number of Monte Carlo samples, default 1000
        mcmc_chains : int, optional
            Number of MCMC chains for sampling, default 4
        burn_in : int, optional
            Number of burn-in samples, default 500
        """
```

#### **`estimate_energy_with_uncertainty()`**
```python
def estimate_energy_with_uncertainty(self, flux, flux_uncertainty=None, 
                                   duration_uncertainty=None, prior_params=None):
    """Estimate flare energy with full Bayesian uncertainty quantification
    
    Parameters
    ----------
    flux : array-like
        Flare flux measurements (W/m²)
    flux_uncertainty : array-like, optional
        Measurement uncertainties in flux
    duration_uncertainty : float, optional
        Uncertainty in duration measurement
    prior_params : dict, optional
        Prior distribution parameters
        
    Returns
    -------
    dict
        Bayesian energy estimation:
        - 'energy_posterior': Posterior energy distribution
        - 'credible_intervals': 68% and 95% credible intervals
        - 'uncertainty_sources': Breakdown of uncertainty contributions
        - 'model_evidence': Evidence for different models
        - 'convergence_diagnostics': MCMC convergence statistics
    """
```

#### **`monte_carlo_analysis()`**
```python
def monte_carlo_analysis(self, data, n_iterations=10000, parameter_priors=None,
                        measurement_uncertainties=None):
    """Comprehensive Monte Carlo analysis for uncertainty propagation
    
    Parameters
    ----------
    data : dict
        Input data with measurements and metadata
    n_iterations : int, optional
        Number of MC iterations, default 10000
    parameter_priors : dict, optional
        Prior distributions for model parameters
    measurement_uncertainties : dict, optional
        Measurement uncertainty specifications
        
    Returns
    -------
    dict
        Monte Carlo analysis results:
        - 'parameter_distributions': Posterior parameter distributions
        - 'correlation_matrix': Parameter correlation analysis
        - 'uncertainty_propagation': Uncertainty propagation analysis
        - 'sensitivity_analysis': Parameter sensitivity analysis
        - 'model_comparison': Bayesian model comparison
    """
```

---

## 📈 4. Analysis Tools (`src.analysis`)

### **4.1 Power Law Analysis Functions**

#### **`calculate_flare_energy()`**
```python
def calculate_flare_energy(flux, duration, distance_factor=1.0, area_correction=True,
                          spectral_model='isothermal'):
    """Calculate flare energy from flux measurements with multiple correction methods
    
    Parameters
    ----------
    flux : array-like
        Peak flux values (W/m²)
    duration : array-like
        Flare durations (seconds)
    distance_factor : float, optional
        Distance correction factor (AU), default 1.0
    area_correction : bool, optional
        Apply area correction for disk position, default True
    spectral_model : str, optional
        Spectral model for temperature correction, default 'isothermal'
        
    Returns
    -------
    dict
        Energy calculation results:
        - 'total_energy': Total radiated energy (J)
        - 'energy_uncertainty': Energy uncertainty estimates
        - 'spectral_corrections': Applied spectral corrections
        - 'distance_corrections': Distance-related corrections
        
    Notes
    -----
    Energy calculation includes:
    - Distance correction for Earth-Sun variations
    - Spectral model corrections for temperature effects
    - Area corrections for solar disk position
    - Uncertainty propagation from measurement errors
    """
```

#### **`fit_power_law()`**
```python
def fit_power_law(energies, weights=None, x_min=None, x_max=None, 
                 method='maximum_likelihood', bootstrap_samples=1000):
    """Fit power-law distribution with comprehensive statistical analysis
    
    Parameters
    ----------
    energies : array-like
        Flare energy values (J)
    weights : array-like, optional
        Statistical weights for each measurement
    x_min : float, optional
        Minimum value for power-law fitting (automatic if None)
    x_max : float, optional
        Maximum value for power-law fitting
    method : str, optional
        Fitting method ('maximum_likelihood', 'least_squares'), default 'maximum_likelihood'
    bootstrap_samples : int, optional
        Number of bootstrap samples for uncertainty, default 1000
        
    Returns
    -------
    dict
        Power-law fitting results:
        - 'alpha': Power-law slope with uncertainty
        - 'x_min': Lower cutoff with uncertainty
        - 'goodness_of_fit': Kolmogorov-Smirnov test results
        - 'alternative_models': Comparison with other distributions
        - 'bootstrap_uncertainty': Bootstrap uncertainty estimates
        - 'confidence_intervals': Parameter confidence intervals
        
    Examples
    --------
    >>> energies = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
    >>> fit_result = fit_power_law(energies, bootstrap_samples=5000)
    >>> alpha = fit_result['alpha']['value']
    >>> alpha_error = fit_result['alpha']['uncertainty']
    """
```

#### **`compare_flare_populations()`**
```python
def compare_flare_populations(population1, population2, parameters=['energy'],
                            statistical_tests=['ks_test', 'mann_whitney'],
                            bootstrap_confidence=0.95):
    """Compare two flare populations with comprehensive statistical analysis
    
    Parameters
    ----------
    population1 : dict or pandas.DataFrame
        First flare population data
    population2 : dict or pandas.DataFrame
        Second flare population data
    parameters : list, optional
        Parameters to compare, default ['energy']
    statistical_tests : list, optional
        Statistical tests to perform, default ['ks_test', 'mann_whitney']
    bootstrap_confidence : float, optional
        Confidence level for bootstrap tests, default 0.95
        
    Returns
    -------
    dict
        Population comparison results:
        - 'statistical_tests': Results of statistical tests
        - 'power_law_comparison': Power-law parameter comparison
        - 'distribution_plots': Generated comparison plots
        - 'effect_sizes': Effect size calculations
        - 'confidence_intervals': Parameter confidence intervals
        
    Notes
    -----
    Performs comprehensive comparison including:
    - Kolmogorov-Smirnov test for distribution differences
    - Mann-Whitney U test for population differences
    - Bootstrap confidence intervals
    - Power-law slope comparison with error propagation
    """
```

---

## 📊 5. Visualization Module (`src.visualization`)

### **5.1 FlareVisualization Class**
```python
class FlareVisualization:
    """Advanced plotting class for solar flare analysis with publication-quality output"""
    
    def __init__(self, style='scientific', figsize=(12, 8), dpi=300, 
                 color_scheme='viridis'):
        """Initialize visualization class with customizable parameters
        
        Parameters
        ----------
        style : str, optional
            Plot style ('scientific', 'presentation', 'publication'), default 'scientific'
        figsize : tuple, optional
            Default figure size, default (12, 8)
        dpi : int, optional
            Resolution for saved figures, default 300
        color_scheme : str, optional
            Color scheme for plots, default 'viridis'
        """
```

#### **`plot_time_series()`**
```python
def plot_time_series(self, times, flux, detected_flares=None, background=None,
                    log_scale=True, interactive=False, save_path=None, **kwargs):
    """Plot XRS time series with comprehensive annotation and overlay options
    
    Parameters
    ----------
    times : array-like
        Time stamps (datetime objects or pandas timestamps)
    flux : array-like
        Flux measurements (W/m²)
    detected_flares : list or dict, optional
        Detected flare events to overlay
    background : array-like, optional
        Background flux for overlay
    log_scale : bool, optional
        Use logarithmic y-axis scale, default True
    interactive : bool, optional
        Create interactive plot with zoom/pan, default False
    save_path : str, optional
        Path to save the figure
    **kwargs : dict
        Additional plotting parameters
        
    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        Generated plot figure with:
        - Professional styling and formatting
        - Flare event annotations
        - Background overlay
        - Customizable appearance
        
    Examples
    --------
    >>> viz = FlareVisualization(style='publication', dpi=300)
    >>> fig = viz.plot_time_series(times, flux, detected_flares=flares,
    ...                           save_path='figures/timeseries.png')
    """
```

#### **`plot_flare_decomposition()`**
```python
def plot_flare_decomposition(self, original, components, residual=None,
                           component_labels=None, time_axis=None, **kwargs):
    """Plot flare decomposition results with individual components
    
    Parameters
    ----------
    original : array-like
        Original time series
    components : list or array-like
        Separated flare components
    residual : array-like, optional
        Decomposition residual
    component_labels : list, optional
        Labels for individual components
    time_axis : array-like, optional
        Time axis for plotting
    **kwargs : dict
        Additional plotting options
        
    Returns
    -------
    matplotlib.figure.Figure
        Decomposition plot showing:
        - Original signal
        - Individual components
        - Reconstructed signal
        - Residual analysis
        - Quality metrics
    """
```

#### **`plot_power_law_analysis()`**
```python
def plot_power_law_analysis(self, energies, power_law_fit, comparison_populations=None,
                          theoretical_models=None, log_binning=True, **kwargs):
    """Plot comprehensive power-law distribution analysis
    
    Parameters
    ----------
    energies : array-like
        Flare energies for distribution analysis
    power_law_fit : dict
        Power-law fitting results from fit_power_law()
    comparison_populations : dict, optional
        Additional populations for comparison
    theoretical_models : dict, optional
        Theoretical model predictions
    log_binning : bool, optional
        Use logarithmic binning for histogram, default True
    **kwargs : dict
        Additional plotting options
        
    Returns
    -------
    matplotlib.figure.Figure
        Power-law analysis plot including:
        - Energy distribution histogram
        - Power-law fit with confidence intervals
        - Comparison with other populations
        - Goodness-of-fit statistics
        - Theoretical model overlays
    """
```

---

## 🌐 6. Backend API (`backend/`)

### **6.1 Flask Server Endpoints**

#### **Data Analysis Endpoint**
```python
@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Comprehensive data analysis endpoint with flexible parameters
    
    Request Body (JSON)
    ------------------
    {
        "file_path": "path/to/data.nc",
        "channel": "B",
        "analysis_type": "comprehensive",
        "time_range": {
            "start": "2017-01-01T00:00:00",
            "end": "2017-01-02T00:00:00"
        },
        "parameters": {
            "threshold_factor": 2.0,
            "nanoflare_detection": true,
            "bayesian_analysis": false,
            "background_method": "polynomial"
        },
        "output_format": "json"
    }
    
    Response (JSON)
    --------------
    {
        "status": "success",
        "execution_time": 15.2,
        "results": {
            "detected_flares": [
                {
                    "start_time": "2017-01-01T10:30:00",
                    "peak_time": "2017-01-01T10:45:00",
                    "end_time": "2017-01-01T11:15:00",
                    "peak_flux": 1.5e-6,
                    "total_energy": 2.3e25,
                    "class": "C1.5",
                    "confidence": 0.95
                }
            ],
            "energy_analysis": {
                "total_events": 15,
                "energy_range": [1e-9, 1e-4],
                "power_law_slope": -1.8,
                "slope_uncertainty": 0.1
            },
            "nanoflare_analysis": {
                "detected_nanoflares": 45,
                "corona_heating_rate": 3.2e-4,
                "heating_efficiency": 0.75
            },
            "statistical_summary": {
                "mean_energy": 2.1e-7,
                "median_duration": 15.5,
                "flare_rate": 2.3
            }
        },
        "quality_metrics": {
            "data_completeness": 0.98,
            "detection_confidence": 0.92,
            "background_quality": 0.89
        }
    }
    
    Error Response
    -------------
    {
        "status": "error",
        "error_type": "DataLoadingError",
        "message": "Failed to load NetCDF file",
        "details": {
            "file_path": "invalid_file.nc",
            "error_code": "NETCDF_READ_ERROR"
        },
        "suggestions": [
            "Check file format and integrity",
            "Verify file permissions",
            "Use debug_raw_data.py for troubleshooting"
        ]
    }
    """
```

#### **ML Prediction Endpoint**
```python
@app.route('/api/predict', methods=['POST'])
def ml_predict():
    """Machine learning prediction endpoint with multiple model options
    
    Request Body (JSON)
    ------------------
    {
        "time_series": [array of 256 flux values],
        "model_type": "enhanced",
        "uncertainty_quantification": true,
        "batch_prediction": false,
        "confidence_threshold": 0.8
    }
    
    Response (JSON)
    --------------
    {
        "status": "success",
        "model_info": {
            "model_type": "enhanced",
            "model_version": "v2.1.0",
            "parameters": 3300000,
            "training_date": "2024-01-15"
        },
        "predictions": {
            "flare_parameters": {
                "start_index": 45,
                "peak_index": 128,
                "end_index": 210,
                "energy_estimate": 1.2e-6,
                "duration_estimate": 25.5
            },
            "confidence_scores": {
                "overall_confidence": 0.92,
                "parameter_confidence": {
                    "start_time": 0.88,
                    "peak_time": 0.95,
                    "end_time": 0.85,
                    "energy": 0.78
                }
            },
            "uncertainty_estimates": {
                "energy_uncertainty": 2.1e-7,
                "timing_uncertainty": 3.2,
                "credible_intervals": {
                    "energy_68": [1.0e-6, 1.4e-6],
                    "energy_95": [8.5e-7, 1.6e-6]
                }
            }
        },
        "processing_time": 0.15
    }
    """
```

#### **Synthetic Data Generation**
```python
@app.route('/api/synthetic', methods=['POST'])
def generate_synthetic():
    """Generate synthetic training data with configurable parameters
    
    Request Body (JSON)
    ------------------
    {
        "n_samples": 1000,
        "output_format": "csv",
        "parameters": {
            "time_series_length": 256,
            "noise_level": 0.1,
            "flare_complexity": "medium",
            "overlapping_probability": 0.3
        },
        "save_to_disk": true,
        "return_data": false
    }
    
    Response (JSON)
    --------------
    {
        "status": "success",
        "generation_info": {
            "samples_generated": 1000,
            "time_series_points": 256000,
            "generation_time": 45.2,
            "parameter_ranges": {
                "energy_range": [1e-9, 1e-5],
                "duration_range": [2, 300]
            }
        },
        "generated_files": [
            "data/synthetic_input_timeseries.csv",
            "data/synthetic_target_parameters.csv",
            "data/synthetic_summary.csv",
            "data/synthetic_example_timeseries.csv"
        ],
        "summary_statistics": {
            "mean_energy": 2.3e-7,
            "std_energy": 1.1e-6,
            "overlapping_fraction": 0.28,
            "signal_to_noise": 12.5
        },
        "quality_metrics": {
            "data_completeness": 1.0,
            "parameter_coverage": 0.95,
            "physical_realism": 0.92
        }
    }
    """
```

#### **Model Training Endpoint**
```python
@app.route('/api/train', methods=['POST'])
def train_model():
    """Train or fine-tune ML models with monitoring and callbacks
    
    Request Body (JSON)
    ------------------
    {
        "model_type": "enhanced",
        "training_data": {
            "source": "synthetic",
            "n_samples": 5000,
            "validation_split": 0.2
        },
        "training_parameters": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": true
        },
        "model_save_path": "models/custom_model.h5",
        "monitoring": {
            "tensorboard": true,
            "checkpoint_frequency": 10
        }
    }
    
    Response (JSON)
    --------------
    {
        "status": "success",
        "training_info": {
            "model_type": "enhanced",
            "total_parameters": 3300000,
            "trainable_parameters": 3295000,
            "training_samples": 4000,
            "validation_samples": 1000
        },
        "training_results": {
            "final_epoch": 85,
            "best_validation_loss": 0.0123,
            "training_accuracy": 0.945,
            "validation_accuracy": 0.932,
            "early_stopped": true,
            "training_time": 1850.5
        },
        "model_artifacts": {
            "model_path": "models/custom_model.h5",
            "training_history": "logs/training_history.json",
            "tensorboard_logs": "logs/tensorboard/",
            "checkpoints": "checkpoints/"
        },
        "performance_metrics": {
            "mse": 0.0089,
            "mae": 0.0234,
            "r2_score": 0.912,
            "custom_metrics": {
                "energy_accuracy": 0.887,
                "timing_accuracy": 0.923
            }
        }
    }
    """
```

---

## ⚙️ 7. Configuration System (`config/`)

### **7.1 Settings Module**
```python
# config/settings.py

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = BASE_DIR / 'models'
CACHE_DIR = BASE_DIR / 'data_cache'

# GOES XRS channel specifications
XRS_CHANNELS = {
    'A': {
        'wavelength_range': (0.05, 0.4),  # nm
        'description': 'Higher-energy soft X-rays',
        'typical_background': 1e-9,  # W/m²
        'detection_threshold': 1e-8
    },
    'B': {
        'wavelength_range': (0.1, 0.8),   # nm
        'description': 'Lower-energy soft X-rays',
        'typical_background': 1e-8,   # W/m²
        'detection_threshold': 1e-7
    }
}

# Flare detection parameters
DETECTION_PARAMS = {
    'threshold_factor': 2.0,          # Standard deviations above moving median
    'window_size': 10,                # Window size for moving median (minutes)
    'start_threshold': 1.5,           # Fraction of peak flux for start time
    'end_threshold': 1.2,             # Fraction of peak flux for end time
    'min_duration': 120,              # Minimum flare duration (seconds)
    'max_duration': 21600,            # Maximum flare duration (seconds)
    'min_prominence': None,           # Minimum peak prominence (auto if None)
    'adaptive_threshold': True        # Use adaptive thresholding
}

# Machine learning model parameters
ML_MODEL_PARAMS = {
    'input_shape': (256,),            # Time series window length
    'output_dim': 10,                 # Number of output parameters
    'learning_rate': 0.001,           # Adam optimizer learning rate
    'batch_size': 32,                 # Training batch size
    'epochs': 100,                    # Maximum training epochs
    'validation_split': 0.2,          # Validation data fraction
    'early_stopping_patience': 10,    # Early stopping patience
    'reduce_lr_patience': 5,          # Learning rate reduction patience
    'dropout_rate': 0.2,              # Dropout regularization rate
    'batch_normalization': True       # Use batch normalization
}

# Enhanced model parameters
ENHANCED_MODEL_PARAMS = {
    'attention_heads': 8,             # Multi-head attention heads
    'model_depth': 6,                 # Number of model layers
    'feature_dim': 128,               # Feature dimension for attention
    'use_attention': True,            # Include attention mechanisms
    'use_residual': True,             # Use residual connections
    'use_multi_scale': True,          # Multi-scale feature extraction
    'total_parameters': 3300000       # Approximate total parameters
}

# Nanoflare detection parameters
NANOFLARE_PARAMS = {
    'energy_threshold': 1e-9,         # Minimum energy threshold (W/m²)
    'alpha_threshold': 2.0,           # Power-law slope threshold for heating
    'statistical_significance': 0.95, # Required significance level
    'detection_sensitivity': 'high',  # Detection sensitivity level
    'temporal_clustering': True,      # Analyze temporal clustering
    'energy_budget_analysis': True    # Perform energy budget calculations
}

# Bayesian analysis parameters
BAYESIAN_PARAMS = {
    'n_samples': 1000,               # Number of Monte Carlo samples
    'mcmc_chains': 4,                # Number of MCMC chains
    'burn_in': 500,                  # Number of burn-in samples
    'thinning': 1,                   # MCMC thinning factor
    'convergence_threshold': 1.1,    # R-hat convergence threshold
    'effective_sample_size': 100     # Minimum effective sample size
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'default_style': 'scientific',    # Default plot style
    'default_figsize': (12, 8),      # Default figure size
    'default_dpi': 300,              # Default resolution
    'color_scheme': 'viridis',       # Default color scheme
    'font_size': 12,                 # Default font size
    'line_width': 1.5,               # Default line width
    'marker_size': 6,                # Default marker size
    'save_format': 'png'             # Default save format
}

# API server configuration
API_CONFIG = {
    'host': '0.0.0.0',               # Server host
    'port': 8080,                    # Server port
    'debug': False,                  # Debug mode
    'threaded': True,                # Enable threading
    'max_content_length': 100 * 1024 * 1024,  # 100MB max upload
    'upload_timeout': 300,           # Upload timeout (seconds)
    'cache_timeout': 3600,           # Cache timeout (seconds)
    'rate_limit': '100/hour'         # Rate limiting
}

# Data processing configuration
DATA_PROCESSING_PARAMS = {
    'chunk_size': 10000,             # Data chunk size for large files
    'cache_enabled': True,           # Enable data caching
    'parallel_processing': True,     # Enable parallel processing
    'max_workers': 4,                # Maximum worker processes
    'memory_limit': '4GB',           # Memory limit for processing
    'interpolation_method': 'linear', # Gap interpolation method
    'spike_removal_threshold': 3.0,  # Spike removal threshold (sigma)
    'quality_flag_filtering': True   # Filter based on quality flags
}
```

---

## 🚀 8. Command Line Interface

### **8.1 Main CLI Arguments**
```bash
python main.py [OPTIONS]

Core Options:
  --data PATH                     Path to NetCDF data file or directory
  --channel {A,B,a,b}            XRS channel to analyze [default: B]
  --output PATH                  Output directory for results [default: output/]
  
Analysis Options:
  --comprehensive                Run comprehensive analysis pipeline
  --nanoflare-analysis           Perform specialized nanoflare detection
  --corona-heating               Assess corona heating contribution
  --bayesian-analysis            Include Bayesian uncertainty quantification
  
Model Options:
  --train                        Train basic ML model with synthetic data
  --train-enhanced               Train enhanced ML model (3.3M parameters)
  --train-bayesian               Train Bayesian ML model with MCMC
  --model PATH                   Path to save/load basic model
  --enhanced-model PATH          Path to enhanced model [default: models/enhanced.h5]
  
Data Generation:
  --generate-synthetic           Generate synthetic training data
  --synthetic-samples INTEGER    Number of synthetic samples [default: 5000]
  
Time Range:
  --start-date YYYY-MM-DD        Start date for analysis
  --end-date YYYY-MM-DD          End date for analysis
  
Advanced Options:
  --config PATH                  Custom configuration file
  --parallel-processing          Enable parallel processing
  --gpu-acceleration             Use GPU acceleration if available
  --cache-dir PATH              Cache directory for processed data
  --log-level {DEBUG,INFO,WARN,ERROR}  Logging level [default: INFO]
  --profile                      Enable performance profiling
```

### **8.2 Detailed Usage Examples**

#### **Basic Analysis**
```bash
# Analyze single NetCDF file
python main.py --data data/goes16_2017_001.nc --channel B --output results/

# Analyze multiple files in directory
python main.py --data "data/goes16_2017_*.nc" --comprehensive --nanoflare-analysis
```

#### **Advanced Analysis Workflows**
```bash
# Comprehensive analysis with all features
python main.py --comprehensive \
  --data data/ \
  --nanoflare-analysis \
  --corona-heating \
  --bayesian-analysis \
  --start-date "2017-01-01" \
  --end-date "2017-12-31" \
  --output results/2017_comprehensive/

# High-performance analysis with GPU
python main.py --comprehensive \
  --data data/large_dataset/ \
  --gpu-acceleration \
  --parallel-processing \
  --cache-dir cache/ \
  --log-level DEBUG
```

#### **Model Training**
```bash
# Train enhanced model with large synthetic dataset
python main.py --train-enhanced \
  --synthetic-samples 50000 \
  --enhanced-model models/custom_enhanced.h5 \
  --gpu-acceleration

# Train Bayesian model with uncertainty quantification
python main.py --train-bayesian \
  --synthetic-samples 10000 \
  --model models/bayesian_model.h5 \
  --bayesian-analysis
```

#### **Synthetic Data Generation**
```bash
# Generate large synthetic dataset for training
python main.py --generate-synthetic \
  --synthetic-samples 100000 \
  --output data/synthetic_large/

# Generate specialized nanoflare training data
python main.py --generate-synthetic \
  --synthetic-samples 20000 \
  --nanoflare-analysis \
  --output data/nanoflare_training/
```

---

## 🔍 9. Error Handling & Exceptions

### **9.1 Custom Exception Hierarchy**
```python
class SolarFlareAnalysisError(Exception):
    """Base exception for all solar flare analysis errors"""
    def __init__(self, message, error_code=None, suggestions=None):
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions or []
        super().__init__(self.message)

class DataLoadingError(SolarFlareAnalysisError):
    """Errors during data loading and preprocessing"""
    pass

class ModelTrainingError(SolarFlareAnalysisError):
    """Errors during model training or inference"""
    pass

class DetectionError(SolarFlareAnalysisError):
    """Errors in flare detection algorithms"""
    pass

class ValidationError(SolarFlareAnalysisError):
    """Errors during result validation"""
    pass

class ConfigurationError(SolarFlareAnalysisError):
    """Configuration and parameter errors"""
    pass

class APIError(SolarFlareAnalysisError):
    """API server and client errors"""
    pass
```

### **9.2 Error Response Format**
```python
{
    "error": {
        "type": "DataLoadingError",
        "code": "NETCDF_READ_ERROR",
        "message": "Failed to load NetCDF file: Invalid format or corrupted data",
        "details": {
            "file_path": "data/corrupted_file.nc",
            "file_size": 0,
            "permissions": "readable",
            "format_check": "failed",
            "corruption_detected": true
        },
        "suggestions": [
            "Check file format and integrity using ncdump -h filename.nc",
            "Verify file permissions and accessibility",
            "Re-download the file if corruption is detected",
            "Use debug_raw_data.py for detailed troubleshooting",
            "Contact data provider if file consistently fails"
        ],
        "documentation": "docs/troubleshooting.md#data-loading-errors",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

---

## 📚 10. Type Definitions & Data Structures

### **10.1 Core Type Definitions**
```python
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# Time series data types
TimeSeriesData = Union[np.ndarray, pd.Series, xr.DataArray]
TimeStamps = Union[List[datetime], pd.DatetimeIndex, np.ndarray]

# Flare event structure
FlareEvent = Dict[str, Union[float, str, datetime, Dict[str, float]]]
FlareEventList = List[FlareEvent]

# Model prediction structures
ModelPrediction = Dict[str, Union[np.ndarray, float, Dict[str, Any]]]
PredictionBatch = List[ModelPrediction]

# Analysis result structures
AnalysisResults = Dict[str, Union[FlareEventList, Dict[str, float], np.ndarray]]
StatisticalResults = Dict[str, Union[float, Tuple[float, float], Dict[str, float]]]

# Configuration structures
ConfigDict = Dict[str, Union[str, float, int, bool, Dict[str, Any]]]
ParameterDict = Dict[str, Union[float, int, str, bool]]
```

### **10.2 Data Structure Examples**

#### **Flare Event Structure**
```python
flare_event = {
    "id": "FLARE_2017001_001",
    "start_time": datetime(2017, 1, 1, 10, 30, 0),
    "peak_time": datetime(2017, 1, 1, 10, 45, 0),
    "end_time": datetime(2017, 1, 1, 11, 15, 0),
    "duration": 2700.0,  # seconds
    "peak_flux": 1.5e-6,  # W/m²
    "integrated_flux": 4.2e-4,  # W·s/m²
    "total_energy": 2.3e25,  # J
    "class": "C1.5",
    "channel": "B",
    "confidence": 0.95,
    "detection_method": "ml_enhanced",
    "background_flux": 2.1e-8,
    "signal_to_noise": 71.4,
    "uncertainty": {
        "timing": 15.0,  # seconds
        "energy": 2.1e24,  # J
        "peak_flux": 1.2e-7  # W/m²
    },
    "metadata": {
        "instrument": "GOES-16/EXIS",
        "processing_version": "v2.1.0",
        "quality_flags": [],
        "solar_coordinates": {"latitude": 15.2, "longitude": -45.8}
    }
}
```

#### **Analysis Results Structure**
```python
analysis_results = {
    "summary": {
        "total_flares": 156,
        "analysis_period": "2017-01-01 to 2017-01-31",
        "data_completeness": 0.987,
        "processing_time": 245.6
    },
    "detected_flares": [flare_event, ...],  # List of FlareEvent objects
    "energy_analysis": {
        "energy_distribution": np.array([...]),
        "power_law_fit": {
            "alpha": -1.82,
            "alpha_uncertainty": 0.05,
            "x_min": 1.2e-8,
            "goodness_of_fit": 0.89
        },
        "total_energy": 3.4e27
    },
    "nanoflare_analysis": {
        "detected_nanoflares": 234,
        "corona_heating_rate": 3.2e-4,
        "heating_efficiency": 0.75,
        "energy_budget": {
            "nanoflare_contribution": 0.45,
            "microflare_contribution": 0.35,
            "background_heating": 0.20
        }
    },
    "statistical_metrics": {
        "flare_rate": 5.03,  # flares per day
        "mean_energy": 2.1e-7,
        "median_duration": 15.5,
        "energy_range": [1e-9, 1e-4]
    },
    "quality_assessment": {
        "detection_confidence": 0.92,
        "validation_score": 0.88,
        "catalog_agreement": 0.94
    }
}
```

---

This enhanced API reference provides comprehensive documentation for all components of the Solar Flare Analysis system, including detailed parameter descriptions, return values, examples, and complete data structure specifications.
