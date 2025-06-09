# Solar Flare Analysis User Guide

This comprehensive guide provides step-by-step instructions for using the solar flare analysis package to detect, analyze, and decompose solar flares from GOES XRS data using advanced machine learning techniques.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Data Acquisition](#data-acquisition)
4. [Basic Usage](#basic-usage)
5. [Advanced Machine Learning Features](#advanced-machine-learning-features)
6. [Web Application Interface](#web-application-interface)
7. [API Server Usage](#api-server-usage)
8. [Analysis Workflows](#analysis-workflows)
9. [Nanoflare Detection](#nanoflare-detection)
10. [Validation & Quality Assessment](#validation--quality-assessment)
11. [Customization & Extension](#customization--extension)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting](#troubleshooting)

## Quick Start

For immediate analysis with synthetic data:

```powershell
# Navigate to project directory
cd c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis

# Activate virtual environment
.\flare_env\Scripts\Activate.ps1

# Run enhanced analysis with synthetic data
python main.py --synthetic --output enhanced_output\quick_analysis.csv --plot --web-preview
```

This will generate synthetic GOES data, perform comprehensive flare analysis, and launch a web preview of results.

## Installation

### Prerequisites

Before installing the package, ensure you have:

- **Python 3.8+** (3.9+ recommended for TensorFlow compatibility)
- **Node.js 18+** (for web application)
- **Git** (for cloning repository)
- **Windows PowerShell** (for script execution)
- **16GB+ RAM** (recommended for ML models)

### Environment Setup

1. **Clone and Navigate:**
   ```powershell
   git clone https://github.com/pretamdas007/solar_flare_analysis.git
   cd solar_flare_analysis
   ```

2. **Create Virtual Environment:**
   ```powershell
   python -m venv goesflareenv
   .\goesflareenv\Scripts\Activate.ps1
   ```

3. **Install Python Dependencies:**
   ```powershell
   # Core dependencies
   pip install -r requirements.txt
   
   # Fix TensorFlow/Keras compatibility
   pip install tf-keras
   ```

4. **Install Web Application (Optional):**
   ```powershell
   cd ml_app
   npm install
   cd ..
   ```

### Verification

Test your installation:

```powershell
# Test core functionality
python -c "from src.data_processing.data_loader import GOESDataLoader; print('✓ Core modules loaded')"

# Test ML models
python -c "import tensorflow as tf; print('✓ TensorFlow version:', tf.__version__)"

# Test with synthetic data
python main.py --synthetic --quick-test
```

## Data Acquisition

### Automated Data Download

The package includes a convenient script for downloading GOES XRS data:

```powershell
# Download recent data
python scripts\download_goes_data.py --start-date 2024-01-01 --end-date 2024-01-31 --output-dir data --satellite 18

# Download with specific parameters
python scripts\download_goes_data.py --start-date 2023-06-01 --end-date 2023-06-07 --output-dir data\june_2023 --satellite 16 --resolution 1min
```

**Parameters:**
- `--start-date`: Start date (YYYY-MM-DD format)
- `--end-date`: End date (YYYY-MM-DD format)  
- `--output-dir`: Directory to save downloaded files
- `--satellite`: GOES satellite number (16, 17, or 18)
- `--resolution`: Data resolution (1min, 5min, or 1hour)

### Manual Data Sources

**Primary Sources:**
- [NOAA NCEI GOES Archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/)
- [NOAA SWPC Real-time Data](https://www.swpc.noaa.gov/products/goes-x-ray-flux)
- [NASA Goddard Space Flight Center](https://cdaweb.gsfc.nasa.gov/)

**File Formats Supported:**
- NetCDF (.nc) - Primary format
- CSV with time, xrsa, xrsb columns
- HDF5 (.h5) files

### Synthetic Data Generation

For testing and development:

```powershell
# Generate synthetic flare data
python main.py --synthetic --duration 24 --flare-count 15 --noise-level 0.1 --output data\synthetic_24h.csv
```

## Basic Usage

### Command Line Interface

The enhanced main script provides comprehensive analysis capabilities:

```powershell
# Basic analysis with real data
python main.py --data data\goes18_xrsf-l2-avg1m_s20240101_e20240102_v1-0-0.nc --output output\results_20240101.csv --plot

# Enhanced analysis with ML models
python main.py --data data\goes_data_2024.nc --output output\enhanced_results.csv --ml-decomposition --bayesian-analysis --plot --plot-dir output\plots

# Comprehensive analysis pipeline
python main.py --data data\ --output output\comprehensive_analysis.csv --all-models --nanoflare-detection --corona-heating --web-preview
```

**Key Parameters:**

**Required:**
- `--data`: Path to NetCDF file or directory containing multiple files
- `--output`: Output file path for results CSV

**Analysis Options:**
- `--synthetic`: Generate and analyze synthetic data
- `--ml-decomposition`: Use ML models for overlapping flare separation
- `--bayesian-analysis`: Apply Bayesian uncertainty quantification
- `--nanoflare-detection`: Enable nanoflare detection algorithms
- `--corona-heating`: Assess corona heating contribution
- `--all-models`: Run all available ML models

**Visualization:**
- `--plot`: Generate basic plots
- `--plot-dir`: Directory for saving plots
- `--web-preview`: Launch web interface with results

**Performance:**
- `--parallel`: Enable parallel processing
- `--batch-size`: Set batch size for ML inference (default: 32)
- `--memory-limit`: Set memory limit in GB (default: 8)

### Python API Usage

Direct integration into Python workflows:

```python
import numpy as np
import pandas as pd
from src.data_processing.data_loader import GOESDataLoader
from src.ml_models.enhanced_flare_analysis import EnhancedFlareDecompositionModel, NanoflareDetector
from src.analysis.power_law import calculate_flare_energy, fit_power_law

# Initialize data loader
loader = GOESDataLoader()

# Load and preprocess data
data_file = 'data/goes18_xrsf-l2-avg1m_s20240101_e20240102_v1-0-0.nc'
df = loader.load_and_preprocess(data_file, channel='B', 
                               remove_bad_data=True, 
                               interpolate_gaps=True,
                               background_method='quantile')

# Traditional flare detection
from src.flare_detection.traditional_detection import detect_flare_peaks, define_flare_bounds

peaks = detect_flare_peaks(df, 'xrsb_bg_subtracted', 
                          threshold_factor=3.5, 
                          window_size=7)

flares = define_flare_bounds(df, 'xrsb_bg_subtracted', 
                           peaks['peak_index'].values,
                           start_threshold=0.3, 
                           end_threshold=0.3,
                           min_duration=2, 
                           max_duration=180)

print(f"Detected {len(flares)} flares using traditional methods")

# Enhanced ML-based detection
if ENHANCED_MODELS_AVAILABLE:
    # Initialize enhanced model
    enhanced_model = EnhancedFlareDecompositionModel(
        sequence_length=128,
        n_features=2,  # XRSA and XRSB channels
        max_flares=5
    )
    
    # Load pre-trained model or train new one
    try:
        enhanced_model.load_model('models/enhanced_flare_model.h5')
        print("✓ Loaded pre-trained enhanced model")
    except FileNotFoundError:
        print("Training enhanced model with synthetic data...")
        enhanced_model.train_with_synthetic_data(
            n_samples=5000,
            epochs=100,
            save_path='models/enhanced_flare_model.h5'
        )
    
    # Analyze overlapping flares
    ml_results = enhanced_model.analyze_time_series(
        df[['xrsa_bg_subtracted', 'xrsb_bg_subtracted']].values
    )
    
    print(f"ML model detected {len(ml_results['individual_flares'])} component flares")
```

## Advanced Machine Learning Features

### Enhanced Flare Decomposition

The package includes state-of-the-art ML models for separating overlapping flares:

```python
from src.ml_models.enhanced_flare_analysis import EnhancedFlareDecompositionModel

# Initialize enhanced model (3.3M parameters)
model = EnhancedFlareDecompositionModel(
    sequence_length=256,    # Longer sequences for complex patterns
    n_features=2,          # XRSA and XRSB channels
    max_flares=7,          # Handle up to 7 overlapping flares
    architecture='residual'  # Use ResNet-style architecture
)

# Load pre-trained model
model.load_model('models/best_enhanced_model.h5')

# Analyze complex overlapping scenario
segment_data = df[['xrsa_bg_subtracted', 'xrsb_bg_subtracted']].values[1000:1500]
decomposition = model.decompose_overlapping_flares(
    segment_data,
    confidence_threshold=0.8,
    min_component_energy=1e-8
)

print(f"Separated {len(decomposition['components'])} individual flares")
print(f"Reconstruction accuracy: {decomposition['accuracy']:.3f}")
print(f"Energy conservation: {decomposition['energy_conservation']:.1%}")
```

### Bayesian Uncertainty Quantification

Advanced uncertainty estimation for scientific analysis:

```python
from src.ml_models.bayesian_flare_analysis import BayesianFlareAnalyzer

# Initialize Bayesian analyzer
bayesian_analyzer = BayesianFlareAnalyzer(
    n_monte_carlo_samples=1000,
    dropout_rate=0.15,
    prior_type='gamma'  # Use gamma priors for energy estimates
)

# Analyze with uncertainty bounds
results = bayesian_analyzer.analyze_flare_population(
    flares_df=flares,
    time_series=df,
    channels=['xrsa_bg_subtracted', 'xrsb_bg_subtracted']
)

# Access uncertainty estimates
for i, flare in enumerate(results['flares']):
    energy_mean = flare['energy_mean']
    energy_std = flare['energy_std']
    confidence_interval = flare['energy_ci_95']
    
    print(f"Flare {i+1}: Energy = {energy_mean:.2e} ± {energy_std:.2e} J")
    print(f"  95% CI: [{confidence_interval[0]:.2e}, {confidence_interval[1]:.2e}] J")
```

### Multi-Model Ensemble

Combine multiple models for robust predictions:

```python
from src.ml_models.ensemble import FlareEnsemble

# Create ensemble of models
ensemble = FlareEnsemble([
    'models/cnn_flare_detector.h5',
    'models/lstm_flare_detector.h5', 
    'models/transformer_flare_detector.h5'
])

# Ensemble prediction with voting
ensemble_results = ensemble.predict_with_voting(
    time_series_data,
    voting_strategy='weighted',  # Weight by model performance
    confidence_threshold=0.7
)

# Access ensemble statistics
print(f"Ensemble agreement: {ensemble_results['agreement']:.1%}")
print(f"Model diversity: {ensemble_results['diversity']:.3f}")
print(f"Prediction confidence: {ensemble_results['confidence']:.3f}")
```

## Web Application Interface

### Starting the Web Application

The package includes a modern React-based web interface:

```powershell
# Start backend API server
python backend\backend_server.py --port 5000 --host 0.0.0.0

# In another terminal, start frontend
cd ml_app
npm run dev
```

**Access the interface at:** `http://localhost:3000`

### Web Interface Features

**1. Data Upload & Preview:**
- Drag-and-drop NetCDF file upload
- Real-time data preview with interactive plots
- Automatic data validation and quality checks

**2. Analysis Configuration:**
- Select detection algorithms (traditional, ML, hybrid)
- Configure parameters via intuitive sliders
- Enable/disable advanced features (Bayesian, nanoflare detection)

**3. Interactive Results:**
- Zoomable time series plots with detected flares
- Flare statistics dashboard with energy distributions
- Power-law fitting with interactive parameters
- Uncertainty visualization for Bayesian results

**4. Export & Sharing:**
- Download results as CSV, JSON, or Excel
- Export high-resolution plots (PNG, SVG, PDF)
- Generate shareable analysis reports

### Web API Usage

Direct API calls for integration:

```python
import requests
import json

# Upload data for analysis
files = {'data': open('data/goes_data.nc', 'rb')}
config = {
    'use_ml': True,
    'bayesian_analysis': True,
    'nanoflare_detection': True,
    'threshold_factor': 3.5
}

# Submit analysis job
response = requests.post('http://localhost:5000/api/analyze', 
                        files=files, 
                        data={'config': json.dumps(config)})

job_id = response.json()['job_id']

# Check job status
status_response = requests.get(f'http://localhost:5000/api/status/{job_id}')
print(f"Analysis status: {status_response.json()['status']}")

# Retrieve results when complete
results_response = requests.get(f'http://localhost:5000/api/results/{job_id}')
analysis_results = results_response.json()
```

## API Server Usage

### Starting the Backend Server

```powershell
# Development mode with hot reload
python backend\backend_server.py --debug --port 5000

# Production mode
python backend\backend_server.py --production --workers 4 --port 8080
```

### Available Endpoints

**Data Processing:**
- `POST /api/upload` - Upload GOES data files
- `GET /api/data/{file_id}` - Retrieve processed data
- `POST /api/validate` - Validate data quality

**Analysis:**
- `POST /api/analyze` - Submit analysis job
- `GET /api/status/{job_id}` - Check job status
- `GET /api/results/{job_id}` - Retrieve analysis results
- `DELETE /api/job/{job_id}` - Cancel running job

**Models:**
- `GET /api/models` - List available ML models
- `POST /api/models/predict` - Direct model prediction
- `GET /api/models/{model_id}/info` - Model specifications

**Visualization:**
- `POST /api/plot/timeseries` - Generate time series plots
- `POST /api/plot/powerlaw` - Generate power-law plots
- `POST /api/plot/distribution` - Generate energy distribution plots

### Authentication & Rate Limiting

```python
# API key authentication (production)
headers = {'Authorization': 'Bearer your_api_key_here'}
response = requests.post('http://localhost:8080/api/analyze', 
                        headers=headers, 
                        files=files)

# Rate limiting: 100 requests per hour for free tier
# Enterprise tier: 10,000 requests per hour
```

## Analysis Workflows

### Standard Solar Physics Research Pipeline

```python
# Complete analysis workflow for research
import numpy as np
import pandas as pd
from src.data_processing.data_loader import GOESDataLoader
from src.analysis.power_law import calculate_flare_energy, fit_power_law, compare_flare_populations
from src.visualization.plotting import FlareVisualization

def research_pipeline(data_file, output_dir):
    """Complete solar physics research pipeline"""
    
    # 1. Data Loading and Quality Control
    loader = GOESDataLoader()
    df = loader.load_and_preprocess(
        data_file, 
        channel='B',
        quality_control=True,
        gap_filling='linear',
        background_method='wavelet'
    )
    
    # 2. Multi-Algorithm Flare Detection
    from src.flare_detection.hybrid_detection import HybridFlareDetector
    
    detector = HybridFlareDetector()
    flares = detector.detect_all_methods(df)
    
    # 3. ML-Enhanced Analysis
    from src.ml_models.enhanced_flare_analysis import EnhancedFlareDecompositionModel
    
    ml_model = EnhancedFlareDecompositionModel()
    ml_model.load_model('models/best_enhanced_model.h5')
    
    # Separate overlapping flares
    enhanced_flares = ml_model.enhance_detection(flares, df)
    
    # 4. Nanoflare Detection for Corona Heating
    from src.ml_models.enhanced_flare_analysis import NanoflareDetector
    
    nanoflare_detector = NanoflareDetector()
    nanoflares = nanoflare_detector.detect_nanoflares(
        df, 
        power_law_threshold=2.0,
        min_energy=1e-27,  # Joules
        corona_heating_focus=True
    )
    
    # 5. Statistical Analysis
    # Calculate energies with uncertainty
    enhanced_flares = calculate_flare_energy(
        enhanced_flares, 
        flux_column='xrsb_bg_subtracted',
        uncertainty=True
    )
    
    # Fit power law
    energies = enhanced_flares['energy'].values
    power_law_params = fit_power_law(
        energies, 
        xmin='auto',
        method='mle',
        bootstrap_samples=1000
    )
    
    # 6. Corona Heating Assessment
    heating_contribution = assess_corona_heating(
        nanoflares, 
        power_law_params,
        quiet_corona_losses=1e22  # W/m²
    )
    
    # 7. Generate Publication-Quality Plots
    viz = FlareVisualization()
    
    # Time series with detected flares
    fig1 = viz.plot_time_series_with_flares(df, enhanced_flares, nanoflares)
    fig1.savefig(f'{output_dir}/time_series_analysis.png', dpi=300)
    
    # Energy distribution and power law
    fig2 = viz.plot_energy_distribution(enhanced_flares, power_law_params)
    fig2.savefig(f'{output_dir}/energy_distribution.png', dpi=300)
    
    # Corona heating contribution
    fig3 = viz.plot_corona_heating_analysis(heating_contribution)
    fig3.savefig(f'{output_dir}/corona_heating.png', dpi=300)
    
    # 8. Generate Research Report
    report = generate_research_report(
        flares=enhanced_flares,
        nanoflares=nanoflares,
        power_law_params=power_law_params,
        heating_contribution=heating_contribution
    )
    
    return {
        'flares': enhanced_flares,
        'nanoflares': nanoflares,
        'power_law': power_law_params,
        'corona_heating': heating_contribution,
        'report': report
    }

# Run complete pipeline
results = research_pipeline('data/goes_sample.nc', 'output/research_results')
```

### Multi-Year Statistical Analysis

```python
def multi_year_analysis(data_directory, start_year, end_year):
    """Analyze multiple years of GOES data for long-term trends"""
    
    yearly_results = {}
    
    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        
        # Load all files for the year
        year_files = glob.glob(f'{data_directory}/goes*{year}*.nc')
        
        combined_data = []
        for file in year_files:
            df = GOESDataLoader().load_and_preprocess(file)
            combined_data.append(df)
        
        # Concatenate year data
        year_df = pd.concat(combined_data, ignore_index=True)
        
        # Run analysis pipeline
        year_results = research_pipeline(year_df, f'output/year_{year}')
        yearly_results[year] = year_results
        
        # Memory cleanup
        del combined_data, year_df
        import gc; gc.collect()
    
    # Compare years
    comparison = compare_yearly_flare_activity(yearly_results)
    
    return yearly_results, comparison
```

## Nanoflare Detection

### Advanced Nanoflare Detection Algorithm

The package includes specialized algorithms for detecting nanoflares crucial for corona heating studies:

```python
from src.ml_models.enhanced_flare_analysis import NanoflareDetector
from src.analysis.corona_heating import CoronaHeatingAnalyzer

# Initialize nanoflare detector
nanoflare_detector = NanoflareDetector(
    energy_threshold=1e-27,     # Joules (nanoflare regime)
    duration_range=(10, 300),   # seconds
    power_law_threshold=2.0,    # |α| > 2 for significant corona heating
    spatial_resolution=1000     # km (approximate resolution)
)

# Load pre-trained nanoflare detection model
nanoflare_detector.load_model('models/nanoflare_detector.h5')

# Detect nanoflares in time series
nanoflares = nanoflare_detector.detect_nanoflares(
    time_series=df,
    channels=['xrsa_bg_subtracted', 'xrsb_bg_subtracted'],
    background_subtraction='advanced',
    noise_filtering=True,
    confidence_threshold=0.85
)

print(f"Detected {len(nanoflares)} nanoflares")

# Analyze nanoflare properties
nanoflare_stats = nanoflare_detector.analyze_nanoflare_population(nanoflares)

print("\nNanoflare Population Statistics:")
print(f"Energy range: {nanoflare_stats['energy_range']}")
print(f"Duration range: {nanoflare_stats['duration_range']}")
print(f"Frequency: {nanoflare_stats['frequency_per_hour']:.1f} events/hour")
print(f"Power law index: α = {nanoflare_stats['power_law_alpha']:.2f} ± {nanoflare_stats['alpha_error']:.2f}")
```

### Corona Heating Assessment

```python
# Assess corona heating contribution
corona_analyzer = CoronaHeatingAnalyzer()

heating_analysis = corona_analyzer.assess_heating_contribution(
    nanoflares=nanoflares,
    regular_flares=flares,
    quiet_sun_losses=1e22,  # W/m² (typical quiet corona energy loss rate)
    active_region_area=1e16  # m² (typical active region area)
)

print("\nCorona Heating Analysis:")
print(f"Nanoflare heating rate: {heating_analysis['nanoflare_heating_rate']:.2e} W/m²")
print(f"Regular flare heating rate: {heating_analysis['regular_flare_heating_rate']:.2e} W/m²")
print(f"Total heating rate: {heating_analysis['total_heating_rate']:.2e} W/m²")
print(f"Heating balance: {heating_analysis['heating_balance']:.1%}")

# Check if nanoflares can sustain corona
if heating_analysis['heating_balance'] >= 0.8:
    print("✓ Nanoflares can significantly contribute to corona heating")
else:
    print("⚠ Additional heating mechanisms may be required")
```

### Nanoflare Energy Scaling

```python
# Analyze energy scaling relationships
scaling_analysis = corona_analyzer.analyze_energy_scaling(
    nanoflares,
    energy_bins=50,
    fit_method='maximum_likelihood',
    bootstrap_iterations=1000
)

# Check for power-law behavior
if scaling_analysis['power_law_valid']:
    alpha = scaling_analysis['power_law_index']
    alpha_err = scaling_analysis['index_error']
    
    print(f"\nEnergy scaling: N(E) ∝ E^(-{alpha:.2f} ± {alpha_err:.2f})")
    
    if alpha > 2.0:
        print("✓ Power law index > 2: Nanoflares dominate energy budget")
        print("✓ Sufficient for corona heating")
    else:
        print("⚠ Power law index < 2: Limited corona heating contribution")
else:
    print("⚠ No clear power-law scaling detected")
```

## Validation & Quality Assessment

### Cross-Validation with Known Catalogs

```python
from src.validation.catalog_validation import NOAAFlareValidator, SolarMonitorValidator

# Download and compare with NOAA flare catalog
noaa_validator = NOAAFlareValidator()
noaa_catalog = noaa_validator.download_catalog(
    start_date='2024-01-01',
    end_date='2024-01-31',
    min_class='C1.0'
)

# Perform detailed comparison
validation_results = noaa_validator.validate_detection(
    detected_flares=flares,
    catalog_flares=noaa_catalog,
    time_tolerance=pd.Timedelta('5 minutes'),
    class_tolerance=1  # Allow ±1 GOES class difference
)

print("\nValidation Results:")
print(f"True Positives: {validation_results['true_positives']}")
print(f"False Positives: {validation_results['false_positives']}")
print(f"False Negatives: {validation_results['false_negatives']}")
print(f"Precision: {validation_results['precision']:.3f}")
print(f"Recall: {validation_results['recall']:.3f}")
print(f"F1-Score: {validation_results['f1_score']:.3f}")

# Detailed error analysis
error_analysis = noaa_validator.analyze_detection_errors(validation_results)
print(f"\nError Analysis:")
print(f"Mean time offset: {error_analysis['mean_time_offset']:.1f} minutes")
print(f"Mean class difference: {error_analysis['mean_class_diff']:.2f}")
print(f"Most common error type: {error_analysis['primary_error_type']}")
```

### Model Performance Evaluation

```python
from src.evaluation.model_evaluation import ModelEvaluator

# Comprehensive model evaluation
evaluator = ModelEvaluator()

# Test on synthetic data with known ground truth
test_results = evaluator.evaluate_on_synthetic(
    model=enhanced_model,
    n_test_samples=1000,
    noise_levels=[0.01, 0.05, 0.1, 0.2],
    overlap_scenarios=['none', 'partial', 'complete']
)

# Evaluate reconstruction quality
reconstruction_metrics = evaluator.evaluate_reconstruction_quality(
    test_results['original'],
    test_results['reconstructed'],
    metrics=['mse', 'mae', 'r2', 'peak_error', 'energy_conservation']
)

print("\nModel Performance:")
for metric, value in reconstruction_metrics.items():
    print(f"{metric}: {value:.4f}")

# Real data validation
real_data_metrics = evaluator.evaluate_on_real_data(
    model=enhanced_model,
    validation_files=['data/validation_set/*.nc'],
    ground_truth_method='manual_annotation'
)
```

## Customization & Extension

### Custom Detection Algorithms

Extend the framework with your own detection methods:

```python
from src.flare_detection.base_detector import BaseFlareDetector

class CustomFlareDetector(BaseFlareDetector):
    """Custom flare detection algorithm"""
    
    def __init__(self, sensitivity=0.8, custom_param=1.5):
        super().__init__()
        self.sensitivity = sensitivity
        self.custom_param = custom_param
    
    def detect_flares(self, time_series, **kwargs):
        """Implement your custom detection logic"""
        
        # Example: Wavelet-based detection
        from scipy import signal
        
        # Continuous wavelet transform
        widths = np.arange(1, 31)
        cwt_matrix = signal.cwt(time_series.values, signal.ricker, widths)
        
        # Find peaks in CWT space
        peak_indices = self._find_cwt_peaks(cwt_matrix, self.sensitivity)
        
        # Convert to flare objects
        flares = self._peaks_to_flares(peak_indices, time_series)
        
        return flares
    
    def _find_cwt_peaks(self, cwt_matrix, sensitivity):
        """Custom peak finding in CWT space"""
        # Your implementation here
        pass
    
    def _peaks_to_flares(self, peaks, time_series):
        """Convert peaks to flare DataFrame"""
        # Your implementation here
        pass

# Register custom detector
from src.flare_detection.detector_registry import register_detector

register_detector('custom_wavelet', CustomFlareDetector)

# Use in analysis
detector = CustomFlareDetector(sensitivity=0.9, custom_param=2.0)
custom_flares = detector.detect_flares(df['xrsb_bg_subtracted'])
```

### Custom ML Models

Create domain-specific neural network architectures:

```python
import tensorflow as tf
from src.ml_models.base_model import BaseMLModel

class CustomTransformerModel(BaseMLModel):
    """Transformer-based flare detection model"""
    
    def __init__(self, sequence_length=512, d_model=128, num_heads=8, num_layers=6):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
    
    def build_model(self):
        """Build transformer architecture"""
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, 2))  # XRSA, XRSB
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = self._transformer_block(x, self.d_model, self.num_heads)
        
        # Classification head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Flare probability
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def _add_positional_encoding(self, x):
        """Add positional encoding for transformer"""
        # Implementation details...
        pass
    
    def _transformer_block(self, x, d_model, num_heads):
        """Single transformer block"""
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(x, x)
        
        # Add & norm
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed forward
        ff = tf.keras.layers.Dense(d_model * 4, activation='relu')(x)
        ff = tf.keras.layers.Dense(d_model)(ff)
        
        # Add & norm
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
        
        return x

# Train custom model
custom_model = CustomTransformerModel()
custom_model.build_model()

# Custom training loop with advanced features
custom_model.train_with_custom_loss(
    train_data=train_dataset,
    validation_data=val_dataset,
    custom_loss=focal_loss,  # Handle class imbalance
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5),
        tf.keras.callbacks.ModelCheckpoint('best_custom_model.h5')
    ]
)
```

### Configuration Management

Customize behavior through configuration files:

```python
# config/custom_settings.py
class CustomConfig:
    """Extended configuration for specialized use cases"""
    
    # Detection parameters
    DETECTION_SENSITIVITY = 0.85
    MIN_FLARE_DURATION = 60  # seconds
    MAX_FLARE_DURATION = 7200  # seconds
    
    # ML model settings
    MODEL_BATCH_SIZE = 64
    MODEL_SEQUENCE_LENGTH = 256
    ENSEMBLE_VOTING_THRESHOLD = 0.7
    
    # Nanoflare detection
    NANOFLARE_ENERGY_THRESHOLD = 1e-27  # Joules
    NANOFLARE_CONFIDENCE_THRESHOLD = 0.9
    
    # Corona heating analysis
    QUIET_CORONA_LOSS_RATE = 1e22  # W/m²
    ACTIVE_REGION_AREA = 1e16  # m²
    
    # Performance settings
    MAX_MEMORY_GB = 16
    N_PARALLEL_WORKERS = 8
    GPU_MEMORY_GROWTH = True
    
    # Output settings
    PLOT_DPI = 300
    SAVE_INTERMEDIATE_RESULTS = True
    COMPRESSION_LEVEL = 6

# Load custom configuration
from config.custom_settings import CustomConfig
import config.settings as settings

# Override default settings
settings.update_from_config(CustomConfig)
```

## Troubleshooting

### Common Installation Issues

**1. TensorFlow/Keras Compatibility Problems**

```powershell
# Error: "No module named 'keras'"
pip uninstall tensorflow keras
pip install tensorflow==2.13.0
pip install tf-keras

# Error: "Could not load dynamic library 'cudnn64_8.dll'"
# Solution: Install CUDA toolkit and cuDNN
# Download from: https://developer.nvidia.com/cuda-downloads
# Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

**2. Memory Issues During Model Loading**

```python
# Error: "ResourceExhaustedError: OOM when allocating tensor"
import tensorflow as tf

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Alternative: Set memory limit
tf.config.experimental.set_memory_limit(gpus[0], 4096)  # 4GB limit
```

**3. NetCDF4 Installation Issues**

```powershell
# Error: "NetCDF4 installation failed"
# Windows solution:
conda install -c conda-forge netcdf4

# Or use pre-compiled wheels:
pip install --only-binary=netCDF4 netCDF4
```

### Runtime Issues

**1. No Flares Detected in Real Data**

```python
# Issue: Detection parameters too strict
# Solution: Adjust sensitivity parameters

from src.flare_detection.parameter_tuner import ParameterTuner

tuner = ParameterTuner()
optimal_params = tuner.tune_detection_parameters(
    data=df,
    known_flares=None,  # If no ground truth available
    method='grid_search',
    metric='detection_rate'
)

print("Optimal parameters:")
for param, value in optimal_params.items():
    print(f"  {param}: {value}")
```

**2. Poor ML Model Performance**

```python
# Issue: Model not generalizing well
# Solution: Retrain with more diverse data

from src.ml_models.training_utils import DataAugmentation

augmenter = DataAugmentation()

# Augment training data
augmented_data = augmenter.augment_training_set(
    original_data,
    augmentation_methods=['noise_injection', 'time_warping', 'amplitude_scaling'],
    augmentation_factor=3
)

# Retrain model with augmented data
model.retrain(
    augmented_data,
    epochs=100,
    early_stopping=True,
    cross_validation=True
)
```

**3. Web Interface Connection Issues**

```powershell
# Issue: Cannot connect to backend API
# Solution: Check firewall and port settings

# Check if backend is running
netstat -an | findstr :5000

# Allow through Windows Firewall
netsh advfirewall firewall add rule name="Flask Backend" dir=in action=allow protocol=TCP localport=5000

# Start backend with specific host
python backend\backend_server.py --host 0.0.0.0 --port 5000 --debug
```

**4. Large File Processing Timeouts**

```python
# Issue: Processing large files takes too long
# Solution: Use chunked processing

from src.utils.chunked_processing import ChunkedProcessor

processor = ChunkedProcessor(
    chunk_size_hours=6,     # Process 6 hours at a time
    overlap_minutes=30,     # 30-minute overlap between chunks
    parallel_chunks=True    # Process chunks in parallel
)

results = processor.process_large_file(
    'data/very_large_file.nc',
    analysis_function=research_pipeline,
    output_dir='output/chunked_results'
)

# Merge chunked results
merged_results = processor.merge_chunked_results(results)
```

### Performance Issues

**1. Slow Model Inference**

```python
# Optimize model for faster inference
def optimize_inference_speed():
    # Use TensorFlow Lite for mobile/edge deployment
    converter = tf.lite.TFLiteConverter.from_saved_model('models/enhanced_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantize model weights
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save optimized model
    with open('models/optimized_model.tflite', 'wb') as f:
        f.write(tflite_model)
```

**2. Memory Leaks in Long-Running Analysis**

```python
# Monitor and manage memory usage
import psutil
import gc

def monitor_memory_usage():
    """Monitor memory usage during analysis"""
    process = psutil.Process()
    
    def memory_callback():
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb > 8000:  # 8GB threshold
            print(f"High memory usage: {memory_mb:.1f} MB")
            gc.collect()  # Force garbage collection
            
            # Clear TensorFlow session
            tf.keras.backend.clear_session()
    
    return memory_callback

# Use memory monitoring
memory_monitor = monitor_memory_usage()

# Process files with memory monitoring
for file in large_file_list:
    process_file(file)
    memory_monitor()
```

### Data Quality Issues

**1. Corrupted or Missing Data**

```python
from src.data_processing.quality_control import DataQualityChecker

qc = DataQualityChecker()

# Check data quality before analysis
quality_report = qc.check_data_quality(data_file)

if quality_report['issues']:
    print("Data quality issues found:")
    for issue in quality_report['issues']:
        print(f"  - {issue['type']}: {issue['description']}")
        
    # Apply automatic fixes
    if quality_report['fixable']:
        fixed_data = qc.apply_quality_fixes(df, quality_report['fixes'])
        print("Applied automatic fixes")
    else:
        print("Manual intervention required")
```

### Getting Advanced Support

**For Research Collaboration:**
- Email: solar.flare.analysis@research.org
- Discord: #solar-physics-ml
- GitHub Discussions: Issues and feature requests

**For Commercial Support:**
- Enterprise licensing available
- Custom model development
- Performance optimization consulting
- 24/7 technical support

**Documentation Resources:**
- [Comprehensive Documentation](comprehensive_documentation.md)
- [API Reference](enhanced_api_reference.md)
- [Advanced Workflows](examples/advanced_workflows.md)
- [Installation Guide](installation.md)

---

## Summary

The Solar Flare Analysis package provides a complete solution for:

✅ **Advanced ML-powered flare detection and decomposition**
✅ **Nanoflare identification for corona heating studies**  
✅ **Bayesian uncertainty quantification**
✅ **Modern web interface with interactive visualizations**
✅ **Production-ready API server**
✅ **Comprehensive validation and quality assessment**
✅ **Extensible architecture for custom algorithms**
✅ **Performance optimization for large-scale analysis**

Whether you're a solar physics researcher, data scientist, or space weather analyst, this package provides the tools needed for cutting-edge solar flare analysis with the latest machine learning techniques.

**Next Steps:**
1. Follow the [Installation Guide](installation.md) for detailed setup
2. Try the Quick Start example with synthetic data
3. Explore the [Advanced Workflows](examples/advanced_workflows.md) for research use cases
4. Join our community for collaboration and support

**Happy Flare Hunting! 🌞⚡**
