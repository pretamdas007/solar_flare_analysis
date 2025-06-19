# 🚀 Quick Start Guide

## Real GOES XRS Data Usage

This project is designed to work with **real GOES XRS data**. Your data directory already contains real GOES XRS files from 2017-2025!

## Installation and Setup

1. **Navigate to the project directory:**
   ```bash
   cd main/solar_flare_analysis
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```
   This will install dependencies and create directories.

3. **Test with real data:**
   ```bash
   python test_real_data.py
   ```
   This will test the pipeline with your real GOES XRS data.

## Usage Options

### Option 1: Complete Workflow with Real Data (Recommended)
```bash
python run_complete_workflow.py --use-real-data
```

This runs the entire pipeline automatically with your real GOES XRS data:
- Uses real GOES XRS data from data/ directory
- Fits flare models to detected events
- Extracts features
- Trains ML models
- Makes predictions
- Creates visualizations

### Option 2: Test Real Data Processing
```bash
python test_real_data.py
```

Quick test to verify the pipeline works with your real data.

### Option 3: Step-by-Step Execution with Real Data

1. **Fit flare models to real data:**
   ```bash
   python src/fit_flare_model.py --data_dir data --output_dir . --plot
   ```

2. **Extract features:**
   ```bash
   python src/feature_extract.py --fits_dir fits --output_dir output --target alpha
   ```

3. **Train models:**
   ```bash
   python src/train_model.py --data output/ml_data_alpha.npz --task regression --models random_forest xgboost
   ```

4. **Make predictions:**
   ```bash
   python src/predict.py --models_dir output/models --input fits/sample.json --output predictions.csv --visualize
   ```

## Real GOES XRS Data Format

Your data files have the format:
```csv
time_minutes,time_seconds,xrsa_flux_observed,xrsb_flux_observed
0,0,8.9234796e-08,1.5142843e-06
1,60,8.247599e-08,1.4532584e-06
...
```

The pipeline automatically:
- Uses `xrsa_flux_observed` (1-8 Å channel) for flare analysis
- Filters out invalid/missing data points
- Converts time to proper format
- Applies realistic flare detection thresholds

## Real Data Advantages

Using real GOES XRS data provides:
- **Authentic flare events** from actual solar observations
- **Realistic noise and background levels**
- **True flare size distributions** (A, B, C, M, X classes)
- **Multi-year dataset** for comprehensive analysis
- **Validated against known solar activity**

## Output Files

- `fits/` - Fitted flare parameters (JSON)
- `output/flare_features.csv` - Extracted features
- `output/models/` - Trained ML models
- `output/*.png` - Visualizations and plots
- `output/predictions.csv` - Prediction results

## Configuration

Edit `config.yaml` to customize:
- Model parameters
- Feature extraction settings
- File paths
- Plot settings

## Troubleshooting

**Common Issues:**

1. **Missing dependencies:** Run `pip install -r requirements.txt`

2. **No data files:** Ensure CSV files are in `data/` directory

3. **TensorFlow warnings:** These are usually safe to ignore for basic usage

4. **Memory issues:** Reduce batch sizes in `config.yaml`

**Getting Help:**

- Check the detailed README.md
- Examine the example outputs
- Review the docstrings in source files

## Next Steps

1. **Experiment with different models:** Edit the `--models` parameter
2. **Try classification:** Use `--target is_nanoflare` for nanoflare detection
3. **Tune parameters:** Modify `config.yaml` for your specific needs
4. **Add your own features:** Extend the `FlareFeatureExtractor` class
