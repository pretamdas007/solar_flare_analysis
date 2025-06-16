# Streamlined Solar Flare Analysis CLI

This CLI has been streamlined to focus only on:
- Monte Carlo Enhanced Model (`src/ml_models/monte_carlo_enhanced_model.py`)
- Simple Bayesian Model (`src/ml_models/simple_bayesian_model.py`) 
- Power Law Analysis (`src/analysis/power_law.py`)

## Data Source
All training and analysis uses XRS CSV files from the `data/XRS/` directory:
- 2017_xrsa_xrsb.csv
- 2019_xrsa_xrsb.csv
- 2020_xrsa_xrsb.csv
- 2021_xrsa_xrsb.csv
- 2022_xrsa_xrsb.csv
- 2023_xrsa_xrsb.csv
- 2024_xrsa_xrsb.csv
- 2025_xrsa_xrsb.csv

## Usage

### Train Monte Carlo Model
```bash
python main.py --train-monte-carlo --data data/XRS --output output
```

### Train Simple Bayesian Model
```bash
python main.py --train-simple-bayesian --data data/XRS --output output
```

### Compare Both Models
```bash
python main.py --compare-models --data data/XRS --output output
```

### Power Law Analysis
```bash
python main.py --analyze --channel B --data data/XRS --output output
```

### With Visualization Options
```bash
# Interactive plots (requires Plotly)
python main.py --analyze --interactive-plots --channel B

# Publication-ready plots (PDF format)
python main.py --analyze --publication-plots --channel B
```

### Filter by Date Range
```bash
python main.py --analyze --start-date 2023-01-01 --end-date 2023-12-31
```

## Built-in Model Visualization

Both models include their own visualization capabilities:
- Monte Carlo model: Training history and uncertainty plots
- Simple Bayesian model: Training history and uncertainty analysis
- Power Law analysis: Interactive and publication-ready plots

## Output

All results are saved to the `output/` directory (or specified directory):
- Model files (.h5)
- Training history plots
- Power law analysis plots
- Model comparison charts
- Analysis results

## Model Status

The CLI automatically checks for model availability and shows status:
- ✓ Simple Bayesian: Available
- ✓ Monte Carlo: Available  
- ✓ Power Law Analysis: Available

## Example Workflow

1. **Train models:**
   ```bash
   python main.py --train-monte-carlo
   python main.py --train-simple-bayesian
   ```

2. **Compare models:**
   ```bash
   python main.py --compare-models
   ```

3. **Run analysis:**
   ```bash
   python main.py --analyze --interactive-plots
   ```

## Removed Features

The following features have been removed to focus on the core functionality:
- Enhanced flare decomposition models
- Nanoflare analysis
- Corona heating assessment
- Traditional flare detection
- Complex synthetic data generation
- Multiple visualization backends
- Legacy model support

The streamlined version focuses specifically on the Monte Carlo and Simple Bayesian models with power law analysis pipeline using XRS CSV data as requested.
