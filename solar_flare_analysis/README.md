# Solar Flare Analysis with Machine Learning

This project uses machine learning to analyze solar flare data from GOES XRS satellite .nc files. The goal is to separate temporally overlapping flares, accurately define flare characteristics, and compare the results to traditional analysis methods.

## Project Structure

- `data/`: Contains the GOES XRS data in .nc format
- `notebooks/`: Jupyter notebooks for interactive exploration and result visualization
- `src/`: Source code for the project
  - `data_processing/`: Code for loading and preprocessing satellite data
  - `flare_detection/`: Implementation of flare detection algorithms
  - `ml_models/`: Machine learning models for flare separation
  - `analysis/`: Code for statistical analysis and power-law fitting
  - `visualization/`: Plotting utilities
- `models/`: Saved machine learning models
- `output/`: Results and figures
- `config/`: Configuration files

## Getting Started

1. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. Place your GOES XRS .nc files in the `data/` directory

3. Run the main analysis pipeline or explore the notebooks

## Analysis Pipeline

1. Load GOES XRS data
2. Train machine learning model to separate overlapping flares
3. Extract flare characteristics (start/end times, peak flux)
4. Remove background flux
5. Calculate flare energy
6. Fit power-law distributions to flare statistics
7. Compare results to traditional (non-ML) methods

## Results

The output of this analysis provides:
- Improved detection of temporally overlapping flares
- More accurate flare energy calculations
- Robust power-law slope estimation for flare energy frequency distribution
- Uncertainty quantification for all analysis steps
