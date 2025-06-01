# Solar Flare Analysis Notebooks

This directory contains Jupyter notebooks for exploring and analyzing solar flares from GOES XRS data. These notebooks demonstrate the key features of the solar flare analysis package and serve as practical tutorials.

## Prerequisites

Before running these notebooks, ensure you have:

1. Installed all required dependencies with `pip install -r ../requirements.txt`
2. Downloaded sample GOES XRS data files (.nc format) to the `../data/` directory

If you don't have GOES XRS data, you can download it using the provided script:

```bash
python ../scripts/download_goes_data.py --satellite goes16 --start_date 20220601
```

## Notebooks Overview

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Loading and examining GOES XRS data
- Preprocessing time series data
- Removing background flux
- Visualizing solar X-ray time series

### 2. Traditional Flare Detection (`02_traditional_flare_detection.ipynb`)
- Peak detection algorithms
- Defining flare boundaries (start/end times)
- Identifying overlapping flares
- Flare classification (A, B, C, M, X)
- Analyzing flare duration statistics

### 3. ML Flare Decomposition (`03_ml_flare_decomposition.ipynb`)
- Building and training a neural network model
- Generating synthetic training data
- Separating overlapping flares
- Evaluating model performance
- Applying the model to real GOES XRS data

### 4. Power Law Analysis (`04_power_law_analysis.ipynb`)
- Calculating flare energies
- Fitting power-law distributions
- Comparing traditional vs ML-separated results
- Statistical significance testing
- Implications for solar physics

## How to Use

1. Start Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. Open the notebooks in sequence, starting with `01_data_exploration.ipynb`

3. Execute the cells in order, reading the explanations and observing the outputs

## Notes

- If you don't see graphs when running the notebooks, you might need to set a different matplotlib backend:
  ```python
  import matplotlib
  matplotlib.use('TkAgg')  # or try 'nbAgg', 'WebAgg', etc.
  ```

- These notebooks are designed to work with GOES XRS data in NetCDF (.nc) format from GOES-16, GOES-17, or GOES-18 satellites.

- Computation times may vary depending on your hardware, especially for the ML model training in notebook 3.

## References

- [GOES XRS Documentation (NOAA)](https://www.ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf)
- [NOAA NCEI GOES Data Access](https://www.ngdc.noaa.gov/stp/satellite/goes/dataaccess.html)
- Aschwanden, M. J. (2011). Self-Organized Criticality in Astrophysics: The Statistics of Nonlinear Processes in the Universe. Springer.
- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. SIAM Review, 51(4), 661–703.
