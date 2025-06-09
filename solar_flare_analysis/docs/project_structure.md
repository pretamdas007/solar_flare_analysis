# Solar Flare Analysis – Project Structure & File Overview

This guide walks through the entire `solar_flare_analysis/` folder, describing each component, its purpose, and how to use it. Use this as a reference to navigate and extend the codebase.

---

## 1. Root Directory

```
solar_flare_analysis/
├── analyze_file.py
├── enhanced_main.py
├── fetch_and_populate_data.py
├── main.py
├── process_real_data.py
├── quick_train.py
├── train_models.py
├── simple_train.py
├── README.md
├── requirements.txt
└── ...
```

- **`main.py`** – Primary CLI entrypoint. Parses arguments for data loading, flare detection, ML training, comprehensive analysis, or synthetic data generation.
- **`enhanced_main.py`** – Alternate entrypoint with extended ML workflows (advanced architectures, nanoflare/corona‐heating analysis).
- **`analyze_file.py`, `process_real_data.py`, `fetch_and_populate_data.py`** – Utility scripts for single‐file analysis, data ingestion, and pipeline orchestration.
- **`quick_train.py`, `train_models.py`, `simple_train.py`** – Scripts to train basic, enhanced, or Bayesian flare models on synthetic or real data.
- **`README.md`** – High‐level overview, install instructions, and quick start examples.
- **`requirements.txt`** – Python dependencies (TensorFlow, xarray, netCDF4, pandas, matplotlib, etc.).

---

## 2. Configuration

### 2.1 `config.py`
Holds global settings (data paths, default channels, model parameters). Imported by core scripts to centralize configuration.

### 2.2 `config/` folder
Additional YAML/JSON settings or environment‐specific files (e.g., logging, external API credentials).

---

## 3. Data

### 3.1 `data/`
- Stores raw GOES NetCDF files (e.g., `sci_xrsf-l2-avg1m_g16_y2017_*.nc`).
- Hosts generated CSV (synthetic) and intermediate outputs.

### 3.2 `data_cache/`
- Caching layer for downloaded or preprocessed data to speed up iterative analysis.

---

## 4. Scripts & Helpers

### 4.1 `scripts/`
- Contains utility scripts such as `download_goes_data.py` (NOAA data downloader).

### 4.2 `check_*.py`, `fix_all_nulls.py`, `clean_null_bytes.py`
- Data validation and cleaning tools to sanitize raw NetCDF or CSV inputs.

### 4.3 `debug/`
- A collection of debug scripts (`debug_detection.py`, `debug_ml_input.py`, etc.) for step‐by‐step troubleshooting.

---

## 5. Model Artifacts

### 5.1 `models/`
- Serialized Keras models (`.h5`), including `best_enhanced_model.h5` and other pre‐trained weights.

### 5.2 `backend/`, `simple_backend.py`
- APIs or minimal servers to serve predictions. The `backend/` folder may contain Flask/FastAPI endpoints.

---

## 6. Application Layers

### 6.1 `ml_app/`
- Frontend/demo application for interactive inference (possibly Streamlit, Dash or Flask).

### 6.2 `notebooks/`
- Jupyter notebooks illustrating end‐to‐end workflows: data loading, preprocessing, flare detection, ML decomposition, visualization, and model evaluation.

---

## 7. Core Library (`src/`)

```
src/
├── data_processing/
│   ├── data_loader.py
│   ├── preprocess.py
│   └── utils.py
├── flare_detection/
│   ├── traditional_detection.py
│   └── overlapping.py
├── ml_models/
│   ├── flare_decomposition.py
│   ├── bayesian_flare_analysis.py
│   ├── enhanced_flare_analysis.py
│   └── utils.py
├── analysis/
│   └── power_law.py
├── visualization/
│   └── plotting.py
├── validation/
│   └── catalog_validation.py
└── evaluation/
    └── model_evaluation.py
```

- **`data_processing/`**: Loading NetCDF, cleaning, interpolation, background removal, resampling.
- **`flare_detection/`**: Peak finding, flare boundary definition, overlapping‐flare detection.
- **`ml_models/`**: Definitions for neural networks (decomposition, Bayesian energy estimation, nanoflare/corona‐heating analysis).
- **`analysis/`**: Post‐processing metrics (flare energy calculation, power‐law fitting, population comparisons).
- **`visualization/`**: Plotting time series, detected events, decomposition results, power‐law plots.
- **`validation/`**: Compare detected flares against NOAA catalogs, compute precision/recall.
- **`evaluation/`**: Reconstruction/separation metrics, learning‐curve plotting.

---

## 8. Testing (`tests/`)

- Unit and integration tests covering data loaders, detection algorithms, model training, and utility functions.
- Run with `pytest`:
  ```powershell
  pytest tests/ --maxfail=1 --disable-warnings -q
  ```

---

## 9. Output & Logs

### 9.1 `output/`
- Default location for CSV results, figures, and trained model checkpoints.

### 9.2 `enhanced_output/`
- Separate folder for outputs from `enhanced_main.py` (e.g., nanoflare analysis, corona‐heating scores).

---

## 10. How to Extend

1. **Add a new detection algorithm**: Implement under `src/flare_detection/`, add CLI flag in `main.py`.
2. **Custom model architecture**: Subclass `FlareDecompositionModel` in `src/ml_models/`, then train via `train_models.py`.
3. **New visualization**: Create a function in `src/visualization/plotting.py` and call from `main.py` or notebook demo.
4. **API endpoint**: Register a new route under `backend_server.py` or extend `ml_app/`.

---

## 11. Quick Start Examples

- **Run basic analysis**:
  ```powershell
  python main.py --data data/your_file.nc --output output/results.csv --channel B
  ```

- **Generate synthetic data**:
  ```powershell
  python main.py --generate-synthetic --synthetic-samples 5000
  ```

- **Train enhanced model**:
  ```powershell
  python quick_train.py --model-path models/enhanced.h5
  ```

---

*End of project structure documentation.*
