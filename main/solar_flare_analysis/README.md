# 🔭 Solar Flare Shape-Based ML Model

This project builds a machine learning pipeline that fits GOES XRS solar flare data using a physically inspired flare model from **Gryciuk et al.**, and predicts either:
- the **flare energy distribution power-law index (α)**, or
- the **presence of nanoflares**.

---

## 🌟 Project Motivation

Understanding solar coronal heating requires insights into flare energy distributions. This model uses physics-based flare profiles to extract structured parameters from GOES XRS flux time series and applies ML to estimate **α** or detect nanoflare-like activity.

---

## 📈 Model Summary

Each solar flare light curve is modeled using a **Gaussian convolved with an exponential decay**:

$$f(t) = \frac{1}{2} \sqrt{\pi} A C \exp\left[D(B - t) + \frac{C^2 D^2}{4}\right] \left[\operatorname{erf}(Z) - \operatorname{erf}\left(\frac{Z - t}{C}\right)\right]$$

$$Z = \frac{2B + C^2 D}{2C}$$

Where:
- **A**: Amplitude (peak height)
- **B**: Time of peak
- **C**: Width/duration
- **D**: Decay rate

These fitted parameters become inputs to the ML model.

---

## 🛠️ Features

- 📊 Automatic flare segmentation from GOES XRS flux
- 🔍 Non-linear regression fitting to extract physical parameters
- 🤖 ML models (BNN / XGBoost / RandomForest) to:
  - Regress power-law index **α**
  - Classify **nanoflares**
- 📦 Support for uncertainty estimation and synthetic nanoflare injection

---

## 📁 Directory Structure

```
solar_flare_analysis/
│
├── data/                   # GOES XRS CSVs
├── fits/                   # Fitted flare parameters
├── models/                 # Trained ML models
├── output/                 # Visualizations, predictions
├── src/
│   ├── fit_flare_model.py  # Fit the Gryciuk model
│   ├── feature_extract.py  # Feature engineering
│   ├── train_model.py      # ML training scripts
│   └── predict.py          # Inference script
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**:
   Place GOES XRS `.csv` files inside the `data/` directory.

3. **Fit flare profiles**:
   ```bash
   python src/fit_flare_model.py
   ```

4. **Train the ML model**:
   ```bash
   python src/train_model.py
   ```

5. **Run predictions**:
   ```bash
   python src/predict.py --input fits/sample.json
   ```

---

## 📊 Output

- Fitted model parameters for each flare
- Predictions of α or nanoflare class
- Plots comparing fits and predicted results

---

## 📚 References

- Gryciuk et al., *Solar Physics*, [DOI Link]
- Mason et al., *ApJ 2023*, "Coronal Heating from Flare Distributions"

---

## 🧠 Author

**Pretam Das** – Physicist | Machine Learning | Solar Flare Research  
University of Chittagong

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
