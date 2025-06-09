# Installation Guide - Solar Flare Analysis System

This guide provides step-by-step instructions for setting up the Solar Flare Analysis system on Windows.

---

## 🔧 System Requirements

### **Hardware Requirements**
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB free space for data and models
- **GPU**: Optional but recommended for ML training (NVIDIA CUDA compatible)
- **CPU**: Multi-core processor recommended for data processing

### **Software Requirements**
- **Operating System**: Windows 10/11 (64-bit)
- **Python**: 3.8, 3.9, 3.10, or 3.11 (3.12 supported with additional steps)
- **Git**: For version control and updates
- **PowerShell**: For running scripts and commands

---

## 📥 1. Environment Setup

### **Step 1: Navigate to Project Directory**
```powershell
cd c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis
```

### **Step 2: Verify Python Virtual Environment**
The project should be run within the `goesflareenv` virtual environment:
```powershell
# Check if virtual environment is active
echo $env:VIRTUAL_ENV
# Should show: c:\Users\srabani\Desktop\goesflareenv

# If not active, activate it
c:\Users\srabani\Desktop\goesflareenv\Scripts\Activate.ps1
```

### **Step 3: Verify Python Version**
```powershell
python --version
# Should show Python 3.8+ 
```

---

## 📦 2. Dependencies Installation

### **Core Dependencies**
```powershell
# Install main requirements
pip install -r requirements.txt

# Install TensorFlow Probability compatibility fix
pip install tf_keras

# Verify critical packages
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
```

### **Backend Dependencies (if using web app)**
```powershell
pip install -r backend_requirements.txt
```

### **Development Dependencies (optional)**
```powershell
# For testing and development
pip install pytest pytest-cov
pip install jupyter notebook
pip install black flake8  # Code formatting
```

---

## 🌐 3. Web Application Setup (Optional)

### **Node.js Installation**
1. Download Node.js 18+ from [nodejs.org](https://nodejs.org/)
2. Install with default settings
3. Verify installation:
```powershell
node --version
npm --version
```

### **Frontend Dependencies**
```powershell
cd ml_app
npm install
cd ..
```

---

## ✅ 4. Verification & Testing

### **Basic System Check**
```powershell
# Verify package imports
python check_src_init.py

# Check for data integrity issues
python check_all_nulls.py

# Test basic imports
python -c "import src; print('Core packages imported successfully')"
```

### **Generate Test Data**
```powershell
# Generate synthetic data to verify ML pipeline
python main.py --generate-synthetic --synthetic-samples 100
```
**Expected Output:**
- 4 CSV files created in `data/` directory
- 25,600 input time series data points
- 100 target parameter records
- No error messages

### **Run Basic Analysis**
```powershell
# Test comprehensive analysis pipeline
python main.py --comprehensive --data data/
```
**Expected Behavior:**
- Model loads successfully (3.3M parameters)
- Data processing completes without errors
- Analysis pipeline executes end-to-end

### **Test Web Application (if installed)**
```powershell
cd ml_app
npm run dev
```
Then navigate to `http://localhost:3000` in your browser.

---

## 🐛 5. Troubleshooting Common Issues

### **TensorFlow Probability Compatibility**
```powershell
# Error: ModuleNotFoundError: No module named 'tf_keras'
pip install tf_keras

# If still failing, try specific version
pip install tf_keras==2.14.0
```

### **Import Errors**
```powershell
# Clean and recreate __init__.py files
python create_clean_inits.py

# Debug specific import issues
python debug_imports.py
```

### **Data File Corruption**
```powershell
# Check for null bytes in files
python check_all_nulls.py

# Clean corrupted files
python clean_null_bytes.py
python fix_all_nulls.py
```

### **Package Path Issues**
```powershell
# Ensure Python can find the project modules
python -c "import sys; print('\n'.join(sys.path))"
# Should include: c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis
```

### **Memory Issues**
```powershell
# For large datasets, monitor memory usage
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Reduce batch size for large datasets
python main.py --comprehensive --data data/ --batch-size 32
```

---

## 🔄 6. Environment Updates

### **Updating Dependencies**
```powershell
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific packages
pip install --upgrade tensorflow tensorflow-probability
pip install --upgrade numpy pandas matplotlib
```

### **Frontend Updates**
```powershell
cd ml_app
npm update
npm audit fix
cd ..
```

---

## 📁 7. Directory Structure Validation

After installation, verify your directory structure:
```powershell
# Check critical directories exist
Test-Path "data" -PathType Container
Test-Path "models" -PathType Container
Test-Path "output" -PathType Container
Test-Path "src" -PathType Container
Test-Path "config" -PathType Container

# Create missing directories
if (!(Test-Path "output")) { New-Item -ItemType Directory -Path "output" }
if (!(Test-Path "models")) { New-Item -ItemType Directory -Path "models" }
```

---

## 🚀 8. Performance Optimization

### **GPU Setup (Optional)**
```powershell
# Check GPU availability
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Install CUDA support (if GPU detected)
pip install tensorflow-gpu
```

### **Memory Optimization**
```powershell
# Set memory growth for GPU
python -c "
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print('GPU memory growth enabled')
"
```

---

## 📊 9. Data Setup

### **Sample Data Download**
```powershell
# Download sample GOES data (if needed)
python fetch_and_populate_data.py --year 2017 --days 1-5

# Verify data files
Get-ChildItem data/*.nc | Measure-Object
```

### **Data Directory Structure**
```
data/
├── sci_xrsf-l2-avg1m_g16_y2017_d001_v2-2-0.nc  # Sample GOES data
├── synthetic_input_timeseries.csv               # Generated data
├── synthetic_target_parameters.csv              # Generated parameters
├── synthetic_summary.csv                        # Generation summary
└── synthetic_example_timeseries.csv             # Example data
```

---

## 🔍 10. Final Verification

### **Complete System Test**
```powershell
# Run comprehensive test
python main.py --comprehensive --generate-synthetic --synthetic-samples 500

# Expected results:
# ✓ Synthetic data generated (4 CSV files)
# ✓ ML model loaded (3.3M parameters)
# ✓ Analysis pipeline completed
# ✓ Output files created in output/ directory
```

### **Performance Benchmark**
```powershell
# Measure execution time
Measure-Command { python main.py --generate-synthetic --synthetic-samples 1000 }
# Should complete in under 60 seconds
```

---

## 📞 11. Support & Resources

### **Getting Help**
- **Documentation**: Check `docs/` directory for detailed guides
- **Debug Scripts**: Use scripts in root directory for troubleshooting
- **Error Logs**: Check output and console for detailed error messages

### **Useful Commands**
```powershell
# Quick health check
python -c "
try:
    import src
    from src.data_processing.data_loader import GOESDataLoader
    from src.ml_models.enhanced_flare_analysis import EnhancedFlareDecompositionModel
    print('✓ All critical modules imported successfully')
    print('✓ Installation appears to be working correctly')
except Exception as e:
    print(f'✗ Installation issue detected: {e}')
"
```

### **Environment Information**
```powershell
# Display environment details
python -c "
import sys, platform, tensorflow as tf, numpy as np, pandas as pd
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'TensorFlow: {tf.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Virtual Environment: {sys.prefix}')
"
```

---

✅ **Installation Complete!** 

Your Solar Flare Analysis system is now ready for use. Proceed to the User Guide for detailed usage instructions.
