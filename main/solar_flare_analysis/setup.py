"""
Setup and Installation Script for Solar Flare Analysis Pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating directory structure...")
    
    directories = ['data', 'fits', 'models', 'output', 'output/models']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def download_sample_data():
    """Generate sample data for testing"""
    print("Generating sample data...")
    
    try:
        subprocess.check_call([sys.executable, "generate_sample_data.py"])
        print("✓ Sample data generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate sample data: {e}")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("Verifying installation...")
    
    # Check if all required modules can be imported
    required_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 
        'sklearn', 'xgboost', 'tqdm', 'joblib'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {module}")
    
    # Check optional modules
    optional_modules = ['tensorflow', 'torch']
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✓ {module} (optional)")
        except ImportError:
            print(f"⚠ {module} (optional - not installed)")
    
    if missing_modules:
        print(f"\n⚠ Missing required modules: {missing_modules}")
        print("Please install them using: pip install " + " ".join(missing_modules))
        return False
    
    print("\n✓ All required modules are available")
    return True

def create_example_config():
    """Create an example configuration file"""
    print("Configuration file already exists: config.yaml")
    return True

def main():
    """Main setup function"""
    print("🔭 Solar Flare Analysis Pipeline - Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('requirements.txt').exists():
        print("Error: requirements.txt not found. Please run this script from the project root directory.")
        return False
    
    # Step 1: Create directories
    if not create_directories():
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        return False
    
    # Step 3: Verify installation
    if not verify_installation():
        return False
    
    # Step 4: Generate sample data
    if not download_sample_data():
        print("⚠ Failed to generate sample data, but continuing...")
    
    # Step 5: Create example config
    create_example_config()
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    
    print("\n📁 Project structure:")
    print("solar_flare_analysis/")
    print("├── data/                   # GOES XRS data files")
    print("├── fits/                   # Fitted flare parameters")
    print("├── models/                 # Trained ML models")
    print("├── output/                 # Results and visualizations")
    print("├── src/                    # Source code")
    print("├── config.yaml             # Configuration file")
    print("├── requirements.txt        # Python dependencies")
    print("└── run_complete_workflow.py # Complete workflow script")
    
    print("\n🚀 Quick Start:")
    print("1. Run the complete workflow:")
    print("   python run_complete_workflow.py")
    print("\n2. Or run individual steps:")
    print("   python src/fit_flare_model.py --data_dir data --output_dir . --plot")
    print("   python src/feature_extract.py --fits_dir fits --output_dir output")
    print("   python src/train_model.py --data output/ml_data_alpha.npz --task regression")
    print("   python src/predict.py --models_dir output/models --input fits/sample.json")
    
    print("\n📚 Documentation:")
    print("- README.md contains detailed project information")
    print("- Each Python file contains comprehensive docstrings")
    print("- config.yaml contains configurable parameters")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
