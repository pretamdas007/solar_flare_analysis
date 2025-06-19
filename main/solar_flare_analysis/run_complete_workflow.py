"""
Complete workflow demonstration for Solar Flare Analysis Pipeline

This script demonstrates the entire pipeline from data generation to model training and prediction.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_command(command, description):
    """
    Run a command and print status
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete solar flare analysis workflow')
    parser.add_argument('--use-real-data', action='store_true', default=True,
                       help='Use real GOES XRS data (default: True)')
    parser.add_argument('--generate-synthetic', action='store_true', 
                       help='Generate synthetic data instead of using real data')
    parser.add_argument('--target', default='alpha', choices=['alpha', 'is_nanoflare'],
                       help='Target variable for ML models')
    parser.add_argument('--models', nargs='+', default=['random_forest', 'xgboost'],
                       choices=['random_forest', 'xgboost', 'bayesian_nn'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    print("🔭 Solar Flare Analysis Pipeline - Complete Workflow")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / 'src').exists():
        print("Error: Please run this script from the solar_flare_analysis directory")
        return False
    
    # Check for real data
    data_dir = Path('data')
    real_data_files = list(data_dir.glob('*.csv'))
    
    if args.generate_synthetic or not real_data_files:
        # Step 1: Generate sample data (if requested or no real data found)
        if not real_data_files:
            print("No real GOES XRS data found in data/ directory.")
            print("Generating synthetic data for demonstration...")
        
        success = run_command(
            "python generate_sample_data.py",
            "Generating synthetic GOES XRS data"
        )
        if not success:
            print("Failed to generate sample data. Exiting.")
            return False
    else:
        print(f"Using real GOES XRS data: {len(real_data_files)} files found")
        for file in real_data_files[:5]:  # Show first 5 files
            print(f"  - {file.name}")
        if len(real_data_files) > 5:
            print(f"  ... and {len(real_data_files) - 5} more files")
        print("Skipping synthetic data generation...")
    
    # Step 2: Fit flare models
    success = run_command(
        "python src/fit_flare_model.py --data_dir data --output_dir . --plot",
        "Fitting Gryciuk flare models to real GOES XRS data"
    )
    if not success:
        print("Failed to fit flare models. Exiting.")
        return False
    
    # Step 3: Extract features
    success = run_command(
        f"python src/feature_extract.py --fits_dir fits --output_dir output --target {args.target}",
        "Extracting features from fitted parameters"
    )
    if not success:
        print("Failed to extract features. Exiting.")
        return False
    
    # Step 4: Train ML models
    task = 'regression' if args.target == 'alpha' else 'classification'
    models_str = ' '.join(args.models)
    
    success = run_command(
        f"python src/train_model.py --data output/ml_data_{args.target}.npz --task {task} --models {models_str} --output_dir output",
        f"Training ML models for {args.target} prediction"
    )
    if not success:
        print("Failed to train models. Exiting.")
        return False
    
    # Step 5: Make predictions on the same data (demonstration)
    # First, find a fits file to use for prediction
    fits_dir = Path('fits')
    fits_files = list(fits_dir.glob('*_fits.json'))
    
    if fits_files:
        test_file = fits_files[0]
        success = run_command(
            f"python src/predict.py --models_dir output/models --input {test_file} --output output/predictions.csv --visualize --output_dir output",
            "Making predictions on test data"
        )
        if not success:
            print("Failed to make predictions, but continuing...")
    else:
        print("No fits files found for prediction demonstration.")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 WORKFLOW COMPLETE!")
    print("="*60)
    
    print("\nGenerated files:")
    print("📂 data/ - Sample GOES XRS data")
    print("📂 fits/ - Fitted flare parameters")
    print("📂 output/ - Features, models, and results")
    print("   ├── flare_features.csv - Extracted features")
    print("   ├── ml_data_*.npz - Preprocessed ML data")
    print("   ├── models/ - Trained models")
    print("   ├── model_comparison.png - Model performance comparison")
    print("   ├── *_feature_importance.png - Feature importance plots")
    print("   └── predictions.csv - Example predictions")
    
    print("\n📊 Key Results:")
    
    # Try to show some results
    try:
        import json
        results_path = Path('output/models/training_results.json')
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("\nModel Performance:")
            for model_name, model_results in results.items():
                if task == 'regression':
                    test_score = model_results.get('test_r2', 'N/A')
                    print(f"  {model_name}: R² = {test_score}")
                else:
                    test_score = model_results.get('test_accuracy', 'N/A')
                    print(f"  {model_name}: Accuracy = {test_score}")
    except:
        print("Could not load training results for summary.")
    
    print(f"\n🎯 Target variable: {args.target}")
    print(f"🤖 Models trained: {args.models}")
    print(f"📈 Task type: {task}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Examine the generated plots in the output/ directory")
    print("2. Use your own GOES XRS data by placing CSV files in data/")
    print("3. Experiment with different model parameters")
    print("4. Try different target variables (alpha vs nanoflare classification)")
    print("="*60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
