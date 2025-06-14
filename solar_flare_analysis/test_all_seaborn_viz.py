#!/usr/bin/env python3
"""
Comprehensive test script for all enhanced seaborn visualizations in both 
Monte Carlo and Bayesian models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import both models
from src.ml_models.monte_carlo_enhanced_model import MonteCarloSolarFlareModel
from src.ml_models.simple_bayesian_model import SimpleBayesianFlareAnalyzer

def create_comprehensive_test_data():
    """Create comprehensive test data for both models"""
    np.random.seed(42)
    
    # Create synthetic input sequences
    n_samples = 150
    sequence_length = 128
    n_features = 2
    
    X = np.random.randn(n_samples, sequence_length, n_features) * 0.3
    
    # Add realistic flare patterns
    for i in range(n_samples):
        n_flares = np.random.poisson(1.5)  # Average 1.5 flares per sequence
        
        for flare_idx in range(n_flares):
            flare_start = np.random.randint(10, 100)
            flare_duration = np.random.randint(8, 25)
            flare_magnitude = np.random.exponential(1.5)
            
            # Add exponential decay flare pattern
            for j in range(flare_duration):
                if flare_start + j < sequence_length:
                    decay_factor = np.exp(-j/8)
                    X[i, flare_start + j, 0] += flare_magnitude * decay_factor
                    if n_features > 1:
                        X[i, flare_start + j, 1] += flare_magnitude * 0.7 * decay_factor
    
    # Create synthetic predictions with varying uncertainties
    n_outputs = 3  # peak time, peak magnitude, duration
    mean_pred = np.random.randn(n_samples, n_outputs) * 1.5
    
    # Make uncertainty dependent on prediction magnitude (realistic)
    std_pred = np.abs(mean_pred) * 0.3 + np.random.exponential(0.2, (n_samples, n_outputs))
    
    # Create confidence intervals
    z_score = 1.96  # 95% confidence
    ci_lower = mean_pred - z_score * std_pred
    ci_upper = mean_pred + z_score * std_pred
    
    predictions_dict = {
        'mean': mean_pred,
        'std': std_pred,
        'confidence_intervals': {
            '2.5th': ci_lower,
            '97.5th': ci_upper
        }
    }
    
    # Create realistic true values with some noise
    true_values = mean_pred + np.random.randn(*mean_pred.shape) * std_pred * 0.4
    
    # Create comprehensive MCMC results
    n_mcmc_samples = 800
    mcmc_results = {
        'method': 'HMC',
        'samples': {
            'weights': np.random.randn(n_mcmc_samples, 15) * 0.4,
            'bias': np.random.randn(n_mcmc_samples, 5) * 0.15,
            'noise_scale': np.abs(np.random.randn(n_mcmc_samples)) * 0.08 + 0.04
        },
        'trace': {
            'is_accepted': np.random.choice([True, False], n_mcmc_samples, p=[0.75, 0.25]),
            'step_size': np.random.uniform(0.008, 0.025, n_mcmc_samples)
        },
        'diagnostics': {
            'acceptance_rate': 0.748,
            'num_samples': n_mcmc_samples
        },
        'posterior_predictions': {
            'samples': np.random.randn(n_mcmc_samples, n_samples, n_outputs) * 0.25
        }
    }
    
    return X, predictions_dict, true_values, mcmc_results

def test_all_visualizations():
    """Test all visualization methods from both models"""
    print("🚀 Testing Enhanced Seaborn Visualizations for Solar Flare ML Models")
    print("=" * 70)
    
    # Create comprehensive test data
    X, predictions_dict, true_values, mcmc_results = create_comprehensive_test_data()
    
    # Create output directory
    output_dir = 'enhanced_output/comprehensive_seaborn_tests'
    os.makedirs(output_dir, exist_ok=True)
      # Test Monte Carlo Model Visualizations
    print("\n📊 Testing Monte Carlo Enhanced Model Visualizations...")
    print("-" * 50)
    
    try:
        mc_model = MonteCarloSolarFlareModel(
            sequence_length=128,
            n_features=2,
            n_classes=6
        )
        
        # Create some training history for the Monte Carlo model
        mc_history = {
            'loss': np.random.exponential(0.5, 50)[::-1] * 0.1 + 0.05,  # Decreasing loss
            'val_loss': np.random.exponential(0.5, 50)[::-1] * 0.12 + 0.07,
            'mae': np.random.exponential(0.3, 50)[::-1] * 0.05 + 0.02,
            'val_mae': np.random.exponential(0.3, 50)[::-1] * 0.06 + 0.025
        }
        
        # Test Monte Carlo visualizations
        tests_mc = [
            ('Training History', lambda: mc_model.plot_training_history(mc_history)),
            ('Prediction Uncertainty', lambda: mc_model.plot_prediction_uncertainty(X, predictions_dict, true_values)),
            ('Model Diagnostics', lambda: mc_model.plot_model_diagnostics(X, predictions_dict, true_values)),
            ('Uncertainty Evolution', lambda: mc_model.plot_uncertainty_evolution(X, predictions_dict))
        ]
        
        for test_name, test_func in tests_mc:
            try:
                fig = test_func()
                fig.savefig(f'{output_dir}/mc_{test_name.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Monte Carlo {test_name}: SUCCESS")
            except Exception as e:
                print(f"❌ Monte Carlo {test_name}: FAILED - {e}")
        
    except Exception as e:
        print(f"❌ Monte Carlo Model Initialization: FAILED - {e}")
    
    # Test Bayesian Model Visualizations
    print("\n🧠 Testing Bayesian Model Visualizations...")
    print("-" * 40)
    
    try:
        bayesian_model = SimpleBayesianFlareAnalyzer(
            sequence_length=128,
            n_features=2,
            max_flares=3
        )
        
        # Test Bayesian visualizations
        tests_bayesian = [
            ('Uncertainty Analysis', lambda: bayesian_model.plot_uncertainty_analysis(X, predictions_dict, true_values)),
            ('MCMC Diagnostics', lambda: bayesian_model.plot_mcmc_diagnostics(mcmc_results)),
            ('Uncertainty Evolution', lambda: bayesian_model.plot_uncertainty_evolution(X, predictions_dict))
        ]
        
        for test_name, test_func in tests_bayesian:
            try:
                fig = test_func()
                fig.savefig(f'{output_dir}/bayesian_{test_name.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Bayesian {test_name}: SUCCESS")
            except Exception as e:
                print(f"❌ Bayesian {test_name}: FAILED - {e}")
        
    except Exception as e:
        print(f"❌ Bayesian Model Initialization: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 COMPREHENSIVE TESTING COMPLETED!")
    print(f"📁 All plots saved to: {output_dir}")
    print("\n🔍 Enhanced Features Include:")
    print("   • Modern seaborn styling with custom color palettes")
    print("   • Robust data type handling (no more numpy array errors)")
    print("   • Statistical annotations (mean, median, correlations)")
    print("   • Enhanced legends and professional formatting")
    print("   • Interactive-ready plots with high DPI output")
    
    # List generated files
    try:
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.endswith('.png')]
        print(f"\n📋 Generated {len(png_files)} visualization files:")
        for file in sorted(png_files):
            print(f"   • {file}")
    except Exception as e:
        print(f"❌ Could not list output files: {e}")

if __name__ == "__main__":
    # Set global seaborn styling
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    test_all_visualizations()
