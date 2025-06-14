#!/usr/bin/env python3
"""
Test script for the enhanced Bayesian model's seaborn visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the Bayesian model
from src.ml_models.simple_bayesian_model import SimpleBayesianFlareAnalyzer

def create_test_data():
    """Create synthetic test data for visualization testing"""
    np.random.seed(42)
    
    # Create synthetic input sequences
    n_samples = 100
    sequence_length = 128
    n_features = 2
    
    X = np.random.randn(n_samples, sequence_length, n_features) * 0.5
    
    # Add some structure to the data (simulated flare patterns)
    for i in range(n_samples):
        # Random flare location
        flare_start = np.random.randint(20, 80)
        flare_duration = np.random.randint(10, 30)
        flare_magnitude = np.random.exponential(2.0)
        
        # Add flare pattern
        for j in range(flare_duration):
            if flare_start + j < sequence_length:
                X[i, flare_start + j, 0] += flare_magnitude * np.exp(-j/10)
                if n_features > 1:
                    X[i, flare_start + j, 1] += flare_magnitude * 0.8 * np.exp(-j/8)
    
    # Create synthetic predictions with uncertainties
    n_outputs = 3  # Example: peak time, peak magnitude, duration
    mean_pred = np.random.randn(n_samples, n_outputs) * 2.0
    std_pred = np.abs(np.random.randn(n_samples, n_outputs) * 0.5) + 0.1
    
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
    
    # Create synthetic true values
    true_values = mean_pred + np.random.randn(*mean_pred.shape) * std_pred * 0.5
    
    # Create synthetic MCMC results
    n_mcmc_samples = 500
    mcmc_results = {
        'method': 'HMC',
        'samples': {
            'weights': np.random.randn(n_mcmc_samples, 10) * 0.5,
            'bias': np.random.randn(n_mcmc_samples, 3) * 0.2,
            'noise_scale': np.abs(np.random.randn(n_mcmc_samples)) * 0.1 + 0.05
        },
        'trace': {
            'is_accepted': np.random.choice([True, False], n_mcmc_samples, p=[0.7, 0.3]),
            'step_size': np.random.uniform(0.005, 0.02, n_mcmc_samples)
        },
        'diagnostics': {
            'acceptance_rate': 0.72,
            'num_samples': n_mcmc_samples
        },
        'posterior_predictions': {
            'samples': np.random.randn(n_mcmc_samples, n_samples, n_outputs) * 0.3
        }
    }
    
    return X, predictions_dict, true_values, mcmc_results

def test_bayesian_visualizations():
    """Test all Bayesian model visualization methods"""
    print("Testing Bayesian model seaborn visualizations...")
    
    # Create test data
    X, predictions_dict, true_values, mcmc_results = create_test_data()
      # Initialize the Bayesian model
    model = SimpleBayesianFlareAnalyzer(
        sequence_length=128,
        n_features=2,
        max_flares=3
    )
    
    # Create output directory
    output_dir = 'enhanced_output/bayesian_seaborn_tests'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTesting uncertainty analysis plot...")
    try:
        fig1 = model.plot_uncertainty_analysis(X, predictions_dict, true_values)
        fig1.savefig(f'{output_dir}/bayesian_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("✓ Uncertainty analysis plot created successfully")
    except Exception as e:
        print(f"✗ Error in uncertainty analysis plot: {e}")
    
    print(f"\nTesting MCMC diagnostics plot...")
    try:
        fig2 = model.plot_mcmc_diagnostics(mcmc_results, f'{output_dir}/bayesian_mcmc_diagnostics.png')
        plt.close(fig2)
        print("✓ MCMC diagnostics plot created successfully")
    except Exception as e:
        print(f"✗ Error in MCMC diagnostics plot: {e}")
    
    print(f"\nTesting uncertainty evolution plot...")
    try:
        fig3 = model.plot_uncertainty_evolution(X, predictions_dict, f'{output_dir}/bayesian_uncertainty_evolution.png')
        plt.close(fig3)
        print("✓ Uncertainty evolution plot created successfully")
    except Exception as e:
        print(f"✗ Error in uncertainty evolution plot: {e}")
    
    print(f"\nAll visualization tests completed!")
    print(f"Check the '{output_dir}' directory for generated plots.")

if __name__ == "__main__":
    # Set seaborn style globally
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    test_bayesian_visualizations()
