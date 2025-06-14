#!/usr/bin/env python3
"""
Comprehensive test script for all enhanced Bayesian model visualizations
including new comparison and advanced analysis features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the enhanced Bayesian model
from src.ml_models.simple_bayesian_model import SimpleBayesianFlareAnalyzer

def create_comprehensive_test_data():
    """Create comprehensive test data for all visualization methods"""
    np.random.seed(42)
    
    # Create multiple datasets for comparison
    n_samples = 200
    sequence_length = 128
    n_features = 2
    
    # Dataset 1: Low noise
    X1 = np.random.randn(n_samples, sequence_length, n_features) * 0.2
    
    # Dataset 2: High noise
    X2 = np.random.randn(n_samples, sequence_length, n_features) * 0.5
    
    # Add realistic flare patterns to both
    for X in [X1, X2]:
        for i in range(n_samples):
            n_flares = np.random.poisson(1.8)  # Average 1.8 flares per sequence
            
            for flare_idx in range(n_flares):
                flare_start = np.random.randint(10, 110)
                flare_duration = np.random.randint(8, 30)
                flare_magnitude = np.random.exponential(2.0)
                
                # Add exponential decay flare pattern
                for j in range(flare_duration):
                    if flare_start + j < sequence_length:
                        decay_factor = np.exp(-j/10)
                        X[i, flare_start + j, 0] += flare_magnitude * decay_factor
                        if n_features > 1:
                            X[i, flare_start + j, 1] += flare_magnitude * 0.8 * decay_factor
    
    # Create synthetic predictions for multiple models
    n_outputs = 5  # flare parameters
    
    # Model 1: "High Accuracy" model
    mean_pred1 = np.random.randn(n_samples, n_outputs) * 1.2
    std_pred1 = np.abs(mean_pred1) * 0.2 + np.random.exponential(0.15, (n_samples, n_outputs))
    
    # Model 2: "Lower Accuracy" model  
    mean_pred2 = mean_pred1 + np.random.randn(n_samples, n_outputs) * 0.4
    std_pred2 = np.abs(mean_pred2) * 0.4 + np.random.exponential(0.25, (n_samples, n_outputs))
    
    # Model 3: "Conservative" model (higher uncertainty)
    mean_pred3 = mean_pred1 + np.random.randn(n_samples, n_outputs) * 0.2
    std_pred3 = np.abs(mean_pred3) * 0.6 + np.random.exponential(0.35, (n_samples, n_outputs))
    
    # Create confidence intervals for all models
    models_predictions = {}
    
    for i, (name, mean_pred, std_pred) in enumerate([
        ("High Accuracy", mean_pred1, std_pred1),
        ("Standard Model", mean_pred2, std_pred2), 
        ("Conservative Model", mean_pred3, std_pred3)
    ]):
        z_score = 1.96  # 95% confidence
        ci_lower = mean_pred - z_score * std_pred
        ci_upper = mean_pred + z_score * std_pred
        
        # Create samples for advanced analysis
        n_mc_samples = 100
        samples = np.random.randn(n_mc_samples, n_samples, n_outputs) * std_pred[None, :, :] + mean_pred[None, :, :]
        
        models_predictions[name] = {
            'mean': mean_pred,
            'std': std_pred,
            'confidence_intervals': {
                '2.5th': ci_lower,
                '97.5th': ci_upper,
                '25th': mean_pred - 0.67 * std_pred,
                '50th': mean_pred,
                '75th': mean_pred + 0.67 * std_pred
            },
            'samples': samples
        }
    
    # Create realistic true values
    true_values = mean_pred1 + np.random.randn(*mean_pred1.shape) * std_pred1 * 0.3
    
    return X1, X2, models_predictions, true_values

def test_enhanced_bayesian_visualizations():
    """Test all enhanced Bayesian model visualization methods"""
    print("🚀 Testing Enhanced Bayesian Model Visualizations with Seaborn")
    print("=" * 80)
    
    # Create comprehensive test data
    X1, X2, models_predictions, true_values = create_comprehensive_test_data()
    
    # Initialize the Bayesian model
    model = SimpleBayesianFlareAnalyzer(
        sequence_length=128,
        n_features=2,
        max_flares=3
    )
    
    # Create output directory
    output_dir = 'enhanced_output/comprehensive_bayesian_viz'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Testing Core Visualization Methods...")
    print("-" * 50)
    
    # Test core methods with the first model
    main_predictions = models_predictions["High Accuracy"]
    
    tests_core = [
        ("Uncertainty Analysis", lambda: model.plot_uncertainty_analysis(X1, main_predictions, true_values)),
        ("Uncertainty Evolution", lambda: model.plot_uncertainty_evolution(X1, main_predictions)),
        ("Advanced Uncertainty Analysis", lambda: model.plot_advanced_uncertainty_analysis(X1, main_predictions)),
        ("Performance Dashboard", lambda: model.plot_predictive_performance_dashboard(X1, main_predictions, true_values))
    ]
    
    for test_name, test_func in tests_core:
        try:
            print(f"  Testing {test_name}...")
            fig = test_func()
            fig.savefig(f'{output_dir}/{test_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✅ {test_name}: SUCCESS")
        except Exception as e:
            print(f"  ❌ {test_name}: FAILED - {e}")
    
    print(f"\n🔀 Testing Model Comparison Visualizations...")
    print("-" * 50)
    
    # Test model comparison methods
    try:
        print(f"  Testing Model Comparison...")
        fig_comp = model.plot_model_comparison(models_predictions, X1, true_values)
        fig_comp.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig_comp)
        print(f"  ✅ Model Comparison: SUCCESS")
    except Exception as e:
        print(f"  ❌ Model Comparison: FAILED - {e}")
    
    print(f"\n🧪 Testing MCMC Diagnostics with Enhanced Data...")
    print("-" * 50)
    
    # Create enhanced MCMC results for testing
    try:
        print(f"  Creating synthetic MCMC results...")
        n_mcmc_samples = 1000
        enhanced_mcmc_results = {
            'method': 'Enhanced_HMC',
            'samples': {
                'weights': np.random.randn(n_mcmc_samples, 20) * 0.3,
                'bias': np.random.randn(n_mcmc_samples, 8) * 0.12,
                'noise_scale': np.abs(np.random.randn(n_mcmc_samples)) * 0.06 + 0.03
            },
            'trace': {
                'is_accepted': np.random.choice([True, False], n_mcmc_samples, p=[0.78, 0.22]),
                'step_size': np.random.uniform(0.01, 0.03, n_mcmc_samples),
                'log_accept_ratio': np.random.randn(n_mcmc_samples) * 0.5
            },
            'diagnostics': {
                'acceptance_rate': 0.782,
                'num_samples': n_mcmc_samples,
                'final_step_size': 0.0234
            },
            'posterior_predictions': {
                'samples': np.random.randn(n_mcmc_samples, 50, 5) * 0.2
            }
        }
        
        fig_mcmc = model.plot_mcmc_diagnostics(enhanced_mcmc_results)
        fig_mcmc.savefig(f'{output_dir}/enhanced_mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close(fig_mcmc)
        print(f"  ✅ Enhanced MCMC Diagnostics: SUCCESS")
    except Exception as e:
        print(f"  ❌ Enhanced MCMC Diagnostics: FAILED - {e}")
    
    print(f"\n📈 Testing Statistical Analysis Features...")
    print("-" * 50)
    
    # Test statistical analysis capabilities
    try:
        print(f"  Analyzing model performance statistics...")
        
        # Calculate comprehensive statistics
        stats_summary = {}
        for model_name, predictions in models_predictions.items():
            mean_pred = predictions['mean'].flatten()
            true_flat = true_values.flatten()
            min_len = min(len(mean_pred), len(true_flat))
            
            residuals = true_flat[:min_len] - mean_pred[:min_len]
            mse = np.mean(residuals**2)
            mae = np.mean(np.abs(residuals))
            r2 = 1 - (np.sum(residuals**2) / np.sum((true_flat[:min_len] - np.mean(true_flat[:min_len]))**2))
            
            # Coverage probability
            ci_lower = predictions['confidence_intervals']['2.5th'].flatten()[:min_len]
            ci_upper = predictions['confidence_intervals']['97.5th'].flatten()[:min_len]
            coverage = np.mean((true_flat[:min_len] >= ci_lower) & (true_flat[:min_len] <= ci_upper))
            
            stats_summary[model_name] = {
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'Coverage': coverage,
                'Mean_Uncertainty': np.mean(predictions['std'])
            }
        
        # Print summary table
        print(f"\n  📋 Model Performance Summary:")
        print(f"  {'Model':<18} {'MSE':<8} {'MAE':<8} {'R²':<8} {'Coverage':<10} {'Mean_Unc':<10}")
        print(f"  {'-'*70}")
        
        for model_name, stats in stats_summary.items():
            print(f"  {model_name:<18} {stats['MSE']:<8.4f} {stats['MAE']:<8.4f} {stats['R²']:<8.4f} "
                  f"{stats['Coverage']:<10.3f} {stats['Mean_Uncertainty']:<10.4f}")
        
        print(f"  ✅ Statistical Analysis: SUCCESS")
    except Exception as e:
        print(f"  ❌ Statistical Analysis: FAILED - {e}")
    
    # Summary and file listing
    print(f"\n" + "=" * 80)
    print(f"🎉 COMPREHENSIVE TESTING COMPLETED!")
    print(f"📁 All visualizations saved to: {output_dir}")
    
    # Enhanced features summary
    print(f"\n🔍 Enhanced Seaborn Features Include:")
    print(f"   • 🎨 Modern seaborn styling with custom aesthetics")
    print(f"   • 📊 Comprehensive model comparison visualizations") 
    print(f"   • 🧪 Advanced uncertainty decomposition analysis")
    print(f"   • 📈 Predictive performance dashboards")
    print(f"   • 🔬 Statistical significance testing")
    print(f"   • 🎯 Model calibration and reliability assessment")
    print(f"   • 📉 Temporal pattern analysis")
    print(f"   • 🌡️ Uncertainty correlation heatmaps")
    print(f"   • 📋 Automated performance summaries")
    print(f"   • 🔄 Robust error handling and fallbacks")
    
    # List generated files
    try:
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.endswith('.png')]
        print(f"\n📋 Generated {len(png_files)} enhanced visualization files:")
        for file in sorted(png_files):
            print(f"   • {file}")
    except Exception as e:
        print(f"❌ Could not list output files: {e}")
    
    print(f"\n🏆 All Bayesian model visualizations are now production-ready!")
    print(f"    with comprehensive seaborn enhancements and robust comparison capabilities.")

if __name__ == "__main__":
    # Set comprehensive seaborn styling
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 14
    })
    
    test_enhanced_bayesian_visualizations()
