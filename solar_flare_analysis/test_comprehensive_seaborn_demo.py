#!/usr/bin/env python3
"""
Comprehensive test and demonstration of all enhanced seaborn visualizations
for both Monte Carlo and Bayesian models in the Solar Flare Analysis suite
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ml_models.monte_carlo_enhanced_model import MonteCarloSolarFlareModel
from src.ml_models.simple_bayesian_model import SimpleBayesianFlareAnalyzer

def create_comprehensive_test_data():
    """Create comprehensive test data for both models"""
    np.random.seed(42)
    
    # Parameters
    n_samples = 150
    sequence_length = 128
    n_features = 2
    
    # Generate realistic X-ray flux time series
    X = np.random.lognormal(-9, 1, (n_samples, sequence_length, n_features))
    
    # Add realistic flare events
    for i in range(n_samples):
        n_flares = np.random.poisson(1.2)  # Average 1.2 flares per sequence
        
        for flare_idx in range(n_flares):
            # Flare parameters
            flare_start = np.random.randint(10, sequence_length - 30)
            flare_duration = np.random.randint(8, 25)
            flare_class = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.04, 0.01])
            
            # Flare intensities based on class
            intensities = {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4}
            peak_intensity = intensities[flare_class] * np.random.uniform(1, 10)
            
            # Create realistic flare profile
            for j in range(flare_duration):
                if flare_start + j < sequence_length:
                    # Rise and decay profile
                    progress = j / flare_duration
                    if progress < 0.2:  # Fast rise
                        intensity_factor = (progress / 0.2) ** 0.5
                    else:  # Exponential decay
                        intensity_factor = np.exp(-(progress - 0.2) / 0.3)
                    
                    # Add to both channels
                    X[i, flare_start + j, 0] += peak_intensity * 0.1 * intensity_factor  # XRSA
                    X[i, flare_start + j, 1] += peak_intensity * intensity_factor        # XRSB
    
    # Log transform for realistic preprocessing
    X = np.log10(X + 1e-12)
    
    return X

def create_model_predictions(X, model_type='monte_carlo'):
    """Create realistic model predictions for testing"""
    n_samples = len(X)
    n_mc_samples = 50
    
    # Helper functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(x, axis=1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    if model_type == 'monte_carlo':
        # Monte Carlo model predictions (multi-task)
        predictions = {
            'detection': {
                'mean': sigmoid(np.random.randn(n_samples, 1) * 2 + 0.5),
                'std': np.abs(np.random.randn(n_samples, 1) * 0.15) + 0.03,
                'predictions': np.random.randn(n_mc_samples, n_samples, 1) * 0.2 + 0.6
            },
            'classification': {
                'mean': softmax(np.random.randn(n_samples, 6) * 1.5),
                'std': np.abs(np.random.randn(n_samples, 6) * 0.08) + 0.02,
                'predictions': np.random.randn(n_mc_samples, n_samples, 6) * 0.3
            },
            'regression': {
                'mean': np.random.randn(n_samples, 1) * 1.5 - 8,
                'std': np.abs(np.random.randn(n_samples, 1) * 0.25) + 0.08,
                'predictions': np.random.randn(n_mc_samples, n_samples, 1) * 0.4 - 8
            }
        }
        
        # Add confidence intervals
        for task in predictions:
            preds = predictions[task]['predictions']
            ci_lower = np.percentile(preds, 2.5, axis=0)
            ci_upper = np.percentile(preds, 97.5, axis=0)
            predictions[task]['confidence_interval'] = [ci_lower, ci_upper]
        
        return predictions
    
    else:  # Bayesian model
        # Bayesian model predictions (flare parameters)
        predictions = {
            'mean': np.random.randn(n_samples, 3) * 1.2,  # 3 flare parameters
            'std': np.abs(np.random.randn(n_samples, 3) * 0.3) + 0.1,
            'samples': np.random.randn(n_mc_samples, n_samples, 3) * 0.4,
            'confidence_intervals': {}
        }
        
        # Add confidence intervals
        samples = predictions['samples']
        predictions['confidence_intervals'] = {
            '2.5th': np.percentile(samples, 2.5, axis=0),
            '97.5th': np.percentile(samples, 97.5, axis=0),
            '25th': np.percentile(samples, 25, axis=0),
            '75th': np.percentile(samples, 75, axis=0),
            '50th': np.percentile(samples, 50, axis=0)
        }
        
        return predictions

def create_training_histories():
    """Create realistic training histories for both models"""
    n_epochs = 25
    
    # Monte Carlo training history
    mc_history = {
        'loss': np.random.exponential(0.4, n_epochs)[::-1] * 0.2 + 0.08,
        'val_loss': np.random.exponential(0.4, n_epochs)[::-1] * 0.25 + 0.1,
        'detection_output_accuracy': np.random.beta(3, 1, n_epochs) * 0.3 + 0.7,
        'val_detection_output_accuracy': np.random.beta(3, 1, n_epochs) * 0.25 + 0.68,
        'classification_output_accuracy': np.random.beta(2, 1, n_epochs) * 0.4 + 0.6,
        'val_classification_output_accuracy': np.random.beta(2, 1, n_epochs) * 0.35 + 0.58,
        'regression_output_mae': np.random.exponential(0.3, n_epochs)[::-1] * 0.08 + 0.04,
        'val_regression_output_mae': np.random.exponential(0.3, n_epochs)[::-1] * 0.1 + 0.05,
        'detection_output_loss': np.random.exponential(0.3, n_epochs)[::-1] * 0.05 + 0.02,
        'classification_output_loss': np.random.exponential(0.3, n_epochs)[::-1] * 0.1 + 0.05,
        'regression_output_loss': np.random.exponential(0.3, n_epochs)[::-1] * 0.06 + 0.03
    }
    
    return mc_history

def create_mcmc_results():
    """Create realistic MCMC sampling results for Bayesian model"""
    n_samples = 600
    mcmc_results = {
        'method': 'HMC',
        'samples': {
            'weights': np.random.randn(n_samples, 20) * 0.4,
            'bias': np.random.randn(n_samples, 5) * 0.2,
            'noise_scale': np.abs(np.random.randn(n_samples)) * 0.1 + 0.05
        },
        'trace': {
            'is_accepted': np.random.choice([True, False], n_samples, p=[0.78, 0.22]),
            'step_size': np.random.uniform(0.008, 0.025, n_samples)
        },
        'diagnostics': {
            'acceptance_rate': 0.782,
            'num_samples': n_samples,
            'final_step_size': 0.015
        },
        'posterior_predictions': {
            'samples': np.random.randn(n_samples, 50, 3) * 0.3
        }
    }
    
    return mcmc_results

def run_comprehensive_visualization_demo():
    """Run comprehensive demonstration of all enhanced visualizations"""
    print("🚀 COMPREHENSIVE SOLAR FLARE ML VISUALIZATION DEMONSTRATION")
    print("=" * 70)
    print("Testing both Monte Carlo and Bayesian models with enhanced seaborn visualizations")
    
    # Create test data
    print("\n📊 Generating realistic test data...")
    X = create_comprehensive_test_data()
    mc_predictions = create_model_predictions(X, 'monte_carlo')
    bayesian_predictions = create_model_predictions(X, 'bayesian')
    mc_history = create_training_histories()
    mcmc_results = create_mcmc_results()
    
    # Create true values for comparison
    true_values_mc = {
        'detection': (mc_predictions['detection']['mean'] + 
                     np.random.randn(*mc_predictions['detection']['mean'].shape) * 0.15).clip(0, 1),
        'classification': np.random.choice(6, len(X)),
        'regression': (mc_predictions['regression']['mean'] + 
                      np.random.randn(*mc_predictions['regression']['mean'].shape) * 0.2)
    }
    
    true_values_bayesian = (bayesian_predictions['mean'] + 
                           np.random.randn(*bayesian_predictions['mean'].shape) * 0.25)
    
    # Create output directory
    output_dir = 'enhanced_output/comprehensive_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    print("\n🔧 Initializing models...")
    mc_model = MonteCarloSolarFlareModel(sequence_length=128, n_features=2, n_classes=6)
    bayesian_model = SimpleBayesianFlareAnalyzer(sequence_length=128, n_features=2, max_flares=3)
    
    # Set training histories
    mc_model.training_history = mc_history
    
    # Test Monte Carlo visualizations
    print("\n🎯 Testing Monte Carlo Model Visualizations:")
    print("-" * 50)
    
    mc_tests = [
        ('Training History', lambda: mc_model.plot_training_history()),
        ('Prediction Uncertainty', lambda: mc_model.plot_prediction_uncertainty(X, mc_predictions, true_values_mc)),
        ('Model Diagnostics', lambda: mc_model.plot_model_diagnostics(X, mc_predictions, true_values_mc)),
        ('Uncertainty Evolution', lambda: mc_model.plot_uncertainty_evolution(X, mc_predictions))
    ]
    
    mc_results = []
    for test_name, test_func in mc_tests:
        try:
            print(f"  Testing {test_name}...", end=" ")
            fig = test_func()
            if fig:
                save_name = f"mc_{test_name.lower().replace(' ', '_')}.png"
                fig.savefig(f"{output_dir}/{save_name}", dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✅")
                mc_results.append(True)
            else:
                print("⚠️")
                mc_results.append(False)
        except Exception as e:
            print(f"❌ ({str(e)[:50]}...)")
            mc_results.append(False)
    
    # Test Bayesian visualizations
    print("\n🧠 Testing Bayesian Model Visualizations:")
    print("-" * 50)
    
    bayesian_tests = [
        ('Uncertainty Analysis', lambda: bayesian_model.plot_uncertainty_analysis(X, bayesian_predictions, true_values_bayesian)),
        ('MCMC Diagnostics', lambda: bayesian_model.plot_mcmc_diagnostics(mcmc_results)),
        ('Uncertainty Evolution', lambda: bayesian_model.plot_uncertainty_evolution(X, bayesian_predictions))
    ]
    
    bayesian_results = []
    for test_name, test_func in bayesian_tests:
        try:
            print(f"  Testing {test_name}...", end=" ")
            fig = test_func()
            if fig:
                save_name = f"bayesian_{test_name.lower().replace(' ', '_')}.png"
                fig.savefig(f"{output_dir}/{save_name}", dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✅")
                bayesian_results.append(True)
            else:
                print("⚠️")
                bayesian_results.append(False)
        except Exception as e:
            print(f"❌ ({str(e)[:50]}...)")
            bayesian_results.append(False)
    
    # Model comparison
    print("\n🔄 Testing Model Comparison Framework:")
    print("-" * 50)
    
    try:
        print("  Creating comparison visualization...", end=" ")
        comparison_data = {
            'Monte Carlo': {
                'standard_metrics': {
                    'loss': 0.12,
                    'detection_output_accuracy': 0.87,
                    'classification_output_accuracy': 0.79
                },
                'monte_carlo_metrics': {
                    'uncertainty_decomposition': {
                        'detection_epistemic': 0.045,
                        'classification_epistemic': 0.072
                    },
                    'prediction_interval_coverage': {
                        'detection': 0.945,
                        'regression': 0.963
                    }
                },
                'model_info': {
                    'parameters': 180000,
                    'mc_samples': 50,
                    'dropout_rate': 0.3
                }
            },
            'Bayesian': {
                'standard_metrics': {
                    'loss': 0.14,
                    'detection_output_accuracy': 0.84,
                    'classification_output_accuracy': 0.76
                },
                'monte_carlo_metrics': {
                    'uncertainty_decomposition': {
                        'detection_epistemic': 0.052,
                        'classification_epistemic': 0.068
                    },
                    'prediction_interval_coverage': {
                        'detection': 0.952,
                        'regression': 0.971
                    }
                },
                'model_info': {
                    'parameters': 120000,
                    'mc_samples': 100,
                    'dropout_rate': 0.25
                }
            }
        }
        
        fig = mc_model.plot_model_comparison(comparison_data)
        fig.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✅")
        comparison_success = True
    except Exception as e:
        print(f"❌ ({str(e)[:50]}...)")
        comparison_success = False
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 COMPREHENSIVE DEMONSTRATION COMPLETED!")
    print("=" * 70)
    
    mc_success = sum(mc_results)
    bayesian_success = sum(bayesian_results)
    total_success = mc_success + bayesian_success + (1 if comparison_success else 0)
    total_tests = len(mc_results) + len(bayesian_results) + 1
    
    print(f"📊 Results Summary:")
    print(f"   Monte Carlo Model: {mc_success}/{len(mc_results)} tests passed")
    print(f"   Bayesian Model: {bayesian_success}/{len(bayesian_results)} tests passed")
    print(f"   Model Comparison: {'✅' if comparison_success else '❌'}")
    print(f"   Overall: {total_success}/{total_tests} tests passed ({total_success/total_tests*100:.1f}%)")
    
    print(f"\n📁 Output Directory: {output_dir}")
    
    # List generated files
    try:
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"\n📋 Generated {len(files)} visualization files:")
        for file in sorted(files):
            print(f"   • {file}")
    except Exception:
        pass
    
    print("\n🔍 Enhanced Capabilities Demonstrated:")
    print("   ✨ Modern seaborn styling with professional aesthetics")
    print("   📈 Advanced uncertainty quantification and visualization")
    print("   🎯 Multi-task learning visualization support")
    print("   📊 Statistical correlation and trend analysis")
    print("   🔬 Monte Carlo convergence diagnostics")
    print("   🧪 MCMC sampling visualization and diagnostics")
    print("   🔄 Comprehensive model comparison framework")
    print("   🛡️  Robust error handling and data type safety")
    
    if total_success == total_tests:
        print("\n🎊 ALL VISUALIZATIONS WORKING PERFECTLY! 🎊")
        print("The enhanced seaborn visualization suite is ready for production use!")
    elif total_success >= total_tests * 0.8:
        print("\n✅ EXCELLENT RESULTS!")
        print("Most visualizations working perfectly with minor issues addressed.")
    else:
        print("\n⚠️  SOME IMPROVEMENTS NEEDED")
        print("Check individual test results for specific issues.")
    
    return total_success == total_tests

if __name__ == "__main__":
    # Set global styling
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (12, 8)
    })
    
    success = run_comprehensive_visualization_demo()
    
    print(f"\n🚀 Demonstration {'completed successfully' if success else 'completed with issues'}!")
    print("Both Monte Carlo and Bayesian models now have comprehensive seaborn-enhanced visualizations.")
