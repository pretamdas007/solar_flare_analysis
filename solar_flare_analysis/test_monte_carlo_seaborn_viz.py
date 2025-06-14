#!/usr/bin/env python3
"""
Test script for enhanced Monte Carlo model's seaborn visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ml_models.monte_carlo_enhanced_model import MonteCarloSolarFlareModel

def create_test_data():
    """Create synthetic test data for Monte Carlo model testing"""
    np.random.seed(42)
    
    # Create synthetic input sequences
    n_samples = 100
    sequence_length = 128
    n_features = 2
    
    X = np.random.randn(n_samples, sequence_length, n_features) * 0.3
    
    # Add realistic flare patterns
    for i in range(n_samples):
        if np.random.random() < 0.4:  # 40% chance of flare
            flare_start = np.random.randint(10, 90)
            flare_duration = np.random.randint(8, 25)
            flare_magnitude = np.random.exponential(2.0)
            
            # Add flare pattern
            for j in range(flare_duration):
                if flare_start + j < sequence_length:
                    decay_factor = np.exp(-j/10)
                    X[i, flare_start + j, 0] += flare_magnitude * decay_factor * 0.1  # XRSA
                    X[i, flare_start + j, 1] += flare_magnitude * decay_factor        # XRSB
    
    # Log transform (similar to real preprocessing)
    X = np.log10(np.abs(X) + 1e-10)
      # Create synthetic predictions (simulate model output)
    n_mc_samples = 50
    
    # Helper function for sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    # Helper function for softmax
    def softmax(x, axis=1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    predictions_dict = {
        'detection': {
            'mean': sigmoid(np.random.randn(n_samples, 1) * 2),
            'std': np.abs(np.random.randn(n_samples, 1) * 0.2) + 0.05,
            'confidence_interval': None,
            'predictions': np.random.randn(n_mc_samples, n_samples, 1) * 0.3 + 0.5
        },
        'classification': {
            'mean': softmax(np.random.randn(n_samples, 6)),
            'std': np.abs(np.random.randn(n_samples, 6) * 0.1) + 0.02,
            'confidence_interval': None,
            'predictions': np.random.randn(n_mc_samples, n_samples, 6) * 0.2
        },
        'regression': {
            'mean': np.random.randn(n_samples, 1) * 2 - 8,  # Log flux values
            'std': np.abs(np.random.randn(n_samples, 1) * 0.3) + 0.1,
            'confidence_interval': None,
            'predictions': np.random.randn(n_mc_samples, n_samples, 1) * 0.5 - 8
        }
    }
    
    # Add confidence intervals
    for task in ['detection', 'classification', 'regression']:
        preds = predictions_dict[task]['predictions']
        ci_lower = np.percentile(preds, 2.5, axis=0)
        ci_upper = np.percentile(preds, 97.5, axis=0)
        predictions_dict[task]['confidence_interval'] = [ci_lower, ci_upper]
    
    # Create synthetic true values
    true_values = {
        'detection': (predictions_dict['detection']['mean'] + 
                     np.random.randn(n_samples, 1) * 0.2).clip(0, 1),
        'classification': np.random.choice(6, n_samples),
        'regression': (predictions_dict['regression']['mean'] + 
                      np.random.randn(n_samples, 1) * 0.3)
    }
    
    # Create synthetic training history
    n_epochs = 30
    training_history = {
        'loss': np.random.exponential(0.5, n_epochs)[::-1] * 0.2 + 0.1,
        'val_loss': np.random.exponential(0.5, n_epochs)[::-1] * 0.25 + 0.12,
        'detection_output_accuracy': np.random.beta(2, 1, n_epochs) * 0.3 + 0.7,
        'val_detection_output_accuracy': np.random.beta(2, 1, n_epochs) * 0.25 + 0.68,
        'classification_output_accuracy': np.random.beta(2, 1, n_epochs) * 0.4 + 0.6,
        'val_classification_output_accuracy': np.random.beta(2, 1, n_epochs) * 0.35 + 0.58,
        'regression_output_mae': np.random.exponential(0.3, n_epochs)[::-1] * 0.1 + 0.05,
        'val_regression_output_mae': np.random.exponential(0.3, n_epochs)[::-1] * 0.12 + 0.06,
        'detection_output_loss': np.random.exponential(0.4, n_epochs)[::-1] * 0.1 + 0.05,
        'classification_output_loss': np.random.exponential(0.4, n_epochs)[::-1] * 0.15 + 0.08,
        'regression_output_loss': np.random.exponential(0.4, n_epochs)[::-1] * 0.08 + 0.04
    }
    
    return X, predictions_dict, true_values, training_history

def test_monte_carlo_visualizations():
    """Test all Monte Carlo model visualization methods"""
    print("🚀 Testing Enhanced Monte Carlo Model Seaborn Visualizations")
    print("=" * 65)
    
    # Create test data
    X, predictions_dict, true_values, training_history = create_test_data()
    
    # Initialize the Monte Carlo model
    model = MonteCarloSolarFlareModel(
        sequence_length=128,
        n_features=2,
        n_classes=6,
        mc_samples=50
    )
    
    # Set the training history for testing
    model.training_history = training_history
    
    # Create output directory
    output_dir = 'enhanced_output/monte_carlo_seaborn_tests'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test all visualization methods
    visualization_tests = [
        {
            'name': 'Training History',
            'method': lambda: model.plot_training_history(training_history),
            'save_name': 'mc_training_history.png'
        },
        {
            'name': 'Prediction Uncertainty',
            'method': lambda: model.plot_prediction_uncertainty(X, predictions_dict, true_values),
            'save_name': 'mc_prediction_uncertainty.png'
        },
        {
            'name': 'Model Diagnostics',
            'method': lambda: model.plot_model_diagnostics(X, predictions_dict, true_values),
            'save_name': 'mc_model_diagnostics.png'
        },
        {
            'name': 'Uncertainty Evolution',
            'method': lambda: model.plot_uncertainty_evolution(X, predictions_dict),
            'save_name': 'mc_uncertainty_evolution.png'
        }
    ]
    
    results = []
    
    print("\n📊 Testing Individual Visualization Methods:")
    print("-" * 50)
    
    for test in visualization_tests:
        try:
            print(f"Testing {test['name']}...", end=" ")
            fig = test['method']()
            
            if fig is not None:
                save_path = f"{output_dir}/{test['save_name']}"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✅ SUCCESS")
                results.append(True)
            else:
                print("⚠️  RETURNED NONE")
                results.append(False)
                
        except Exception as e:
            print(f"❌ FAILED - {e}")
            results.append(False)
    
    # Test model comparison visualization
    print(f"\nTesting Model Comparison...", end=" ")
    try:
        comparison_data = {
            'Monte Carlo v1': {
                'standard_metrics': {
                    'loss': 0.15,
                    'detection_output_accuracy': 0.85,
                    'classification_output_accuracy': 0.78,
                    'regression_output_mae': 0.12
                },
                'monte_carlo_metrics': {
                    'uncertainty_decomposition': {
                        'detection_epistemic': 0.05,
                        'classification_epistemic': 0.08,
                        'regression_epistemic': 0.15
                    },
                    'prediction_interval_coverage': {
                        'detection': 0.94,
                        'regression': 0.96
                    }
                },
                'model_info': {
                    'parameters': 150000,
                    'mc_samples': 50,
                    'dropout_rate': 0.3
                }
            },
            'Monte Carlo v2': {
                'standard_metrics': {
                    'loss': 0.12,
                    'detection_output_accuracy': 0.88,
                    'classification_output_accuracy': 0.82,
                    'regression_output_mae': 0.10
                },
                'monte_carlo_metrics': {
                    'uncertainty_decomposition': {
                        'detection_epistemic': 0.04,
                        'classification_epistemic': 0.06,
                        'regression_epistemic': 0.12
                    },
                    'prediction_interval_coverage': {
                        'detection': 0.95,
                        'regression': 0.97
                    }
                },
                'model_info': {
                    'parameters': 200000,
                    'mc_samples': 100,
                    'dropout_rate': 0.4
                }
            }
        }
        
        fig = model.plot_model_comparison(comparison_data)
        fig.savefig(f"{output_dir}/mc_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✅ SUCCESS")
        results.append(True)
        
    except Exception as e:
        print(f"❌ FAILED - {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 65)
    print("🎉 MONTE CARLO VISUALIZATION TESTING COMPLETED!")
    print("=" * 65)
    
    successful_tests = sum(results)
    total_tests = len(results)
    
    print(f"📊 Results: {successful_tests}/{total_tests} tests passed")
    print(f"📁 Output directory: {output_dir}")
    
    if successful_tests == total_tests:
        print("🎊 All visualizations working perfectly!")
    elif successful_tests >= total_tests * 0.8:
        print("✅ Most visualizations working well!")
    else:
        print("⚠️  Some visualizations need attention")
    
    # List generated files
    try:
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.endswith('.png')]
        print(f"\n📋 Generated {len(png_files)} visualization files:")
        for file in sorted(png_files):
            print(f"   • {file}")
    except Exception as e:
        print(f"❌ Could not list output files: {e}")
    
    print("\n🔍 Enhanced Features Include:")
    print("   • Modern seaborn styling with professional aesthetics")
    print("   • Robust uncertainty quantification visualizations")
    print("   • Multi-task prediction analysis")
    print("   • Statistical correlation analysis")
    print("   • Monte Carlo convergence diagnostics")
    print("   • Comprehensive model comparison capabilities")
    
    return successful_tests == total_tests

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
    
    success = test_monte_carlo_visualizations()
    
    if success:
        print("\n🎉 All Monte Carlo visualizations are working perfectly! 🎉")
    else:
        print("\n⚠️  Some issues detected. Check the output above for details.")
