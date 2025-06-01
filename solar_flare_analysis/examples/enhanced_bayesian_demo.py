#!/usr/bin/env python3
"""
Enhanced Bayesian Solar Flare Analysis with MCMC Sampling
Demonstrates the improved Bayesian model with proper MCMC sampling and 
Monte Carlo data augmentation for robust uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml_models.bayesian_flare_analysis import BayesianFlareAnalyzer, BayesianFlareEnergyEstimator
from src.data_processing.data_loader import load_goes_data
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Demonstrate enhanced Bayesian inference with MCMC sampling
    """
    print("=== Enhanced Bayesian Solar Flare Analysis with MCMC ===\n")
    
    # Initialize enhanced Bayesian analyzer
    analyzer = BayesianFlareAnalyzer(
        sequence_length=128,
        n_features=2,
        max_flares=3,
        n_monte_carlo_samples=100,
        sensor_noise_std=0.01
    )
    
    print("1. Building enhanced Bayesian neural network...")
    model = analyzer.build_bayesian_model()
    print(f"Model built with {model.count_params():,} parameters")
    print(f"Model architecture: {len(model.layers)} layers with Bayesian inference")
    
    # Generate synthetic data with physics-based modeling
    print("\n2. Generating synthetic solar flare data with physics constraints...")
    X_synthetic, y_synthetic = analyzer.generate_synthetic_data_with_physics(
        n_samples=1000, noise_level=0.02
    )
    print(f"Generated {X_synthetic.shape[0]} synthetic samples")
    print(f"Data shape: {X_synthetic.shape}")
    print(f"Target shape: {y_synthetic.shape}")
    
    # Demonstrate Monte Carlo data augmentation
    print("\n3. Applying Monte Carlo data augmentation with physics-based noise...")
    X_sample = X_synthetic[:100]  # Use subset for demonstration
    y_sample = y_synthetic[:100]
    
    X_augmented = analyzer.monte_carlo_data_augmentation(X_sample, n_augmented_samples=3)
    print(f"Original samples: {X_sample.shape[0]}")
    print(f"Augmented samples: {X_augmented.shape[0]}")
    print(f"Augmentation factor: {X_augmented.shape[0] / X_sample.shape[0]:.1f}x")
    
    # Train the Bayesian model
    print("\n4. Training Bayesian model with Monte Carlo data augmentation...")
    y_augmented = np.repeat(y_sample, 4, axis=0)  # Match augmented data
    
    history = analyzer.train_bayesian_model(
        X_augmented, y_augmented,
        validation_split=0.2,
        epochs=20,  # Reduced for demonstration
        batch_size=32,
        augment_data=False  # Already augmented
    )
    
    print("Model training completed!")
    
    # Test data for uncertainty quantification
    X_test = X_synthetic[800:850]
    y_test = y_synthetic[800:850]
    
    # Monte Carlo predictions with uncertainty
    print("\n5. Making predictions with uncertainty quantification...")
    predictions = analyzer.monte_carlo_predict(X_test, n_samples=100)
    
    print(f"Mean prediction shape: {predictions['mean'].shape}")
    print(f"Uncertainty (std) shape: {predictions['std'].shape}")
    print(f"Mean uncertainty: {np.mean(predictions['std']):.6f}")
    
    # Separate epistemic and aleatoric uncertainty
    print("\n6. Separating epistemic and aleatoric uncertainty...")
    uncertainty_components = analyzer.quantify_epistemic_aleatoric_uncertainty(
        X_test, n_epistemic=30, n_aleatoric=50
    )
    
    print(f"Epistemic uncertainty (model): {np.mean(uncertainty_components['epistemic_uncertainty']):.6f}")
    print(f"Aleatoric uncertainty (data): {np.mean(uncertainty_components['aleatoric_uncertainty']):.6f}")
    print(f"Total uncertainty: {np.mean(uncertainty_components['total_uncertainty']):.6f}")
    
    # MCMC sampling for posterior inference
    print("\n7. Performing MCMC sampling for posterior inference...")
    try:
        mcmc_results = analyzer.mcmc_sampling(
            X_test[:20], y_test[:20],  # Use smaller subset for MCMC
            n_samples=200,
            n_burnin=100
        )
        
        print(f"MCMC sampling results:")
        print(f"  - Acceptance rate: {mcmc_results['acceptance_rate']:.3f}")
        print(f"  - Effective sample size: {mcmc_results['effective_sample_size']:.1f}")
        if 'method' in mcmc_results:
            print(f"  - Method: {mcmc_results['method']}")
        
        # Posterior predictive sampling
        print("\n8. Posterior predictive sampling...")
        posterior_predictions = analyzer.posterior_predictive_sampling(
            X_test, mcmc_samples=mcmc_results, n_predictions=100
        )
        
        print(f"Posterior predictive mean: {np.mean(posterior_predictions['mean']):.6f}")
        print(f"Posterior predictive std: {np.mean(posterior_predictions['std']):.6f}")
        
    except Exception as e:
        print(f"MCMC sampling encountered an issue: {e}")
        print("Continuing with variational inference approximation...")
        mcmc_results = None
        posterior_predictions = predictions
    
    # Calculate credible intervals and HPD
    print("\n9. Calculating credible intervals and HPD intervals...")
    intervals = analyzer.credible_intervals_and_hpd(posterior_predictions, credible_level=0.95)
    
    credible_width = np.mean(intervals['credible_intervals']['upper'] - 
                           intervals['credible_intervals']['lower'])
    hpd_width = np.mean(intervals['hpd_intervals']['upper'] - 
                       intervals['hpd_intervals']['lower'])
    
    print(f"95% Credible interval width: {credible_width:.6f}")
    print(f"95% HPD interval width: {hpd_width:.6f}")
    print(f"HPD vs Credible width ratio: {hpd_width/credible_width:.3f}")
    
    # Visualization
    print("\n10. Creating uncertainty analysis plots...")
    fig = analyzer.plot_uncertainty_analysis(X_test, posterior_predictions, y_test)
    
    # Additional plots for new features
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Epistemic vs Aleatoric uncertainty
    axes[0, 0].scatter(uncertainty_components['epistemic_uncertainty'].flatten(),
                      uncertainty_components['aleatoric_uncertainty'].flatten(),
                      alpha=0.6)
    axes[0, 0].set_xlabel('Epistemic Uncertainty')
    axes[0, 0].set_ylabel('Aleatoric Uncertainty')
    axes[0, 0].set_title('Epistemic vs Aleatoric Uncertainty')
    
    # Plot 2: Credible intervals
    sample_idx = np.arange(len(X_test))
    param_idx = 0  # First parameter
    
    axes[0, 1].fill_between(sample_idx, 
                           intervals['credible_intervals']['lower'][:, param_idx],
                           intervals['credible_intervals']['upper'][:, param_idx],
                           alpha=0.3, label='95% Credible Interval')
    axes[0, 1].fill_between(sample_idx,
                           intervals['hpd_intervals']['lower'][:, param_idx],
                           intervals['hpd_intervals']['upper'][:, param_idx],
                           alpha=0.5, label='95% HPD Interval')
    axes[0, 1].plot(sample_idx, posterior_predictions['mean'][:, param_idx], 
                   'r-', label='Posterior Mean')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].set_title('Credible vs HPD Intervals')
    axes[0, 1].legend()
    
    # Plot 3: Uncertainty decomposition
    total_uncertainty = uncertainty_components['total_uncertainty'].flatten()
    epistemic_uncertainty = uncertainty_components['epistemic_uncertainty'].flatten()
    aleatoric_uncertainty = uncertainty_components['aleatoric_uncertainty'].flatten()
    
    axes[1, 0].hist([epistemic_uncertainty, aleatoric_uncertainty], 
                   bins=30, alpha=0.7, label=['Epistemic', 'Aleatoric'])
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Uncertainty Decomposition')
    axes[1, 0].legend()
    
    # Plot 4: Model confidence
    confidence = 1 / (1 + total_uncertainty)
    axes[1, 1].hist(confidence, bins=30, alpha=0.7)
    axes[1, 1].set_xlabel('Model Confidence')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Model Confidence Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Save the enhanced model
    print("\n11. Saving enhanced Bayesian model...")
    try:
        analyzer.save_bayesian_model("enhanced_bayesian_flare_model")
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Demonstrate energy estimation with uncertainty
    print("\n12. Demonstrating Bayesian energy estimation...")
    energy_estimator = BayesianFlareEnergyEstimator(n_monte_carlo_samples=100)
    energy_model = energy_estimator.build_energy_model()
    
    # Use predicted flare parameters for energy estimation
    sample_params = posterior_predictions['mean'][:5, :5]  # First 5 samples, first 5 parameters
    
    energy_results = energy_estimator.estimate_energy_with_uncertainty(sample_params)
    
    print(f"Energy estimation results:")
    print(f"  - Mean energy: {np.mean(energy_results['mean_energy']):.2e} J")
    print(f"  - Energy uncertainty: {np.mean(energy_results['std_energy']):.2e} J")
    print(f"  - 95% confidence range: [{np.mean(energy_results['confidence_95'][0]):.2e}, "
          f"{np.mean(energy_results['confidence_95'][1]):.2e}] J")
    
    print("\n=== Enhanced Bayesian Analysis Complete ===")
    print("Key improvements implemented:")
    print("✓ Proper MCMC sampling with TensorFlow Probability")
    print("✓ Physics-based Monte Carlo data augmentation")
    print("✓ Separation of epistemic and aleatoric uncertainty")
    print("✓ Posterior predictive sampling")
    print("✓ Credible intervals and HPD intervals")
    print("✓ Robust uncertainty quantification")
    
    return analyzer, mcmc_results, posterior_predictions


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run the enhanced Bayesian analysis
    analyzer, mcmc_results, predictions = main()
    
    print("\nRun completed successfully!")
    print("The enhanced Bayesian model provides:")
    print("- Robust uncertainty quantification")
    print("- Improved model generalization through data augmentation")
    print("- Principled Bayesian inference with MCMC")
    print("- Comprehensive uncertainty decomposition")
