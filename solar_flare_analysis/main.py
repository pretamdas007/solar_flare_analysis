"""
Main script for solar flare analysis.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.abspath('c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis'))

# Import project modules
from config import settings
from src.data_processing.data_loader import load_goes_data, preprocess_xrs_data, remove_background
from src.flare_detection.traditional_detection import (detect_flare_peaks, define_flare_bounds, 
                                                    detect_overlapping_flares)
from src.ml_models.flare_decomposition import FlareDecompositionModel, reconstruct_flares
from src.ml_models.bayesian_flare_analysis import BayesianFlareAnalyzer, BayesianFlareEnergyEstimator
from src.analysis.power_law import calculate_flare_energy, fit_power_law, compare_flare_populations
from src.visualization.plotting import (plot_xrs_time_series, plot_detected_flares, 
                                      plot_flare_decomposition, plot_power_law_comparison)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solar Flare Analysis')
    parser.add_argument('--data', type=str, help='Path to NetCDF data file',
                       default=None)
    parser.add_argument('--channel', type=str, choices=['A', 'B'], 
                       default='B', help='XRS channel to analyze')
    parser.add_argument('--train', action='store_true', 
                       help='Train the ML model with synthetic data')
    parser.add_argument('--train-bayesian', action='store_true',
                       help='Train the Bayesian ML model with Monte Carlo sampling')
    parser.add_argument('--model', type=str, 
                       help='Path to save/load model',
                       default=os.path.join(settings.MODEL_DIR, 'flare_decomposition_model'))
    parser.add_argument('--output', type=str,
                       help='Output directory for results',
                       default=settings.OUTPUT_DIR)
    return parser.parse_args()


def train_model(args):
    """Train the flare decomposition model with synthetic data."""
    print("\n=== Training Flare Decomposition Model ===")
    
    # Create model
    model = FlareDecompositionModel(
        sequence_length=settings.ML_PARAMS['sequence_length'],
        n_features=settings.ML_PARAMS['n_features'],
        max_flares=settings.ML_PARAMS['max_flares'],
        dropout_rate=settings.ML_PARAMS['dropout_rate']
    )
    
    # Build model architecture
    model.build_model()
    model.model.summary()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    X_train, y_train = model.generate_synthetic_data(n_samples=5000, noise_level=0.05)
    X_val, y_val = model.generate_synthetic_data(n_samples=1000, noise_level=0.05)
    
    # Train model
    print("Training model...")
    history = model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=settings.ML_PARAMS['epochs'],
        batch_size=settings.ML_PARAMS['batch_size'],
        save_path=args.model
    )
    
    # Plot training history
    history_fig = model.plot_training_history()
    history_fig.savefig(os.path.join(args.output, 'training_history.png'), dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Evaluate on test data
    print("Evaluating model...")
    X_test, y_test = model.generate_synthetic_data(n_samples=500, noise_level=0.08)
    eval_results = model.evaluate(X_test, y_test)
    print(f"Evaluation results: {eval_results}")
    
    # Save model
    model.save_model(args.model)
    print(f"Model saved to {args.model}")
    
    return model


def analyze_flares(args):
    """Analyze solar flares in GOES XRS data."""
    # Step 1: Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    if args.data:
        data_path = args.data
    else:
        # Use sample data if available, or exit if not
        sample_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.nc')]
        if sample_files:
            data_path = os.path.join(settings.DATA_DIR, sample_files[0])
            print(f"Using sample data file: {data_path}")
        else:
            print("No data file provided and no sample files found.")
            print(f"Please place GOES XRS .nc files in {settings.DATA_DIR} or specify --data")
            return
    
    # Load data
    data = load_goes_data(data_path)
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Preprocess data
    print(f"Preprocessing {args.channel} channel data...")
    df = preprocess_xrs_data(data, channel=args.channel, remove_bad_data=True, interpolate_gaps=True)
    
    # Plot raw time series
    fig = plot_xrs_time_series(df, f'xrs{args.channel.lower()}', 
                              title=f'GOES XRS {args.channel} Raw Data', log_scale=True)
    fig.savefig(os.path.join(args.output, f'raw_timeseries_{args.channel}.png'), 
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 2: Detect flares with traditional method
    print("\n=== Detecting Flares with Traditional Method ===")
    flux_col = f'xrs{args.channel.lower()}'
    
    # Detect peaks
    print("Detecting peaks...")
    peaks = detect_flare_peaks(
        df, flux_col,
        threshold_factor=settings.DETECTION_PARAMS['threshold_factor'],
        window_size=settings.DETECTION_PARAMS['window_size']
    )
    print(f"Detected {len(peaks)} potential flare peaks")
    
    # Define flare bounds
    print("Defining flare bounds...")
    flares = define_flare_bounds(
        df, flux_col, peaks['peak_index'].values,
        start_threshold=settings.DETECTION_PARAMS['start_threshold'],
        end_threshold=settings.DETECTION_PARAMS['end_threshold'],
        min_duration=settings.DETECTION_PARAMS['min_duration'],
        max_duration=settings.DETECTION_PARAMS['max_duration']
    )
    print(f"Defined bounds for {len(flares)} flares")
    
    # Plot detected flares
    fig = plot_detected_flares(df, flux_col, flares)
    fig.savefig(os.path.join(args.output, f'detected_flares_{args.channel}.png'), 
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 3: Detect overlapping flares
    print("\n=== Detecting Overlapping Flares ===")
    overlapping = detect_overlapping_flares(flares, min_overlap='2min')
    print(f"Detected {len(overlapping)} potentially overlapping flare pairs")
    
    if overlapping:
        # Print information about overlapping flares
        print("Overlapping flare pairs:")
        for i, j, duration in overlapping:
            print(f"  Flares {i+1} and {j+1} overlap by {duration}")
    
    # Step 4: Remove background
    print("\n=== Removing Background Flux ===")
    df_bg = remove_background(
        df, 
        window_size=settings.BACKGROUND_PARAMS['window_size'],
        quantile=settings.BACKGROUND_PARAMS['quantile']
    )
    
    # Plot background-subtracted time series
    fig = plot_xrs_time_series(df_bg, f'{flux_col}_no_background', 
                             title=f'GOES XRS {args.channel} Background-Subtracted Data',
                             log_scale=True)
    fig.savefig(os.path.join(args.output, f'background_subtracted_{args.channel}.png'),
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 5: Load ML model for flare separation
    print("\n=== Loading ML Model for Flare Separation ===")
    model = FlareDecompositionModel(
        sequence_length=settings.ML_PARAMS['sequence_length'],
        n_features=settings.ML_PARAMS['n_features'],
        max_flares=settings.ML_PARAMS['max_flares']
    )
    model.build_model()
    
    try:
        model.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training a new model with synthetic data...")
        model = train_model(args)
    
    # Step 6: Apply ML model to separate overlapping flares
    print("\n=== Separating Overlapping Flares with ML Model ===")
    
    # Prepare data for ML analysis - focus on regions with overlapping flares
    ml_results = []
    
    if overlapping:
        for i, j, duration in overlapping:
            # Get the start and end indices for the overlapping region
            start_idx = min(flares.iloc[i]['start_index'], flares.iloc[j]['start_index'])
            end_idx = max(flares.iloc[i]['end_index'], flares.iloc[j]['end_index'])
            
            # Ensure we have enough context around the flares
            padding = settings.ML_PARAMS['sequence_length'] // 4
            start_idx = max(0, start_idx - padding)
            end_idx = min(len(df) - 1, end_idx + padding)
            
            # Extract the time series segment
            segment = df.iloc[start_idx:end_idx][flux_col].values
            
            # Ensure the segment has the required length for the model
            if len(segment) < settings.ML_PARAMS['sequence_length']:
                # Pad if too short
                segment = np.pad(segment, 
                                (0, settings.ML_PARAMS['sequence_length'] - len(segment)), 
                                'constant')
            elif len(segment) > settings.ML_PARAMS['sequence_length']:
                # Truncate or use a sliding window if too long
                segment = segment[:settings.ML_PARAMS['sequence_length']]
            
            # Reshape for model input
            segment = segment.reshape(1, -1, 1)
            
            # Decompose the flares
            original, individual_flares, combined = reconstruct_flares(
                model, segment, window_size=settings.ML_PARAMS['sequence_length'], plot=False
            )
            
            # Store the result
            ml_results.append({
                'overlapping_pair': (i, j),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'original': original.flatten(),
                'individual_flares': individual_flares,
                'combined': combined.flatten()
            })
            
            # Plot the decomposition
            timestamps = df.index[start_idx:start_idx+len(segment.flatten())]
            fig = plot_flare_decomposition(original.flatten(), individual_flares, timestamps)
            fig.savefig(os.path.join(args.output, f'decomposed_flares_{i+1}_{j+1}.png'),
                       dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 7: Calculate flare energies
    print("\n=== Calculating Flare Energies ===")
    
    # Traditional method - no separation
    print("Calculating energies with traditional method...")
    energy_results_trad = {}
    
    for i, flare in flares.iterrows():
        start_idx = flare['start_index']
        end_idx = flare['end_index']
        
        # Extract the flare segment
        flare_segment = df.iloc[start_idx:end_idx+1].copy()
        
        # Calculate energy
        flare_energy = calculate_flare_energy(
            flare_segment, flux_col, 
            background_column=f'{flux_col}_background' if f'{flux_col}_background' in df_bg.columns else None
        )
        
        # Store the energy
        energy_results_trad[i] = {
            'peak_flux': flare['peak_flux'],
            'integrated_flux': flare_energy['energy'].iloc[-1] if 'energy' in flare_energy.columns else None,
            'duration': flare['duration']
        }
    
    # ML method - with separation
    print("Calculating energies with ML separation...")
    energy_results_ml = {}
    
    for result in ml_results:
        i, j = result['overlapping_pair']
        individual_flares = result['individual_flares']
        
        # For each separated flare
        for k in range(individual_flares.shape[1]):
            if np.max(individual_flares[:, k]) > 0.05 * np.max(result['original']):
                # Calculate the energy using trapezoidal rule
                energy = np.trapezoid(individual_flares[:, k])
                
                # Store the energy
                energy_results_ml[f"{i}_{j}_{k}"] = {
                    'peak_flux': np.max(individual_flares[:, k]),
                    'integrated_flux': energy,
                    'original_flare': (i, j)
                }
    
    # Step 8: Fit power-law distributions
    print("\n=== Fitting Power-Law Distributions ===")
    
    # Prepare energy lists
    traditional_energies = [result['integrated_flux'] for result in energy_results_trad.values() 
                           if result['integrated_flux'] is not None]
    ml_energies = [result['integrated_flux'] for result in energy_results_ml.values() 
                   if result['integrated_flux'] is not None]
    
    # Fit power laws
    print("Fitting power law to traditional method energies...")
    powerlaw_trad = fit_power_law(
        traditional_energies,
        xmin=settings.POWERLAW_PARAMS['xmin'],
        xmax=settings.POWERLAW_PARAMS['xmax'],
        n_bootstrap=settings.POWERLAW_PARAMS['n_bootstrap'],
        plot=True
    )
    plt.savefig(os.path.join(args.output, 'powerlaw_traditional.png'), 
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    if ml_energies:
        print("Fitting power law to ML-separated energies...")
        powerlaw_ml = fit_power_law(
            ml_energies,
            xmin=settings.POWERLAW_PARAMS['xmin'],
            xmax=settings.POWERLAW_PARAMS['xmax'],
            n_bootstrap=settings.POWERLAW_PARAMS['n_bootstrap'],
            plot=True
        )
        plt.savefig(os.path.join(args.output, 'powerlaw_ml_separated.png'),
                   dpi=settings.VISUALIZATION_PARAMS['dpi'])
        
        # Compare the power laws
        print("\n=== Comparing Power-Law Distributions ===")
        comparison = compare_flare_populations(
            traditional_energies, "Traditional Method",
            ml_energies, "ML-Separated",
            xmin=settings.POWERLAW_PARAMS['xmin'],
            xmax=settings.POWERLAW_PARAMS['xmax'],
            plot=True
        )
        plt.savefig(os.path.join(args.output, 'powerlaw_comparison.png'),
                   dpi=settings.VISUALIZATION_PARAMS['dpi'])
        
        # Print comparison results
        print("\nPower-law comparison results:")
        print(f"  Traditional method: α = {powerlaw_trad['alpha']:.3f} ± {powerlaw_trad['alpha_err']:.3f}")
        print(f"  ML-separated method: α = {powerlaw_ml['alpha']:.3f} ± {powerlaw_ml['alpha_err']:.3f}")
        print(f"  Difference: {comparison['alpha_diff']:.3f} ± {comparison['alpha_err_combined']:.3f}")
        print(f"  Significance: {comparison['significance']:.2f}σ")
        print(f"  p-value: {comparison['p_value']:.3e}")
    else:
        print("No ML-separated flare energies available for comparison.")
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {args.output}")


def train_bayesian_model(args):
    """Train the enhanced Bayesian flare analysis model."""
    print("\n=== Training Enhanced Bayesian Flare Analysis Model ===")
    
    # Initialize enhanced Bayesian analyzer
    analyzer = BayesianFlareAnalyzer(
        sequence_length=128,
        n_features=2,
        max_flares=3,
        n_monte_carlo_samples=100,
        sensor_noise_std=0.01
    )
    
    print("Building enhanced Bayesian neural network...")
    model = analyzer.build_bayesian_model()
    print(f"Model built with {model.count_params():,} parameters")
    
    # Generate physics-based synthetic data
    print("Generating synthetic solar flare data with physics constraints...")
    X_synthetic, y_synthetic = analyzer.generate_synthetic_data_with_physics(
        n_samples=2000, noise_level=0.02
    )
    print(f"Generated {X_synthetic.shape[0]} synthetic samples")
    
    # Train with Monte Carlo data augmentation
    print("Training with Monte Carlo data augmentation...")
    history = analyzer.train_bayesian_model(
        X_synthetic, y_synthetic,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        augment_data=True
    )
    
    # Save the trained model
    model_path = os.path.join(args.output, 'enhanced_bayesian_flare_model')
    analyzer.save_bayesian_model(model_path)
    print(f"Enhanced Bayesian model saved to {model_path}")
    
    # Demonstrate uncertainty quantification
    print("\nDemonstrating uncertainty quantification...")
    X_test = X_synthetic[-100:]
    y_test = y_synthetic[-100:]
    
    # Monte Carlo predictions
    predictions = analyzer.monte_carlo_predict(X_test, n_samples=100)
    print(f"Mean uncertainty: {np.mean(predictions['std']):.6f}")
    
    # Separate epistemic and aleatoric uncertainty
    uncertainty_components = analyzer.quantify_epistemic_aleatoric_uncertainty(
        X_test[:20], n_epistemic=30, n_aleatoric=50
    )
    
    print(f"Epistemic uncertainty: {np.mean(uncertainty_components['epistemic_uncertainty']):.6f}")
    print(f"Aleatoric uncertainty: {np.mean(uncertainty_components['aleatoric_uncertainty']):.6f}")
    
    # MCMC sampling
    try:
        print("Performing MCMC sampling...")
        mcmc_results = analyzer.mcmc_sampling(
            X_test[:10], y_test[:10],
            n_samples=200, n_burnin=100
        )
        print(f"MCMC acceptance rate: {mcmc_results['acceptance_rate']:.3f}")
    except Exception as e:
        print(f"MCMC sampling issue: {e}")
    
    # Visualization
    fig = analyzer.plot_uncertainty_analysis(X_test, predictions, y_test)
    plt.savefig(os.path.join(args.output, 'bayesian_uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Enhanced Bayesian model training completed!")
    return analyzer


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Train model if requested
    if args.train:
        train_model(args)
    
    # Analyze flares
    analyze_flares(args)


if __name__ == "__main__":
    main()
