"""
Simple Example: Fit a single flare and make a prediction

This script demonstrates the basic usage of the solar flare analysis pipeline
on a single synthetic flare.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from src.fit_flare_model import SolarFlareModel
from src.feature_extract import FlareFeatureExtractor

def generate_synthetic_flare():
    """Generate a synthetic flare for demonstration"""
    # Time array (10 minutes of data, 1-second resolution)
    t = np.arange(0, 600, 1)
    
    # Flare parameters
    A = 1e-7  # Amplitude (W/m²)
    B = 200   # Peak time (seconds)
    C = 100   # Width (seconds)
    D = 5e-4  # Decay rate (1/seconds)
    
    # Generate flare profile using Gryciuk model
    Z = (2 * B + C**2 * D) / (2 * C)
    exp_term = np.exp(D * (B - t) + (C**2 * D**2) / 4)
    erf_term1 = erf(Z)
    erf_term2 = erf((Z - t) / C)
    flux = 0.5 * np.sqrt(np.pi) * A * C * exp_term * (erf_term1 - erf_term2)
    
    # Add background and noise
    background = 1e-9
    noise = np.random.normal(0, background * 0.05, len(t))
    flux = flux + background + noise
    
    return t, flux, {'A': A, 'B': B, 'C': C, 'D': D}

def main():
    """Main demonstration function"""
    print("🔭 Solar Flare Analysis - Simple Example")
    print("=" * 50)
    
    # Step 1: Generate synthetic flare data
    print("1. Generating synthetic flare data...")
    t, flux, true_params = generate_synthetic_flare()
    print(f"   Generated flare with {len(t)} data points")
    print(f"   True parameters: A={true_params['A']:.2e}, B={true_params['B']}, "
          f"C={true_params['C']}, D={true_params['D']:.2e}")
    
    # Step 2: Fit the model
    print("\n2. Fitting flare model...")
    model = SolarFlareModel()
    fit_result = model.fit_flare(t, flux - np.min(flux))  # Remove background
    
    if fit_result['success']:
        fitted_params = fit_result['parameters']
        print(f"   ✓ Fit successful!")
        print(f"   Fitted parameters: A={fitted_params['A']:.2e}, B={fitted_params['B']:.1f}, "
              f"C={fitted_params['C']:.1f}, D={fitted_params['D']:.2e}")
        print(f"   R² = {fit_result['r_squared']:.4f}")
        print(f"   RMSE = {fit_result['rmse']:.2e}")
    else:
        print("   ✗ Fit failed!")
        return
    
    # Step 3: Extract features
    print("\n3. Extracting features...")
    extractor = FlareFeatureExtractor()
    features = extractor.extract_all_features(fit_result)
    
    print(f"   Extracted {len(features)} features")
    print("   Key features:")
    print(f"   - Peak flux: {features.get('peak_flux', 0):.2e} W/m²")
    print(f"   - Duration: {features.get('duration', 0):.1f} s")
    print(f"   - Rise time: {features.get('time_to_peak', 0):.1f} s")
    print(f"   - Asymmetry: {features.get('asymmetry', 0):.2f}")
    
    # Step 4: Create visualization
    print("\n4. Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Original data and fit
    ax1.plot(t, flux, 'ko-', markersize=2, alpha=0.6, label='Observed')
    ax1.plot(fit_result['normalized_time'], fit_result['fitted_flux'] + np.min(flux), 
             'r-', linewidth=2, label='Fitted Model')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Flux (W/m²)')
    ax1.set_title(f'Solar Flare Fit (R² = {fit_result["r_squared"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add parameter annotations
    param_text = (f"A = {fitted_params['A']:.2e}\n"
                  f"B = {fitted_params['B']:.1f} s\n"
                  f"C = {fitted_params['C']:.1f} s\n"
                  f"D = {fitted_params['D']:.2e} s⁻¹")
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Residuals
    residuals = flux[:-len(flux)+len(fit_result['fitted_flux'])] - (fit_result['fitted_flux'] + np.min(flux))
    if len(residuals) == len(fit_result['normalized_time']):
        ax2.plot(fit_result['normalized_time'], residuals, 'bo-', markersize=2, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Residuals (W/m²)')
        ax2.set_title('Fit Residuals')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_example_result.png', dpi=300, bbox_inches='tight')
    print("   ✓ Plot saved as 'simple_example_result.png'")
    
    # Step 5: Parameter comparison
    print("\n5. Parameter comparison:")
    print("   Parameter | True Value | Fitted Value | Error (%)")
    print("   ----------|------------|--------------|----------")
    for param in ['A', 'B', 'C', 'D']:
        true_val = true_params[param]
        fitted_val = fitted_params[param]
        error_pct = abs(fitted_val - true_val) / true_val * 100
        print(f"   {param:9s} | {true_val:10.2e} | {fitted_val:12.2e} | {error_pct:8.1f}")
    
    # Step 6: Synthetic prediction example
    print("\n6. Synthetic ML prediction example:")
    
    # Simple heuristic for alpha prediction (for demonstration)
    # In reality, this would use a trained ML model
    duration = features.get('duration', 1)
    peak_flux = features.get('peak_flux', 1e-9)
    
    # Rough heuristic: smaller, longer flares have higher alpha
    flux_factor = np.log10(peak_flux / 1e-9)  # log scale relative to nanoflare level
    duration_factor = duration / 1000  # normalize by typical duration
    
    predicted_alpha = 2.0 - 0.3 * flux_factor + 0.2 * duration_factor
    predicted_alpha = np.clip(predicted_alpha, 1.5, 3.0)
    
    is_nanoflare = peak_flux < 1e-8
    
    print(f"   Predicted α (power-law index): {predicted_alpha:.2f}")
    print(f"   Nanoflare classification: {'Yes' if is_nanoflare else 'No'}")
    print("   (Note: This is a simple heuristic for demonstration)")
    
    print("\n" + "=" * 50)
    print("🎉 Example complete!")
    print("Next steps:")
    print("- Run 'python setup.py' to set up the full pipeline")
    print("- Run 'python run_complete_workflow.py' for the complete analysis")
    print("- Place your own GOES XRS data in the data/ directory")
    print("=" * 50)

if __name__ == '__main__':
    main()
