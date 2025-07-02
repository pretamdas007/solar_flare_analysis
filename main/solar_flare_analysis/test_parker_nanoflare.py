#!/usr/bin/env python3
"""
Test Parker's Nanoflare Hypothesis Analysis

This script creates sample data to test the updated nanoflare analysis
with Parker's threshold (α > 2.0) for coronal heating capability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nanoflare_analysis import NanoflareAnalyzer

def create_sample_data():
    """Create sample alpha values for testing Parker's hypothesis"""
    
    # Create two scenarios to test
    scenarios = {
        'coronal_heating_capable': {
            'description': 'α > 2.0 - Nanoflares can power coronal heating',
            'mean_alpha': 2.3,
            'std_alpha': 0.4,
            'n_samples': 500
        },
        'insufficient_heating': {
            'description': 'α < 2.0 - Insufficient for coronal heating',
            'mean_alpha': 1.7,
            'std_alpha': 0.3,
            'n_samples': 500
        }
    }
    
    for scenario_name, params in scenarios.items():
        print(f"\n{'='*70}")
        print(f"TESTING: {params['description']}")
        print(f"{'='*70}")
        
        # Generate sample alpha values
        np.random.seed(42)  # For reproducible results
        alpha_values = np.random.normal(
            params['mean_alpha'], 
            params['std_alpha'], 
            params['n_samples']
        )
        
        # Ensure positive values
        alpha_values = np.abs(alpha_values)
        
        print(f"Generated {len(alpha_values)} alpha values")
        print(f"Range: [{alpha_values.min():.3f}, {alpha_values.max():.3f}]")
        print(f"Mean: {alpha_values.mean():.3f} ± {alpha_values.std():.3f}")
        
        # Initialize Parker's analyzer
        analyzer = NanoflareAnalyzer(alpha_threshold=2.0, alpha_uncertainty=0.03)
        
        # Perform analysis
        analysis_result = analyzer.analyze_alpha_predictions(alpha_values)
        
        # Print detailed results
        alpha_stats = analysis_result['alpha_statistics']
        power_law = analysis_result['power_law_analysis']
        nanoflare = analysis_result['nanoflare_detection']
        
        print(f"\nALPHA STATISTICS:")
        print(f"  Mean: {alpha_stats['mean']:.4f} ± {alpha_stats['std']:.4f}")
        print(f"  Range: [{alpha_stats['min']:.4f}, {alpha_stats['max']:.4f}]")
        print(f"  Median: {alpha_stats.get('median', np.median(alpha_values)):.4f}")
        
        print(f"\nPOWER-LAW ANALYSIS:")
        if not np.isnan(power_law.get('alpha', np.nan)):
            print(f"  Measured β: {power_law['alpha']:.4f} ± {power_law.get('alpha_error', 0):.4f}")
            print(f"  R²: {power_law.get('r_squared', 0):.4f}")
            print(f"  Fit quality: {power_law.get('fit_quality', 'unknown')}")
        else:
            print(f"  Fit failed: {power_law.get('fit_quality', 'unknown')}")
        
        print(f"\nPARKER'S NANOFLARE DETECTION:")
        print(f"  Parker Threshold: 2.0 ± 0.03")
        
        if nanoflare['is_nanoflare']:
            print(f"  🟢 NANOFLARES DETECTED - CORONAL HEATING CAPABLE!")
            print(f"  🔥 Can power solar coronal heating via Parker's mechanism")
            print(f"  Detection confidence: {nanoflare['detection_confidence']:.1%}")
            if not np.isnan(nanoflare.get('z_score', np.nan)):
                print(f"  Statistical significance: {nanoflare['z_score']:.2f}σ")
            if not np.isnan(power_law.get('alpha', np.nan)):
                excess = power_law['alpha'] - 2.0
                print(f"  Exceeds Parker threshold by: {excess:.3f}")
        else:
            print(f"  🔴 NO NANOFLARES DETECTED - INSUFFICIENT FOR CORONAL HEATING")
            print(f"  ❄️ Cannot power coronal heating via Parker's mechanism")
            print(f"  Reason: {nanoflare['reason'].replace('_', ' ').title()}")
            print(f"  Detection confidence: {nanoflare['detection_confidence']:.1%}")
            if not np.isnan(power_law.get('alpha', np.nan)):
                deficit = 2.0 - power_law['alpha']
                print(f"  Below Parker threshold by: {deficit:.3f}")
        
        # Create visualization
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / f'parker_test_{scenario_name}.png'
        print(f"\nCreating visualization: {plot_path}")
        analyzer.plot_analysis(analysis_result, alpha_values, plot_path)
        
        # Save sample data for further testing
        df = pd.DataFrame({'alpha': alpha_values})
        data_path = output_dir / f'sample_data_{scenario_name}.csv'
        df.to_csv(data_path, index=False)
        print(f"Sample data saved: {data_path}")

if __name__ == '__main__':
    print("Testing Parker's Nanoflare Hypothesis Analysis")
    print("Testing coronal heating capability with α > 2.0 threshold")
    create_sample_data()
    print(f"\n{'='*70}")
    print("PARKER'S HYPOTHESIS TESTING COMPLETE!")
    print("Check the output/ directory for visualizations and sample data")
    print(f"{'='*70}")
