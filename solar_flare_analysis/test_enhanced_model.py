"""
Test script for the enhanced ML model components
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_enhanced_model():
    """Test the enhanced flare decomposition model"""
    print("Testing Enhanced Flare Decomposition Model...")
    
    try:
        from ml_models.enhanced_flare_analysis import EnhancedFlareDecompositionModel, NanoflareDetector
        
        # Initialize model
        model = EnhancedFlareDecompositionModel(
            sequence_length=128,
            n_features=2,
            max_flares=3
        )
        
        # Build model
        model.build_enhanced_model()
        print("✓ Model built successfully")
        
        # Generate synthetic data
        X, y = model.generate_enhanced_synthetic_data(n_samples=100)
        print(f"✓ Generated synthetic data: X shape {X.shape}")
        
        # Test training (short run)
        history = model.train_enhanced_model(X, y, epochs=5, batch_size=16)
        print("✓ Model training completed")
        
        # Test prediction
        predictions = model.predict_enhanced(X[:10])
        print("✓ Model prediction completed")
        
        # Test nanoflare detector
        detector = NanoflareDetector()
        test_signal = np.random.lognormal(-8, 1, 1000) * 1e-6
        nanoflare_results = detector.detect_nanoflares(test_signal)
        print(f"✓ Nanoflare detector found {nanoflare_results['total_count']} events")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing enhanced model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """Test the enhanced data loader"""
    print("\nTesting Enhanced Data Loader...")
    
    try:
        from data_processing.enhanced_data_loader import GOESDataLoader, SyntheticDataGenerator
        
        # Test synthetic data generation
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate_synthetic_dataset(
            duration_hours=2, 
            flare_rate=1.0,
            include_nanoflares=True
        )
        print(f"✓ Generated synthetic dataset: {len(synthetic_data)} points")
        
        # Test data loader preprocessing
        loader = GOESDataLoader()
        processed_data = loader.preprocess_data(
            synthetic_data,
            resample_freq='1min',
            apply_quality_filter=True,
            normalize_channels=True
        )
        print(f"✓ Preprocessed data: {len(processed_data)} points")
        
        # Test sequence creation
        X, y = loader.create_ml_training_sequences(
            processed_data,
            sequence_length=64,
            step_size=32
        )
        if X is not None:
            print(f"✓ Created training sequences: {X.shape}")
        else:
            print("! No sequences created (normal for short data)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all components"""
    print("\nTesting Integration...")
    
    try:
        from data_processing.enhanced_data_loader import SyntheticDataGenerator
        from ml_models.enhanced_flare_analysis import (
            EnhancedFlareDecompositionModel, 
            NanoflareDetector, 
            FlareEnergyAnalyzer
        )
        
        # Generate realistic synthetic data
        generator = SyntheticDataGenerator()
        data = generator.generate_synthetic_dataset(
            duration_hours=6,
            flare_rate=2.0,
            include_nanoflares=True
        )
        print(f"✓ Generated {len(data)} data points")
        
        # Initialize components
        model = EnhancedFlareDecompositionModel(sequence_length=64, n_features=2, max_flares=3)
        model.build_enhanced_model()
        
        detector = NanoflareDetector()
        analyzer = FlareEnergyAnalyzer()
        
        # Train model quickly
        X, y = model.generate_enhanced_synthetic_data(n_samples=200)
        model.train_enhanced_model(X, y, epochs=3, batch_size=32)
        print("✓ Model trained")
        
        # Analyze nanoflares
        nanoflare_results = detector.detect_nanoflares(data['xrs_a'].values)
        print(f"✓ Detected {nanoflare_results['total_count']} nanoflares")
        
        # Energy analysis
        if nanoflare_results['total_count'] > 0:
            flare_data = {'energies': nanoflare_results['energies']}
            energy_results = analyzer.analyze_energy_distribution(flare_data)
            print("✓ Energy analysis completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Enhanced Solar Flare Analysis - Component Tests")
    print("=" * 50)
    
    results = []
    
    # Test individual components
    results.append(test_enhanced_model())
    results.append(test_data_loader())
    results.append(test_integration())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Enhanced Model: {'PASS' if results[0] else 'FAIL'}")
    print(f"Data Loader: {'PASS' if results[1] else 'FAIL'}")
    print(f"Integration: {'PASS' if results[2] else 'FAIL'}")
    
    if all(results):
        print("\n✓ All tests PASSED! The enhanced ML system is ready.")
    else:
        print("\n✗ Some tests FAILED. Please check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
