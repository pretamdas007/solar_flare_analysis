#!/usr/bin/env python3
"""
Test script to verify that the enhanced visualization section works properly
after removing the old broken visualization code.
"""

import numpy as np
import sys
sys.path.append('solar_flare_analysis')

def test_visualization_methods():
    """Test the enhanced visualization methods"""
    from solar_flare_analysis.enhanced_train_production import EnhancedMLTrainer
    
    print("🔬 Testing Enhanced Visualization Methods")
    print("=" * 50)
    
    # Create a trainer instance
    trainer = EnhancedMLTrainer()
    
    # Check that the visualization method exists and is callable
    if hasattr(trainer, '_create_enhanced_visualizations'):
        print("✅ _create_enhanced_visualizations method exists")
    else:
        print("❌ _create_enhanced_visualizations method missing")
        return False
    
    # Check specific visualization methods
    required_methods = [
        '_create_professional_timeseries_plot',
        '_create_enhanced_distribution_plot', 
        '_create_statistical_dashboard',
        '_create_advanced_correlation_matrix',
        '_create_flare_intensity_analysis',
        '_create_feature_analysis_plot',
        '_create_professional_performance_heatmap',
        '_create_convergence_analysis',
        '_create_complexity_performance_plot',
        '_create_model_history_plot',
        '_create_enhanced_summary_panel'
    ]
    
    for method_name in required_methods:
        if hasattr(trainer, method_name):
            print(f"✅ {method_name} method exists")
        else:
            print(f"❌ {method_name} method missing")
    
    print("\n🎯 Testing with synthetic data...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    sequence_length = 50
    n_features = 2
    
    X_train = np.random.randn(n_samples, sequence_length, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    X_val = np.random.randn(20, sequence_length, n_features) 
    y_val = np.random.randint(0, 2, 20)
    
    # Mock results
    results = {
        'transformer': {'status': 'success', 'history': None},
        'monte_carlo': {'status': 'failed', 'error': 'Test error'}
    }
    
    try:
        # Test the main visualization function
        trainer._create_enhanced_visualizations(X_train, y_train, X_val, y_val, results)
        print("✅ Enhanced visualizations created successfully!")
        print(f"📁 Check output in: {trainer.output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_methods()
    if success:
        print("\n🎉 All visualization tests passed!")
    else:
        print("\n💥 Some visualization tests failed!")
