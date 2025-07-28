"""
Usage example for the enhanced Solar Flare and Storm Detection Transformer Model
This script demonstrates how to use the new SolarFlareStormDetector class
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_flare_model import SolarFlareStormDetector

def main():
    """
    Example usage of the Solar Flare and Storm Detection model
    """
    print("🌟 Solar Flare and Storm Detection Model Example")
    print("=" * 50)
    
    # Initialize the model with custom parameters
    model = SolarFlareStormDetector(
        sequence_length=256,        # Longer sequences for better context
        n_features=2,              # GOES-A and GOES-B channels
        d_model=128,               # Model dimension
        num_heads=12,              # Number of attention heads
        num_transformer_blocks=6,   # Number of transformer blocks
        ff_dim=256,                # Feed-forward dimension
        dropout_rate=0.15,         # Dropout for regularization
        learning_rate=0.0001,      # Learning rate
        data_dir='c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis\\data\\XRS'
    )
    
    print("✅ Model initialized successfully!")
    print(f"📋 Model Configuration:")
    print(f"   • Sequence Length: {model.sequence_length}")
    print(f"   • Features: {model.n_features}")
    print(f"   • Model Dimension: {model.d_model}")
    print(f"   • Attention Heads: {model.num_heads}")
    print(f"   • Transformer Blocks: {model.num_transformer_blocks}")
    
    try:
        # Run the complete pipeline
        print("\\n🚀 Starting complete pipeline...")
        
        # Note: Adjust years based on available data
        train_years = [2023, 2024]
        test_years = [2025]
        
        history, predictions = model.run_complete_pipeline(
            train_years=train_years,
            test_years=test_years,
            epochs=50,  # Reduced for demo
            batch_size=32,
            save_results=True
        )
        
        print("\\n🎉 Pipeline completed successfully!")
        print("\\n📊 Results saved:")
        print("   • Model: solar_flare_storm_model.h5")
        print("   • Scalers: solar_flare_storm_model_scalers.pkl")
        print("   • Evaluation: evaluation_dashboard.png")
        print("   • Training History: training_history.png")
        print("   • Attention Analysis: attention_analysis.png")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("💡 Please ensure XRS data files are available in the specified directory")
        print("   Expected format: CSV files with columns like 'time_tag', 'A_FLUX', 'B_FLUX'")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        print("💡 This might be due to insufficient data or memory constraints")

if __name__ == "__main__":
    main()
