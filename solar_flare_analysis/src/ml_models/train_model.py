"""
Solar Flare and Storm Detection Model Training Script
This script initializes and trains the transformer model with your XRS data
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_flare_model import TransformerFlareModel
import matplotlib.pyplot as plt

def main():
    """
    Main training function
    """
    print("🌟" + "="*80)
    print("🚀 SOLAR FLARE & STORM DETECTION MODEL TRAINING")
    print("🌟" + "="*80)
    
    # Configuration
    print("\n📋 Configuration:")
    
    # Data directory (adjust path as needed)
    data_directory = r'c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\data\XRS'
    
    # Training parameters
    config = {
        'sequence_length': 256,      # Longer sequences for better temporal context
        'n_features': 2,             # GOES-A and GOES-B channels
        'd_model': 128,              # Model embedding dimension
        'num_heads': 12,             # Number of attention heads
        'num_transformer_blocks': 6, # Number of transformer layers
        'ff_dim': 256,               # Feed-forward network dimension
        'dropout_rate': 0.15,        # Dropout for regularization
        'learning_rate': 0.0001,     # Learning rate
        'epochs': 1,                # Number of training epochs (start with 50)
        'batch_size': 32,            # Batch size (adjust based on GPU memory)
        'train_years': [2023, 2024], # Years for training data
        'test_years': [2025]         # Years for testing data
    }
    
    # Print configuration
    for key, value in config.items():
        print(f"   • {key}: {value}")
    
    print(f"\n📁 Data Directory: {data_directory}")
    
    # Check if data directory exists
    if not os.path.exists(data_directory):
        print(f"❌ ERROR: Data directory not found: {data_directory}")
        print("💡 Please ensure the XRS data directory exists and contains CSV files")
        print("   Expected file format: CSV files with columns like 'time_tag', 'A_FLUX', 'B_FLUX'")
        return
    
    # List available data files
    import glob
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    print(f"\n📊 Found {len(csv_files)} CSV files in data directory")
    
    if len(csv_files) == 0:
        print("❌ ERROR: No CSV files found in the data directory")
        print("💡 Please check that your XRS data files are in CSV format")
        return
    
    # Show first few files
    print("📁 Sample files:")
    for i, file in enumerate(csv_files[:5]):
        print(f"   {i+1}. {os.path.basename(file)}")
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files)-5} more files")
    
    try:
        # Initialize the model
        print(f"\n🔧 Initializing Solar Flare Storm Detector...")
        
        model = TransformerFlareModel(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_transformer_blocks=config['num_transformer_blocks'],
            ff_dim=config['ff_dim'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            data_dir=data_directory
        )
        
        print("✅ Model initialized successfully!")
        
        # Display model configuration
        print(f"\n🏗️ Model Architecture:")
        print(f"   • Input Shape: ({config['sequence_length']}, {config['n_features']})")
        print(f"   • Model Dimension: {config['d_model']}")
        print(f"   • Attention Heads: {config['num_heads']}")
        print(f"   • Transformer Blocks: {config['num_transformer_blocks']}")
        print(f"   • Feed-Forward Dim: {config['ff_dim']}")
        print(f"   • Dropout Rate: {config['dropout_rate']}")
        
        # Start training pipeline
        print(f"\n🚀 Starting Complete Training Pipeline...")
        print("   This will:")
        print("   1. Load and preprocess XRS data")
        print("   2. Apply baseline correction (AsLS algorithm)")
        print("   3. Apply Savitzky-Golay smoothing")
        print("   4. Create training sequences")
        print("   5. Build and compile the transformer model")
        print("   6. Train the model with advanced callbacks")
        print("   7. Evaluate performance")
        print("   8. Save model and generate visualizations")
        
        # Run the complete pipeline
        history, predictions = model.run_complete_pipeline(
            train_years=config['train_years'],
            test_years=config['test_years'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            save_results=True
        )
        
        # Training completed successfully
        print("\n🎉" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉" + "="*60)
        
        # Summary of outputs
        print(f"\n💾 Generated Files:")
        output_files = [
            "solar_flare_storm_model.h5",
            "solar_flare_storm_model_scalers.pkl", 
            "evaluation_dashboard.png",
            "training_history.png",
            "attention_analysis.png",
            "training_log.csv",
            "best_solar_flare_storm_model.h5"
        ]
        
        for file in output_files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ⚠️ {file} (may not be generated)")
        
        # Performance summary
        if history and hasattr(history, 'history'):
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
            best_val_loss = min(history.history['val_loss']) if 'val_loss' in history.history else 0
            
            print(f"\n📈 Training Summary:")
            print(f"   • Final Training Loss: {final_loss:.4f}")
            print(f"   • Final Validation Loss: {final_val_loss:.4f}")
            print(f"   • Best Validation Loss: {best_val_loss:.4f}")
            print(f"   • Total Epochs Trained: {len(history.history['loss'])}")
        
        print(f"\n🔍 Next Steps:")
        print("   1. Check the evaluation_dashboard.png for model performance")
        print("   2. Review training_history.png for training progress")
        print("   3. Examine attention_analysis.png for model interpretability")
        print("   4. Use the saved model for predictions on new data")
        
        print(f"\n💡 Model Usage:")
        print("   # Load the trained model")
        print("   model = SolarFlareStormDetector()")
        print("   model.load_model('solar_flare_storm_model.h5')")
        print("   # Make predictions on new data")
        print("   predictions = model.model.predict(new_data)")
        
    except FileNotFoundError as e:
        print(f"\n❌ DATA ERROR: {e}")
        print("💡 Solutions:")
        print("   1. Check that the data directory path is correct")
        print("   2. Ensure CSV files contain the required columns:")
        print("      - 'time_tag' (datetime)")
        print("      - 'A_FLUX' or 'xrsa' (GOES-A channel)")
        print("      - 'B_FLUX' or 'xrsb' (GOES-B channel)")
        print("   3. Verify data files for the specified years exist")
        
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        print("💡 Possible solutions:")
        print("   1. Reduce batch_size if you get memory errors")
        print("   2. Reduce sequence_length if data is insufficient")
        print("   3. Check data format and column names")
        print("   4. Ensure sufficient disk space for model saving")
        
        # Print detailed error for debugging
        import traceback
        print(f"\n🔍 Detailed Error Information:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
