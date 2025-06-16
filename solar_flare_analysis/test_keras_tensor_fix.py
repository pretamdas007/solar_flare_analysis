#!/usr/bin/env python3
"""
Test script to verify KerasTensor error fixes
Tests that all models can be built without TensorFlow function errors
"""

import sys
import os
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the src directory to the path
sys.path.append('src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transformer_model():
    """Test TransformerFlareModel can be built without KerasTensor errors"""
    try:
        from src.ml_models.transformer_flare_model import TransformerFlareModel
        
        logger.info("Testing TransformerFlareModel...")
        model = TransformerFlareModel(
            sequence_length=128,
            n_features=2,
            n_classes=6,
            d_model=64,
            num_heads=4,
            num_transformer_blocks=2
        )
        
        # Build the model - this is where KerasTensor errors would occur
        keras_model = model.build_model()
        logger.info(f"✓ TransformerFlareModel built successfully with {keras_model.count_params():,} parameters")
        
        # Test with sample data
        sample_data = np.random.randn(2, 128, 2)
        output = keras_model.predict(sample_data, verbose=0)
        logger.info(f"✓ Model prediction successful, output shapes: {[o.shape for o in output]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TransformerFlareModel test failed: {e}")
        return False

def test_convolutional_transformer_model():
    """Test ConvolutionalTransformerModel can be built"""
    try:
        from src.ml_models.transformer_flare_model import ConvolutionalTransformerModel
        
        logger.info("Testing ConvolutionalTransformerModel...")
        model = ConvolutionalTransformerModel(
            sequence_length=128,
            n_features=2,
            n_classes=6,
            conv_filters=[32, 64],
            d_model=64,
            num_heads=4,
            num_transformer_blocks=2
        )
        
        keras_model = model.build_model()
        logger.info(f"✓ ConvolutionalTransformerModel built successfully with {keras_model.count_params():,} parameters")
        
        # Test with sample data
        sample_data = np.random.randn(2, 128, 2)
        output = keras_model.predict(sample_data, verbose=0)
        logger.info(f"✓ Model prediction successful, output shapes: {[o.shape for o in output]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ConvolutionalTransformerModel test failed: {e}")
        return False

def test_graph_neural_model():
    """Test GraphNeuralFlareModel can be built"""
    try:
        from src.ml_models.graph_neural_model import GraphNeuralFlareModel
        
        logger.info("Testing GraphNeuralFlareModel...")
        model = GraphNeuralFlareModel(
            sequence_length=64,  # Smaller for testing
            n_features=2,
            n_classes=6,
            hidden_units=32,
            num_gat_layers=2,
            num_heads=4
        )
        
        keras_model = model.build_model()
        logger.info(f"✓ GraphNeuralFlareModel built successfully with {keras_model.count_params():,} parameters")
        
        # Test with sample data (needs both node features and adjacency matrix)
        sample_features = np.random.randn(2, 64, 2)
        sample_adjacency = np.random.randint(0, 2, (2, 64, 64)).astype(np.float32)
        output = keras_model.predict([sample_features, sample_adjacency], verbose=0)
        logger.info(f"✓ Model prediction successful, output shapes: {[o.shape for o in output]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ GraphNeuralFlareModel test failed: {e}")
        return False

def test_contrastive_learning_model():
    """Test ContrastiveLearningModel can be built"""
    try:
        from src.ml_models.self_supervised_models import ContrastiveLearningModel
        
        logger.info("Testing ContrastiveLearningModel...")
        model = ContrastiveLearningModel(
            sequence_length=128,
            n_features=2,
            projection_dim=64,
            temperature=0.1
        )
        
        # Build encoder and projection head
        encoder = model.build_encoder()
        projection_head = model.build_projection_head()
        
        logger.info(f"✓ ContrastiveLearningModel built successfully")
        logger.info(f"  Encoder parameters: {encoder.count_params():,}")
        logger.info(f"  Projection head parameters: {projection_head.count_params():,}")
        
        # Test with sample data
        sample_data = np.random.randn(2, 128, 2)
        representations = encoder.predict(sample_data, verbose=0)
        projections = projection_head.predict(representations, verbose=0)
        logger.info(f"✓ Model prediction successful, shapes: repr={representations.shape}, proj={projections.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ContrastiveLearningModel test failed: {e}")
        return False

def test_masked_autoencoder_model():
    """Test MaskedAutoencoderModel can be built"""
    try:
        from src.ml_models.self_supervised_models import MaskedAutoencoderModel
        
        logger.info("Testing MaskedAutoencoderModel...")
        model = MaskedAutoencoderModel(
            sequence_length=128,
            n_features=2,
            mask_ratio=0.15,
            encoder_dim=128,
            decoder_dim=64
        )
        
        autoencoder = model.build_autoencoder()
        logger.info(f"✓ MaskedAutoencoderModel built successfully with {autoencoder.count_params():,} parameters")
        
        # Test with sample data
        sample_data = np.random.randn(2, 128, 2)
        output = autoencoder.predict(sample_data, verbose=0)
        logger.info(f"✓ Model prediction successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MaskedAutoencoderModel test failed: {e}")
        return False

def main():
    """Run all model tests"""
    logger.info("="*60)
    logger.info("TESTING KERAS TENSOR FIXES")
    logger.info("="*60)
    
    tests = [
        ("Transformer Model", test_transformer_model),
        ("Convolutional Transformer Model", test_convolutional_transformer_model),
        ("Graph Neural Model", test_graph_neural_model),
        ("Contrastive Learning Model", test_contrastive_learning_model),
        ("Masked Autoencoder Model", test_masked_autoencoder_model)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            failed += 1
    
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS")
    logger.info("="*60)
    logger.info(f"✓ Passed: {passed}")
    logger.info(f"✗ Failed: {failed}")
    logger.info(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All KerasTensor fixes successful! Models can be built without errors.")
    else:
        logger.info("⚠️  Some issues remain - check the error messages above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
