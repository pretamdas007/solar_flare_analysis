#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for ML flare decomposition module.
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.ml_models.flare_decomposition import FlareDecompositionModel, reconstruct_flares


class TestFlareDecomposition(unittest.TestCase):
    """Test cases for ML flare decomposition module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set parameters
        self.sequence_length = 64
        self.n_features = 1
        self.max_flares = 3
        
        # Initialize model
        self.model = FlareDecompositionModel(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            max_flares=self.max_flares,
            dropout_rate=0.2
        )
        self.model.build_model()
        
        # Create simple test data
        self.X_test = np.zeros((2, self.sequence_length, self.n_features))
        
        # First sample: single Gaussian peak
        x = np.linspace(-3, 3, self.sequence_length)
        y = np.exp(-x**2)
        self.X_test[0, :, 0] = y
        
        # Second sample: two Gaussian peaks
        x1 = np.linspace(-4, 2, self.sequence_length)
        x2 = np.linspace(-2, 4, self.sequence_length)
        y1 = 0.8 * np.exp(-x1**2)
        y2 = 0.6 * np.exp(-x2**2)
        self.X_test[1, :, 0] = y1 + y2
        
    def test_model_architecture(self):
        """Test model architecture."""
        # Check model structure
        self.assertIsNotNone(self.model.model)
        
        # Check input shape
        self.assertEqual(self.model.model.input.shape[1:], (self.sequence_length, self.n_features))
        
        # Check output shape
        self.assertEqual(self.model.model.output.shape[1:], (self.sequence_length, self.max_flares))
        
        # Check model has been compiled
        self.assertIsNotNone(self.model.model.optimizer)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        # Generate synthetic data
        X, y = self.model.generate_synthetic_data(n_samples=10, noise_level=0.05)
        
        # Check shapes
        self.assertEqual(X.shape, (10, self.sequence_length, self.n_features))
        self.assertEqual(y.shape, (10, self.sequence_length, self.max_flares))
        
        # Check that X is the sum of the components in y plus noise
        # For each sample, sum y across flare components
        for i in range(len(X)):
            y_sum = np.sum(y[i, :, :], axis=1).reshape(-1, 1)
            
            # Original should be approximately the sum of components (accounting for noise)
            self.assertLess(np.mean(np.abs(X[i] - y_sum)), 0.1)
    
    def test_reconstruct_flares(self):
        """Test flare reconstruction."""
        # Skip actual model prediction - just test the function interface
        # Normally you would use the trained model, but for testing we can mock the output
        def mock_predict(x):
            """Mock prediction function."""
            batch_size = x.shape[0]
            result = np.zeros((batch_size, self.sequence_length, self.max_flares))
            
            if batch_size >= 1:
                # First sample: single peak
                result[0, :, 0] = self.X_test[0, :, 0]
                
            if batch_size >= 2:
                # Second sample: two peaks
                peak1 = np.zeros((self.sequence_length, 1))
                peak2 = np.zeros((self.sequence_length, 1))
                
                x1 = np.linspace(-4, 2, self.sequence_length)
                x2 = np.linspace(-2, 4, self.sequence_length)
                peak1[:, 0] = 0.8 * np.exp(-x1**2)
                peak2[:, 0] = 0.6 * np.exp(-x2**2)
                
                result[1, :, 0] = peak1[:, 0]
                result[1, :, 1] = peak2[:, 0]
                
            return result
        
        # Patch the model's predict method for testing
        self.model.model.predict = mock_predict
        
        # Test reconstruction with first sample
        original, individual_flares, combined = reconstruct_flares(
            self.model, self.X_test[0:1], window_size=self.sequence_length, plot=False
        )
        
        # Check shapes
        self.assertEqual(original.shape, (1, self.sequence_length))
        self.assertEqual(individual_flares.shape, (self.sequence_length, self.max_flares))
        self.assertEqual(combined.shape, (1, self.sequence_length))
        
        # For the single peak, the first component should match the original
        self.assertLess(np.mean(np.abs(original[0] - individual_flares[:, 0])), 0.01)
        
        # Other components should be close to zero
        self.assertLess(np.max(individual_flares[:, 1]), 0.01)
        self.assertLess(np.max(individual_flares[:, 2]), 0.01)


if __name__ == '__main__':
    unittest.main()
