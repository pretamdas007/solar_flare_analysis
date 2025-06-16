"""
Self-Supervised Learning Models for Solar Flare Analysis
Uses contrastive learning and masked autoencoding for representation learning
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ContrastiveLearningModel:
    """
    Contrastive learning model for solar flare representation learning
    Uses SimCLR-style contrastive learning
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 projection_dim: int = 128,
                 temperature: float = 0.1):
        """
        Initialize contrastive learning model
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        projection_dim : int
            Dimension of projection head
        temperature : float
            Temperature parameter for contrastive loss
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.encoder = None
        self.projection_head = None
        self.contrastive_model = None
        self.classifier = None
        self.scaler_X = RobustScaler()
    
    def create_augmentations(self, x):
        """
        Create augmented versions of input data using Keras operations
        """
        # Define augmentation functions as Lambda layers
        def add_noise(inputs):
            noise_factor = 0.1
            noise_shape = keras.ops.shape(inputs)
            noise = keras.random.normal(noise_shape) * noise_factor
            return inputs + noise
        
        def time_masking(inputs):
            mask_prob = 0.15
            mask_shape = keras.ops.shape(inputs)
            mask = keras.random.uniform(mask_shape) > mask_prob
            return keras.ops.where(mask, inputs, 0.0)
        
        def amplitude_scaling(inputs):
            scale_factor = keras.random.uniform((), minval=0.8, maxval=1.2)
            return inputs * scale_factor
        
        def time_shift(inputs):
            # Simple time shifting by circular shift
            shift_max = keras.ops.shape(inputs)[1] // 8  # Max shift is 1/8 of sequence length
            shift = keras.random.uniform((), minval=-shift_max, maxval=shift_max, dtype='int32')
            return keras.ops.roll(inputs, shift, axis=1)
        
        # Randomly select and apply augmentations
        # For deterministic behavior in functional API, we'll apply a combination
        # Remove explicit names to avoid conflicts when called multiple times
        x_aug1 = layers.Lambda(add_noise)(x)
        x_aug2 = layers.Lambda(time_masking)(x_aug1)
        x_aug3 = layers.Lambda(amplitude_scaling)(x_aug2)
        
        return x_aug3
    
    def build_encoder(self):
        """
        Build the encoder network
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # CNN backbone
        x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        representations = layers.Dense(256, activation='relu', name='representations')(x)
        
        encoder = keras.Model(inputs, representations, name='encoder')
        self.encoder = encoder
        return encoder
    
    def build_projection_head(self):
        """
        Build projection head for contrastive learning
        """
        inputs = layers.Input(shape=(256,))
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        projections = layers.Dense(self.projection_dim, activation=None, 
                                 name='projections')(x)
        
        projection_head = keras.Model(inputs, projections, name='projection_head')
        self.projection_head = projection_head
        return projection_head
    def contrastive_loss(self, projections_1, projections_2):
        """
        Compute contrastive loss using Keras operations
        """
        # Normalize projections using Keras operations
        # L2 normalize along the last axis
        norm_1 = keras.ops.sqrt(keras.ops.sum(keras.ops.square(projections_1), axis=1, keepdims=True))
        projections_1 = projections_1 / (norm_1 + 1e-8)
        
        norm_2 = keras.ops.sqrt(keras.ops.sum(keras.ops.square(projections_2), axis=1, keepdims=True))
        projections_2 = projections_2 / (norm_2 + 1e-8)
        
        # Compute similarities using Keras operations
        similarities = keras.ops.matmul(projections_1, keras.ops.transpose(projections_2)) / self.temperature
        
        # Get batch size dynamically
        batch_size = keras.ops.shape(projections_1)[0]
        
        # Compute positive similarities (diagonal elements)
        positive_similarities = keras.ops.diagonal(similarities)
        
        # For each sample, compute the contrastive loss
        # Numerator: exp(positive similarity)
        numerator = keras.ops.exp(positive_similarities)
        
        # Denominator: sum of exp(all similarities) for each row
        denominator = keras.ops.sum(keras.ops.exp(similarities), axis=1)
        
        # Contrastive loss: -log(numerator/denominator)
        loss_1 = -keras.ops.log(numerator / (denominator + 1e-8))
        
        # Symmetric loss for the other direction (transpose)
        similarities_t = keras.ops.transpose(similarities)
        positive_similarities_t = keras.ops.diagonal(similarities_t)
        numerator_2 = keras.ops.exp(positive_similarities_t)
        denominator_2 = keras.ops.sum(keras.ops.exp(similarities_t), axis=1)
        loss_2 = -keras.ops.log(numerator_2 / (denominator_2 + 1e-8))
          # Return mean of both directions
        return keras.ops.mean(loss_1 + loss_2) / 2.0
    
    def build_contrastive_model(self):
        """
        Build the full contrastive learning model
        """
        if self.encoder is None:
            self.build_encoder()
        if self.projection_head is None:
            self.build_projection_head()
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Create augmented views
        augmented_1 = self.create_augmentations(inputs)
        augmented_2 = self.create_augmentations(inputs)
        
        # Get representations and projections
        representations_1 = self.encoder(augmented_1)
        representations_2 = self.encoder(augmented_2)
        
        projections_1 = self.projection_head(representations_1)
        projections_2 = self.projection_head(representations_2)
        
        # Create a custom layer to compute contrastive loss
        class ContrastiveLossLayer(layers.Layer):
            def __init__(self, temperature, **kwargs):
                super().__init__(**kwargs)
                self.temperature = temperature
            
            def call(self, inputs):
                projections_1, projections_2 = inputs
                
                # L2 normalize projections using Keras operations
                norm_1 = keras.ops.sqrt(keras.ops.sum(keras.ops.square(projections_1), axis=1, keepdims=True))
                projections_1_norm = projections_1 / (norm_1 + 1e-8)
                
                norm_2 = keras.ops.sqrt(keras.ops.sum(keras.ops.square(projections_2), axis=1, keepdims=True))
                projections_2_norm = projections_2 / (norm_2 + 1e-8)
                
                # Compute similarities using Keras operations
                similarities = keras.ops.matmul(projections_1_norm, keras.ops.transpose(projections_2_norm)) / self.temperature
                
                # Compute positive similarities (diagonal elements)
                positive_similarities = keras.ops.diagonal(similarities)
                
                # For each sample, compute the contrastive loss
                # Numerator: exp(positive similarity)
                numerator = keras.ops.exp(positive_similarities)
                
                # Denominator: sum of exp(all similarities) for each row
                denominator = keras.ops.sum(keras.ops.exp(similarities), axis=1)
                
                # Contrastive loss: -log(numerator/denominator)
                loss_1 = -keras.ops.log(numerator / (denominator + 1e-8))
                
                # Symmetric loss for the other direction (transpose)
                similarities_t = keras.ops.transpose(similarities)
                positive_similarities_t = keras.ops.diagonal(similarities_t)
                numerator_2 = keras.ops.exp(positive_similarities_t)
                denominator_2 = keras.ops.sum(keras.ops.exp(similarities_t), axis=1)
                loss_2 = -keras.ops.log(numerator_2 / (denominator_2 + 1e-8))
                
                # Return mean of both directions
                loss = keras.ops.mean(loss_1 + loss_2) / 2.0
                self.add_loss(loss)
                
                # Return projections unchanged
                return [projections_1, projections_2]
        
        # Apply contrastive loss layer
        loss_layer = ContrastiveLossLayer(temperature=self.temperature)
        outputs = loss_layer([projections_1, projections_2])
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001))
        
        self.contrastive_model = model
        return model
    
    def pretrain(self, X_train, epochs=100, batch_size=32, verbose=1):
        """
        Pretrain using contrastive learning
        """
        if self.contrastive_model is None:
            self.build_contrastive_model()
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(
            X_train.reshape(-1, self.n_features)
        ).reshape(X_train.shape)
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='loss', patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=10, min_lr=1e-7
            )
        ]
        
        # Train
        history = self.contrastive_model.fit(
            X_scaled, epochs=epochs, batch_size=batch_size,
            callbacks=callbacks_list, verbose=verbose
        )
        
        return history
    
    def build_classifier(self, n_classes: int = 6):
        """
        Build classifier on top of pretrained encoder
        """
        if self.encoder is None:
            raise ValueError("Must pretrain encoder first")
        
        # Freeze encoder weights
        for layer in self.encoder.layers:
            layer.trainable = False
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        representations = self.encoder(inputs)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(representations)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        classifier = keras.Model(inputs, outputs, name='classifier')
        classifier.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.classifier = classifier
        return classifier
    
    def fine_tune(self, X_train, y_train, X_val, y_val, 
                  n_classes=6, epochs=50, batch_size=32, verbose=1):
        """
        Fine-tune classifier on labeled data
        """
        if self.classifier is None:
            self.build_classifier(n_classes)
        
        # Scale data
        X_train_scaled = self.scaler_X.transform(
            X_train.reshape(-1, self.n_features)
        ).reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(
            X_val.reshape(-1, self.n_features)
        ).reshape(X_val.shape)
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'best_contrastive_classifier.h5',
                monitor='val_accuracy', save_best_only=True
            )
        ]
        
        # Train
        history = self.classifier.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks_list, verbose=verbose
        )
        
        return history


class MaskedAutoencoderModel:
    """
    Masked Autoencoder for self-supervised representation learning
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 encoder_dim: int = 256,
                 decoder_dim: int = 128,
                 mask_ratio: float = 0.25):
        """
        Initialize Masked Autoencoder
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        encoder_dim : int
            Encoder hidden dimension
        decoder_dim : int
            Decoder hidden dimension
        mask_ratio : float
            Ratio of sequence to mask
        """
        self.sequence_length = sequence_length
        self.n_features = n_features        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio
        
        self.autoencoder = None
        self.encoder = None
        self.classifier = None
        self.scaler_X = RobustScaler()


class RandomMaskingLayer(layers.Layer):
    """
    Custom layer for random masking of input sequences
    """
    
    def __init__(self, mask_ratio=0.15, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
    
    def call(self, x):
        """
        Apply random masking to input sequences using Keras operations
        """
        batch_size = keras.ops.shape(x)[0]
        seq_len = keras.ops.shape(x)[1]
        
        # Number of tokens to mask - ensure it's an integer
        len_mask = keras.ops.cast(
            keras.ops.cast(seq_len, 'float32') * self.mask_ratio, 'int32'
        )
        
        # Generate random indices for masking
        noise = keras.random.uniform([batch_size, seq_len])
        ids_shuffle = keras.ops.argsort(noise, axis=1)
        
        # Create mask - start with all True (unmasked)
        mask = keras.ops.ones([batch_size, seq_len], dtype='bool')
        
        # For simplicity, use a different masking approach compatible with Keras ops
        # Create a mask based on shuffled indices
        mask_indices = ids_shuffle[:, :len_mask]
        
        # Apply masking by setting masked positions to False
        # Using a simpler approach: threshold-based masking
        threshold = self.mask_ratio
        random_values = keras.random.uniform(keras.ops.shape(x)[:2])
        mask = random_values > threshold
        
        # Apply mask
        masked_x = keras.ops.where(
            keras.ops.expand_dims(mask, -1),
            x,
            keras.ops.zeros_like(x)
        )
        
        return masked_x
    
    def compute_output_shape(self, input_shape):
        return input_shape


class MaskedAutoencoderModel:
    """
    Masked Autoencoder for self-supervised representation learning
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 mask_ratio: float = 0.15,
                 encoder_dim: int = 256,
                 decoder_dim: int = 128):
        """
        Initialize Masked Autoencoder model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.mask_ratio = mask_ratio
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler_X = RobustScaler()
    
    def random_masking(self, x):
        """
        Random masking of input sequences - replaced by custom layer
        """
        # This method is now replaced by RandomMaskingLayer
        pass
    
    def build_autoencoder(self):
        """
        Build the masked autoencoder model
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Apply masking using custom layer
        masked_inputs = RandomMaskingLayer(mask_ratio=self.mask_ratio)(inputs)
        
        # Encoder
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(masked_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Dense(self.encoder_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        x = layers.Dense(self.decoder_dim, activation='relu')(encoded)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        decoded = layers.Dense(self.n_features, activation='linear', name='decoded')(x)
        
        # Standard reconstruction loss (masking loss is handled internally by the layer)
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Create separate encoder model
        encoder = keras.Model(inputs, encoded, name='mae_encoder')
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        return autoencoder
    
    def pretrain(self, X_train, epochs=100, batch_size=32, verbose=1):
        """
        Pretrain the masked autoencoder
        """
        if self.autoencoder is None:
            self.build_autoencoder()
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(
            X_train.reshape(-1, self.n_features)
        ).reshape(X_train.shape)
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='loss', patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=10, min_lr=1e-7
            )
        ]
        
        # Train
        history = self.autoencoder.fit(
            X_scaled, epochs=epochs, batch_size=batch_size,
            callbacks=callbacks_list, verbose=verbose
        )
        
        return history
    
    def build_classifier(self, n_classes: int = 6):
        """
        Build classifier using pretrained encoder
        """
        if self.encoder is None:
            raise ValueError("Must pretrain encoder first")
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Get encoder features (without masking)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        features = layers.Dense(self.encoder_dim, activation='relu')(x)
        
        # Global pooling
        pooled = layers.GlobalAveragePooling1D()(features)
        
        # Classification head
        x = layers.Dense(256, activation='relu')(pooled)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        classifier = keras.Model(inputs, outputs, name='mae_classifier')
        classifier.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.classifier = classifier
        return classifier
