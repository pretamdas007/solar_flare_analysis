"""
Advanced Transformer-based Solar Flare Analysis Model
Uses attention mechanisms for better temporal pattern recognition
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(layers.Layer):
    """
    Custom positional encoding layer to avoid KerasTensor issues
    """
    
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=d_model
        )
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embeddings = self.embedding(positions)
        # Broadcast to batch dimension
        position_embeddings = tf.expand_dims(position_embeddings, 0)
        position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])
        return x + position_embeddings


class ConvolutionalPositionalEncoding(layers.Layer):
    """
    Positional encoding for convolutional transformer with variable sequence length
    """
    
    def __init__(self, max_sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.embedding = layers.Embedding(
            input_dim=max_sequence_length, output_dim=d_model
        )
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = self.embedding(positions)
        # Broadcast to batch dimension
        position_embeddings = tf.expand_dims(position_embeddings, 0)
        position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])
        return x + position_embeddings


class TransformerFlareModel:
    """
    Transformer-based model for solar flare detection and classification
    Uses multi-head attention to capture complex temporal dependencies
    """
    
    def __init__(self, 
                 sequence_length: int = 128,
                 n_features: int = 2,
                 n_classes: int = 6,
                 d_model: int = 64,
                 num_heads: int = 8,
                 num_transformer_blocks: int = 4,
                 ff_dim: int = 128,
                 dropout_rate: float = 0.1):
        """
        Initialize Transformer model
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        n_classes : int
            Number of flare classes
        d_model : int
            Dimension of model embeddings
        num_heads : int
            Number of attention heads
        num_transformer_blocks : int
            Number of transformer blocks
        ff_dim : int
            Feed-forward dimension
        dropout_rate : float
            Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        self.scaler_X = RobustScaler()
        self.attention_weights = None
        
    def create_transformer_block(self, inputs, name_prefix="transformer"):
        """
        Create a transformer block with multi-head attention
        """
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            name=f"{name_prefix}_attention"
        )(inputs, inputs)
        
        attention_output = layers.Dropout(self.dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_ln1"
        )(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(
            self.ff_dim, activation="relu", name=f"{name_prefix}_ffn1"
        )(attention_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(
            self.d_model, name=f"{name_prefix}_ffn2"
        )(ffn_output)
        
        ffn_output = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_ln2"
        )(attention_output + ffn_output)
        
        return ffn_output
    
    def build_model(self) -> keras.Model:
        """
        Build the transformer model architecture
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
          # Input projection to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)
        
        # Positional encoding using custom layer
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)
        
        # Transformer blocks
        for i in range(self.num_transformer_blocks):
            x = self.create_transformer_block(x, name_prefix=f"transformer_{i}")
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Multi-task outputs
        flare_class = layers.Dense(self.n_classes, activation='softmax', name='flare_class')(x)
        flare_intensity = layers.Dense(1, activation='linear', name='flare_intensity')(x)
        flare_duration = layers.Dense(1, activation='relu', name='flare_duration')(x)
        
        model = keras.Model(inputs=inputs, 
                           outputs=[flare_class, flare_intensity, flare_duration])
        
        # Compile with multiple losses
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001),
            loss={
                'flare_class': 'sparse_categorical_crossentropy',
                'flare_intensity': 'mse',
                'flare_duration': 'mse'
            },
            loss_weights={
                'flare_class': 1.0,
                'flare_intensity': 0.5,
                'flare_duration': 0.3
            },
            metrics={
                'flare_class': ['accuracy'],
                'flare_intensity': ['mae'],
                'flare_duration': ['mae']
            }
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the transformer model
        """
        if self.model is None:
            self.build_model()
        
        # Prepare callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'best_transformer_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        return self.history
    
    def extract_attention_weights(self, X_sample):
        """
        Extract attention weights for interpretability
        """
        # Create a model that outputs attention weights
        attention_model = keras.Model(
            inputs=self.model.input,
            outputs=[layer.output for layer in self.model.layers 
                    if 'attention' in layer.name]
        )
        
        attention_outputs = attention_model.predict(X_sample)
        return attention_outputs
    
    def visualize_attention(self, X_sample, sample_idx=0, save_path=None):
        """
        Visualize attention patterns
        """
        attention_weights = self.extract_attention_weights(X_sample)
        
        if len(attention_weights) == 0:
            print("No attention weights found")
            return
        
        # Plot attention weights for the first transformer block
        attention = attention_weights[0][sample_idx]  # Shape: [seq_len, seq_len]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention, cmap='Blues', cbar=True)
        plt.title('Transformer Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ConvolutionalTransformerModel:
    """
    Hybrid CNN-Transformer model for solar flare analysis
    Combines convolutional feature extraction with transformer attention
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 n_classes: int = 6,
                 conv_filters: list = [32, 64, 128],
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_transformer_blocks: int = 2):
        """
        Initialize hybrid CNN-Transformer model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.conv_filters = conv_filters
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        
        self.model = None
        self.history = None
        self.scaler_X = RobustScaler()
    
    def build_model(self) -> keras.Model:
        """
        Build the hybrid CNN-Transformer model
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Convolutional feature extraction
        x = inputs
        for i, filters in enumerate(self.conv_filters):
            x = layers.Conv1D(
                filters, kernel_size=3, padding='same',
                activation='relu', name=f'conv1d_{i}'
            )(x)
            x = layers.BatchNormalization()(x)
            if i < len(self.conv_filters) - 1:  # Don't pool on last layer
                x = layers.MaxPooling1D(pool_size=2)(x)
          # Project to transformer dimension
        x = layers.Dense(self.d_model)(x)
        
        # Positional encoding using custom layer
        x = ConvolutionalPositionalEncoding(self.sequence_length, self.d_model)(x)
        
        # Transformer blocks
        for i in range(self.num_transformer_blocks):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model,
                name=f"conv_transformer_attention_{i}"
            )(x, x)
            
            attention_output = layers.Dropout(0.1)(attention_output)
            attention_output = layers.LayerNormalization(
                epsilon=1e-6
            )(x + attention_output)
            
            # Feed-forward
            ffn_output = layers.Dense(256, activation="relu")(attention_output)
            ffn_output = layers.Dropout(0.1)(ffn_output)
            ffn_output = layers.Dense(self.d_model)(ffn_output)
            
            x = layers.LayerNormalization(
                epsilon=1e-6
            )(attention_output + ffn_output)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Outputs
        flare_class = layers.Dense(self.n_classes, activation='softmax', 
                                  name='flare_class')(x)
        flare_magnitude = layers.Dense(1, activation='linear', 
                                      name='flare_magnitude')(x)
        
        model = keras.Model(inputs=inputs, 
                           outputs=[flare_class, flare_magnitude])
        
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001),
            loss={
                'flare_class': 'sparse_categorical_crossentropy',
                'flare_magnitude': 'mse'
            },
            loss_weights={'flare_class': 1.0, 'flare_magnitude': 0.5},
            metrics={
                'flare_class': ['accuracy'],
                'flare_magnitude': ['mae']
            }
        )
        
        self.model = model
        return model
