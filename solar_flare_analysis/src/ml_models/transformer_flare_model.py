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
        Enhanced attention visualization with professional seaborn aesthetics
        """
        attention_weights = self.extract_attention_weights(X_sample)
        
        if len(attention_weights) == 0:
            print("No attention weights found")
            return
        
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        # Create comprehensive attention analysis
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        # 1. Main Attention Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        attention = attention_weights[0][sample_idx]
        
        # Enhanced heatmap with professional styling
        sns.heatmap(attention, ax=ax1, cmap='viridis', cbar=True,
                   square=True, linewidths=0.1, linecolor='white',
                   cbar_kws={'label': 'Attention Weight', 'shrink': 0.8})
        ax1.set_title('🔍 Transformer Attention Pattern Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Key Position (Time Steps)', fontsize=12, fontweight='semibold')
        ax1.set_ylabel('Query Position (Time Steps)', fontsize=12, fontweight='semibold')
        
        # 2. Attention Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        attention_flat = attention.flatten()
        sns.histplot(attention_flat, kde=True, ax=ax2, color='skyblue', alpha=0.7)
        ax2.set_title('Attention Weight Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Attention Weight', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Average Attention by Position
        ax3 = fig.add_subplot(gs[1, 0])
        avg_attention = np.mean(attention, axis=0)
        positions = np.arange(len(avg_attention))
        
        sns.lineplot(x=positions, y=avg_attention, ax=ax3, 
                    marker='o', linewidth=2.5, markersize=4, color='coral')
        ax3.fill_between(positions, avg_attention, alpha=0.3, color='coral')
        ax3.set_title('Average Attention by Position', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Position', fontsize=11)
        ax3.set_ylabel('Average Attention', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Attention Head Comparison (if multiple heads)
        ax4 = fig.add_subplot(gs[1, 1])
        if len(attention_weights) > 1:
            head_data = []
            for i, head_attention in enumerate(attention_weights[:4]):  # Show first 4 heads
                head_avg = np.mean(head_attention[sample_idx])
                head_std = np.std(head_attention[sample_idx])
                head_data.append({'Head': f'Head {i+1}', 'Mean': head_avg, 'Std': head_std})
            
            head_df = pd.DataFrame(head_data)
            sns.barplot(data=head_df, x='Head', y='Mean', ax=ax4, palette='Set2')
            ax4.set_title('Attention Head Comparison', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Mean Attention Weight', fontsize=11)
        else:
            ax4.text(0.5, 0.5, 'Single Head\nTransformer', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            ax4.set_title('Attention Configuration', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Temporal Focus Analysis
        ax5 = fig.add_subplot(gs[1, 2])
        # Calculate attention focus (how much attention is concentrated)
        attention_entropy = -np.sum(attention * np.log(attention + 1e-8), axis=1)
        
        sns.lineplot(x=np.arange(len(attention_entropy)), y=attention_entropy, 
                    ax=ax5, marker='s', linewidth=2.5, markersize=4, color='darkgreen')
        ax5.fill_between(np.arange(len(attention_entropy)), attention_entropy, 
                        alpha=0.3, color='darkgreen')
        ax5.set_title('Attention Entropy (Focus)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Query Position', fontsize=11)
        ax5.set_ylabel('Entropy (bits)', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # Add comprehensive title
        fig.suptitle('🚀 Professional Transformer Attention Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Enhanced training history visualization with seaborn
        """
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle('🎯 Transformer Training History Dashboard', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Prepare data for seaborn
        epochs = range(1, len(history.history['loss']) + 1)
        
        # 1. Loss Evolution
        loss_data = []
        for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], 
                                                          history.history.get('val_loss', [])), 1):
            loss_data.append({'Epoch': epoch, 'Loss': train_loss, 'Type': 'Training'})
            if val_loss is not None:
                loss_data.append({'Epoch': epoch, 'Loss': val_loss, 'Type': 'Validation'})
        
        loss_df = pd.DataFrame(loss_data)
        sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Type', 
                    ax=axes[0,0], marker='o', linewidth=2.5, markersize=6)
        axes[0,0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Accuracy Evolution (if available)
        if 'accuracy' in history.history:
            acc_data = []
            for epoch, acc in enumerate(history.history['accuracy'], 1):
                acc_data.append({'Epoch': epoch, 'Accuracy': acc, 'Type': 'Training'})
            if 'val_accuracy' in history.history:
                for epoch, acc in enumerate(history.history['val_accuracy'], 1):
                    acc_data.append({'Epoch': epoch, 'Accuracy': acc, 'Type': 'Validation'})
            
            acc_df = pd.DataFrame(acc_data)
            sns.lineplot(data=acc_df, x='Epoch', y='Accuracy', hue='Type', 
                        ax=axes[0,1], marker='s', linewidth=2.5, markersize=6)
            axes[0,1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        else:
            axes[0,1].text(0.5, 0.5, 'Accuracy\nNot Available', ha='center', va='center',
                          transform=axes[0,1].transAxes, fontsize=12, fontweight='bold')
            axes[0,1].set_title('Accuracy Metrics', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Learning Rate (if available)
        if 'lr' in history.history:
            sns.lineplot(x=epochs, y=history.history['lr'], ax=axes[1,0], 
                        marker='d', linewidth=2.5, markersize=6, color='orange')
            axes[1,0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].set_yscale('log')
        else:
            axes[1,0].text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center',
                          transform=axes[1,0].transAxes, fontsize=12, fontweight='bold')
            axes[1,0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Training Summary
        axes[1,1].axis('off')
        summary_text = f"""📊 TRAINING SUMMARY
        
🏆 Final Metrics:
• Training Loss: {history.history['loss'][-1]:.4f}
• Validation Loss: {history.history.get('val_loss', [0])[-1]:.4f}
• Total Epochs: {len(history.history['loss'])}

🎯 Best Performance:
• Min Train Loss: {min(history.history['loss']):.4f}
• Min Val Loss: {min(history.history.get('val_loss', [999])):.4f}

⚡ Model Configuration:
• Architecture: Multi-head Transformer
• Sequence Length: {self.sequence_length}
• Features: {self.n_features}
• Heads: {self.num_heads}
        """
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_model_predictions(self, X_test, y_test, predictions, save_path=None):
        """
        Enhanced prediction analysis with seaborn
        """
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        # 1. Prediction vs True Values
        ax1 = fig.add_subplot(gs[0, :2])
        pred_data = pd.DataFrame({
            'True': y_test.flatten(),
            'Predicted': predictions.flatten(),
            'Sample': range(len(y_test.flatten()))
        })
        
        sns.scatterplot(data=pred_data, x='True', y='Predicted', ax=ax1, 
                       alpha=0.6, s=50, color='skyblue', edgecolor='navy', linewidth=0.5)
        
        # Add perfect prediction line
        min_val, max_val = min(pred_data['True'].min(), pred_data['Predicted'].min()), \
                          max(pred_data['True'].max(), pred_data['Predicted'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_title('🎯 Prediction vs True Values', fontsize=16, fontweight='bold')
        ax1.set_xlabel('True Values', fontsize=12, fontweight='semibold')
        ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='semibold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        residuals = predictions.flatten() - y_test.flatten()
        sns.histplot(residuals, kde=True, ax=ax2, color='coral', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Residuals', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample Predictions Time Series
        ax3 = fig.add_subplot(gs[1, :])
        sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            time_steps = np.arange(X_test.shape[1])
            
            # Plot input sequence
            ax3.plot(time_steps, X_test[idx, :, 0], 
                    alpha=0.7, linewidth=2, label=f'Sample {i+1} Input')
            
        ax3.set_title('🔍 Sample Input Sequences Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time Steps', fontsize=12, fontweight='semibold')
        ax3.set_ylabel('Input Values', fontsize=12, fontweight='semibold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle('🚀 Professional Transformer Prediction Analysis', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
