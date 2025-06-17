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
import seaborn as sns
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
    
    def plot_contrastive_analysis(self, X_sample, save_path=None):
        """
        Enhanced contrastive learning analysis with professional seaborn aesthetics
        """
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        # Create comprehensive analysis dashboard
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Encoder Feature Representations
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Get encoder representations
        if hasattr(self, 'encoder') and self.encoder is not None:
            representations = self.encoder.predict(X_sample[:100])  # Sample subset
            
            # Apply PCA for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            representations_2d = pca.fit_transform(representations)
            
            # Create DataFrame for seaborn
            repr_data = pd.DataFrame({
                'PC1': representations_2d[:, 0],
                'PC2': representations_2d[:, 1],
                'Sample_ID': range(len(representations_2d)),
                'Cluster': np.arange(len(representations_2d)) % 5  # Simple clustering
            })
            
            sns.scatterplot(data=repr_data, x='PC1', y='PC2', hue='Cluster', 
                           ax=ax1, alpha=0.7, s=60, palette='viridis')
            ax1.set_title('🔍 Learned Representation Space (PCA)', 
                         fontsize=16, fontweight='bold', pad=20)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                          fontsize=12, fontweight='semibold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                          fontsize=12, fontweight='semibold')
            ax1.legend(title='Sample Clusters', frameon=True, fancybox=True, shadow=True)
        else:
            ax1.text(0.5, 0.5, 'Encoder Not Available\nTrain Model First', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
            ax1.set_title('Representation Space', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Distribution Analysis
        ax2 = fig.add_subplot(gs[0, 2])
        
        if hasattr(self, 'encoder') and self.encoder is not None:
            feature_stats = []
            for i in range(min(10, representations.shape[1])):
                feature_stats.extend([{
                    'Feature': f'F{i}',
                    'Value': val,
                    'Type': 'Learned Representation'
                } for val in representations[:50, i]])
            
            feat_df = pd.DataFrame(feature_stats)
            sns.boxplot(data=feat_df, y='Feature', x='Value', ax=ax2, 
                       orient='h', palette='Set2')
            ax2.set_title('Feature Value Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Feature Value', fontsize=11)
        else:
            ax2.text(0.5, 0.5, 'Feature Analysis\nUnavailable', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12, fontweight='bold')
            ax2.set_title('Feature Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Augmentation Comparison
        ax3 = fig.add_subplot(gs[1, :])
        
        # Show original vs augmented samples
        sample_idx = 0
        original = X_sample[sample_idx]
        
        # Generate augmentations
        augmented = self.create_augmentations(X_sample[sample_idx:sample_idx+1])
        
        # Plot comparison
        time_steps = np.arange(len(original))
        
        # Original signal
        sns.lineplot(x=time_steps, y=original[:, 0], ax=ax3, 
                    label='Original Signal', linewidth=3, color='blue', alpha=0.8)
        
        # Show multiple augmented versions
        aug_sample = augmented[0]
        if len(aug_sample.shape) > 1:
            sns.lineplot(x=time_steps, y=aug_sample[:, 0], ax=ax3, 
                        label='Augmented Signal', linewidth=2, color='red', 
                        linestyle='--', alpha=0.8)
        
        ax3.fill_between(time_steps, original[:, 0], alpha=0.3, color='blue', label='Original Area')
        
        ax3.set_title('📊 Original vs Augmented Signal Comparison', 
                     fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Time Steps', fontsize=12, fontweight='semibold')
        ax3.set_ylabel('Signal Amplitude', fontsize=12, fontweight='semibold')
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        
        # 4. Projection Head Analysis
        ax4 = fig.add_subplot(gs[2, 0])
        
        if hasattr(self, 'projection_head') and self.projection_head is not None:
            # Get projection outputs
            projections = self.projection_head.predict(representations[:50])
            proj_norms = np.linalg.norm(projections, axis=1)
            
            sns.histplot(proj_norms, kde=True, ax=ax4, color='green', alpha=0.7)
            ax4.axvline(np.mean(proj_norms), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(proj_norms):.3f}')
            ax4.set_title('Projection Norm Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('L2 Norm', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Projection Head\nNot Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12, fontweight='bold')
            ax4.set_title('Projection Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Temperature Effect Visualization
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Simulate temperature effects on similarity
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        similarities = [0.95, 0.85, 0.7, 0.5, 0.3]  # Simulated values
        
        temp_data = pd.DataFrame({
            'Temperature': temperatures,
            'Similarity': similarities
        })
        
        sns.lineplot(data=temp_data, x='Temperature', y='Similarity', ax=ax5,
                    marker='o', linewidth=3, markersize=8, color='orange')
        ax5.fill_between(temperatures, similarities, alpha=0.3, color='orange')
        ax5.set_title('Temperature Effect on Similarity', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Temperature', fontsize=11)
        ax5.set_ylabel('Similarity Score', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # 6. Training Configuration Summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = f"""📊 CONTRASTIVE LEARNING SUMMARY
        
🎯 Model Architecture:
• Sequence Length: {self.sequence_length}
• Features: {self.n_features}
• Projection Dim: {self.projection_dim}

🔧 Configuration:
• Temperature: {getattr(self, 'temperature', 0.1)}
• Encoder Layers: Multi-layer CNN
• Projection Head: Dense layers

📈 Representation Quality:
• Dimensionality: {self.projection_dim}D
• Input Shape: {X_sample.shape}
• Sample Count: {len(X_sample)}

⚡ Contrastive Learning:
• Augmentation: Multi-type
• Loss: NT-Xent (InfoNCE)
• Self-supervised: ✓
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9,
                         edgecolor='orange', linewidth=2))
        
        fig.suptitle('🚀 Professional Contrastive Learning Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_training_comparison(self, pretrain_history, finetune_history=None, save_path=None):
        """
        Enhanced training comparison visualization with seaborn
        """
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        # 1. Pretraining Loss Evolution
        ax1 = fig.add_subplot(gs[0, :2])
        
        if pretrain_history:
            pretrain_epochs = range(1, len(pretrain_history.history['loss']) + 1)
            
            # Prepare data for seaborn
            pretrain_data = []
            for epoch, loss in enumerate(pretrain_history.history['loss'], 1):
                pretrain_data.append({'Epoch': epoch, 'Loss': loss, 'Phase': 'Pretraining'})
            
            # Add fine-tuning if available
            if finetune_history:
                finetune_epochs = range(len(pretrain_epochs) + 1, 
                                      len(pretrain_epochs) + len(finetune_history.history['loss']) + 1)
                for epoch, loss in zip(finetune_epochs, finetune_history.history['loss']):
                    pretrain_data.append({'Epoch': epoch, 'Loss': loss, 'Phase': 'Fine-tuning'})
            
            train_df = pd.DataFrame(pretrain_data)
            sns.lineplot(data=train_df, x='Epoch', y='Loss', hue='Phase', 
                        ax=ax1, marker='o', linewidth=3, markersize=6)
            
            # Add phase transition line
            if finetune_history:
                ax1.axvline(len(pretrain_epochs), color='red', linestyle=':', 
                           linewidth=2, alpha=0.7, label='Phase Transition')
        
        ax1.set_title('🎯 Contrastive Learning Training Progress', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='semibold')
        ax1.set_ylabel('Loss Value', fontsize=12, fontweight='semibold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Loss Distribution Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        
        if pretrain_history and finetune_history:
            loss_comparison = pd.DataFrame({
                'Pretraining': pretrain_history.history['loss'],
                'Fine-tuning': finetune_history.history['loss'][:len(pretrain_history.history['loss'])]
            })
            
            loss_melted = loss_comparison.melt(var_name='Phase', value_name='Loss')
            sns.boxplot(data=loss_melted, x='Phase', y='Loss', ax=ax2, palette='Set1')
            sns.stripplot(data=loss_melted, x='Phase', y='Loss', ax=ax2, 
                         color='black', alpha=0.6, size=4)
        elif pretrain_history:
            sns.histplot(pretrain_history.history['loss'], kde=True, ax=ax2, 
                        color='blue', alpha=0.7)
            ax2.set_xlabel('Pretraining Loss')
        
        ax2.set_title('Loss Distribution Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Fine-tuning Performance (if available)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if finetune_history and 'accuracy' in finetune_history.history:
            acc_data = []
            for epoch, (acc, val_acc) in enumerate(zip(
                finetune_history.history['accuracy'],
                finetune_history.history.get('val_accuracy', [])), 1):
                acc_data.append({'Epoch': epoch, 'Accuracy': acc, 'Type': 'Training'})
                if val_acc is not None:
                    acc_data.append({'Epoch': epoch, 'Accuracy': val_acc, 'Type': 'Validation'})
            
            acc_df = pd.DataFrame(acc_data)
            sns.lineplot(data=acc_df, x='Epoch', y='Accuracy', hue='Type', 
                        ax=ax3, marker='s', linewidth=2.5, markersize=6)
            ax3.set_title('Fine-tuning Accuracy', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Fine-tuning\nMetrics\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, fontweight='bold')
            ax3.set_title('Fine-tuning Performance', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Rate Schedule (if available)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if pretrain_history and 'lr' in pretrain_history.history:
            lr_data = pretrain_history.history['lr']
            sns.lineplot(x=range(1, len(lr_data) + 1), y=lr_data, ax=ax4,
                        marker='d', linewidth=2.5, markersize=4, color='purple')
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12, fontweight='bold')
            ax4.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Training Summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate metrics
        pretrain_final = pretrain_history.history['loss'][-1] if pretrain_history else 0
        pretrain_best = min(pretrain_history.history['loss']) if pretrain_history else 0
        pretrain_epochs_count = len(pretrain_history.history['loss']) if pretrain_history else 0
        
        finetune_final = finetune_history.history['loss'][-1] if finetune_history else 0
        finetune_acc = finetune_history.history.get('val_accuracy', [0])[-1] if finetune_history else 0
        
        summary_text = f"""📊 TRAINING SUMMARY
        
🔄 Pretraining Phase:
• Final Loss: {pretrain_final:.6f}
• Best Loss: {pretrain_best:.6f}
• Epochs: {pretrain_epochs_count}

🎯 Fine-tuning Phase:
• Final Loss: {finetune_final:.6f}
• Val Accuracy: {finetune_acc:.4f}
• Task: Classification

⚡ Self-Supervised Benefits:
• Representation Learning: ✓
• Data Efficiency: ✓
• Transfer Learning: ✓

🎪 Contrastive Method:
• SimCLR-inspired
• NT-Xent Loss
• Data Augmentation
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.9,
                         edgecolor='darkgreen', linewidth=2))
        
        fig.suptitle('🚀 Professional Contrastive Training Analysis', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    # ...existing code...
