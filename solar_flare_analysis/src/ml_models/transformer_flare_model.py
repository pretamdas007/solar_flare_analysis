"""
Advanced Transformer-based Solar Flare Detection and Solar Storm Prediction Model
Uses attention mechanisms for better temporal pattern recognition with advanced preprocessing
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
from scipy.signal import savgol_filter
from pybaselines import Baseline
import glob
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(layers.Layer):
    """
    Custom positional encoding layer with dynamic sequence length support
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


class SolarFlareStormDetector:
    """
    Advanced Transformer-based model for solar flare detection and solar storm prediction
    Uses multi-head attention to capture complex temporal dependencies with advanced preprocessing
    """
    
    def __init__(self, 
                 sequence_length: int = 256,
                 n_features: int = 2,
                 d_model: int = 128,
                 num_heads: int = 12,
                 num_transformer_blocks: int = 6,
                 ff_dim: int = 256,
                 dropout_rate: float = 0.15,
                 learning_rate: float = 0.0001,
                 loss_weights: Optional[Dict[str, float]] = None,
                 data_dir: str = 'c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis\\data\\XRS'):
        """
        Initialize Solar Flare and Storm Detection model
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences (increased for better context)
        n_features : int
            Number of input features (GOES-A and GOES-B channels)
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
        learning_rate : float
            Learning rate for optimizer
        loss_weights : Dict[str, float], optional
            Weights for multi-task losses
        data_dir : str
            Directory containing XRS data files
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.data_dir = data_dir
        self.loss_weights = loss_weights or {
            'flare_detection': 1.0,      # Binary flare/no-flare detection
            'flare_intensity': 0.8,      # Flare peak intensity prediction
            'storm_prediction': 1.2,     # Solar storm probability prediction
            'time_to_peak': 0.6          # Time until flare peak
        }
        
        self.model = None
        self.history = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.attention_weights = None
        self.baseline_corrector = Baseline()
        
        # Flare classification thresholds (GOES classification)
        self.flare_thresholds = {
            'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4
        }
    
    def build_model(self) -> keras.Model:
        """
        Build the multi-task transformer model for solar flare and storm prediction
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Positional encoding
        x = PositionalEncoding(self.sequence_length, self.n_features)(inputs)
        
        # Transformer blocks
        for i in range(self.num_transformer_blocks):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.n_features,
                name=f"transformer_attention_{i}"
            )(x, x)
            
            # Add & Norm
            x = layers.LayerNormalization(name=f"norm1_{i}")(x + attention_output)
            
            # Feed forward
            ff_output = layers.Dense(self.ff_dim, activation='relu', name=f"ff1_{i}")(x)
            ff_output = layers.Dropout(self.dropout_rate)(ff_output)
            ff_output = layers.Dense(self.n_features, name=f"ff2_{i}")(ff_output)
            
            # Add & Norm
            x = layers.LayerNormalization(name=f"norm2_{i}")(x + ff_output)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Outputs - 4 tasks for solar flare and storm prediction
        flare_detection = layers.Dense(1, activation='sigmoid', 
                                     name='flare_detection')(x)
        flare_intensity = layers.Dense(1, activation='linear', 
                                     name='flare_intensity')(x)
        storm_prediction = layers.Dense(1, activation='sigmoid', 
                                      name='storm_prediction')(x)
        time_to_peak = layers.Dense(1, activation='linear', 
                                  name='time_to_peak')(x)
        
        model = keras.Model(inputs=inputs, 
                           outputs=[flare_detection, flare_intensity, storm_prediction, time_to_peak])
        
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=self.learning_rate),
            loss={
                'flare_detection': 'binary_crossentropy',
                'flare_intensity': 'mse',
                'storm_prediction': 'binary_crossentropy',
                'time_to_peak': 'mse'
            },
            loss_weights={
                'flare_detection': 1.0,
                'flare_intensity': 0.5,
                'storm_prediction': 0.8,
                'time_to_peak': 0.3
            },
            metrics={
                'flare_detection': ['accuracy'],
                'flare_intensity': ['mae'],
                'storm_prediction': ['accuracy'],
                'time_to_peak': ['mae']
            }
        )
        
        self.model = model
        return model
        
    def load_and_preprocess_data(self, train_years: List[int] = [2023, 2024], 
                                test_years: List[int] = [2025]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load XRS data, apply baseline correction and smoothing
        
        Parameters
        ----------
        train_years : List[int]
            Years to use for training
        test_years : List[int] 
            Years to use for testing
            
        Returns
        -------
        X_train, y_train, X_test, y_test : np.ndarray
            Preprocessed training and testing data
        """
        print("🔄 Loading and preprocessing XRS data...")
        
        # Load data
        train_data = self._load_xrs_data(train_years)
        test_data = self._load_xrs_data(test_years)
        
        # Apply advanced preprocessing
        print("📊 Applying baseline correction and smoothing...")
        train_processed = self._advanced_preprocessing(train_data)
        test_processed = self._advanced_preprocessing(test_data)
        
        # Create sequences and labels
        print("🔄 Creating sequences and labels...")
        X_train, y_train = self._create_sequences_and_labels(train_processed)
        X_test, y_test = self._create_sequences_and_labels(test_processed)
        
        # Scale data
        print("📏 Scaling data...")
        X_train = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        y_test = self.scaler_y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        
        print(f"✅ Data preprocessing complete!")
        print(f"📈 Training data shape: {X_train.shape}")
        print(f"📉 Testing data shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def _load_xrs_data(self, years: List[int]) -> pd.DataFrame:
        """Load XRS data for specified years"""
        all_data = []
        
        for year in years:
            year_pattern = os.path.join(self.data_dir, f"*{year}*.csv")
            files = glob.glob(year_pattern)
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    
                    # Handle different time column formats
                    if 'time_tag' in df.columns:
                        df['time_tag'] = pd.to_datetime(df['time_tag'])
                    elif 'time_minutes' in df.columns and 'time_seconds' in df.columns:
                        # Convert time_minutes and time_seconds to datetime
                        # Assuming time_minutes is minutes since epoch start of year
                        df['time_tag'] = pd.to_datetime(f'{year}-01-01') + pd.to_timedelta(df['time_minutes'], unit='minutes')
                    
                    df['year'] = year
                    all_data.append(df)
                    print(f"📁 Loaded: {os.path.basename(file)} - Shape: {df.shape}")
                except Exception as e:
                    print(f"⚠️ Error loading {file}: {e}")
        
        if not all_data:
            raise ValueError(f"No data found for years {years} in {self.data_dir}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by time_tag if available
        if 'time_tag' in combined_data.columns:
            combined_data = combined_data.sort_values('time_tag').reset_index(drop=True)
        
        return combined_data
    
    def _advanced_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply advanced preprocessing: baseline correction + Savitzky-Golay smoothing
        Process each year's data separately to avoid length mismatches
        """
        processed_data = data.copy()
        
        # Channels to process - detect available column format
        if 'A_FLUX' in data.columns and 'B_FLUX' in data.columns:
            channels = ['A_FLUX', 'B_FLUX']
        elif 'xrsa' in data.columns and 'xrsb' in data.columns:
            channels = ['xrsa', 'xrsb']
        elif 'xrsa_flux_observed' in data.columns and 'xrsb_flux_observed' in data.columns:
            channels = ['xrsa_flux_observed', 'xrsb_flux_observed']
        else:
            raise ValueError(f"No recognized flux columns found. Available columns: {data.columns.tolist()}")
        
        # Process each year separately to avoid length mismatches
        if 'year' in data.columns:
            years = data['year'].unique()
            processed_parts = []
            
            for year in years:
                year_data = data[data['year'] == year].copy()
                print(f"📊 Processing {year} data: {len(year_data)} samples")
                
                for channel in channels:
                    if channel in year_data.columns:
                        flux_data = year_data[channel].values
                        
                        # Clean data: replace invalid values
                        flux_data = np.where(np.isfinite(flux_data) & (flux_data > 0), flux_data, 1e-10)
                        
                        # Convert to log scale for processing
                        log_flux = np.log10(flux_data + 1e-10)
                        
                        # Additional cleaning for baseline correction
                        log_flux = np.where(np.isfinite(log_flux), log_flux, -10.0)
                        
                        # Baseline correction using Asymmetric Least Squares (AsLS)
                        try:
                            baseline_corrected, _ = self.baseline_corrector.asls(
                                log_flux, 
                                lam=1e4,      # Smoothness parameter
                                p=0.001,      # Asymmetry parameter
                                max_iter=50
                            )
                            baseline_removed = log_flux - baseline_corrected
                            print(f"✅ Baseline correction applied to {channel} for {year}")
                        except Exception as e:
                            print(f"⚠️ Baseline correction failed for {channel} in {year}: {e}")
                            baseline_removed = log_flux
                        
                        # Savitzky-Golay smoothing
                        try:
                            # Choose window length based on data sampling
                            window_length = min(51, len(baseline_removed) // 4)
                            if window_length % 2 == 0:
                                window_length += 1
                            if window_length < 5:
                                window_length = 5
                            
                            smoothed = savgol_filter(
                                baseline_removed, 
                                window_length=window_length, 
                                polyorder=3,
                                mode='nearest'
                            )
                            print(f"✅ Savitzky-Golay smoothing applied to {channel} for {year}")
                        except Exception as e:
                            print(f"⚠️ Savitzky-Golay smoothing failed for {channel} in {year}: {e}")
                            smoothed = baseline_removed
                        
                        # Convert back to linear scale
                        year_data[f'{channel}_processed'] = 10**smoothed
                
                processed_parts.append(year_data)
            
            # Combine all processed years
            processed_data = pd.concat(processed_parts, ignore_index=True)
            
        else:
            # If no year column, process the entire dataset as one
            print("📊 Processing entire dataset (no year separation)")
            for channel in channels:
                if channel in processed_data.columns:
                    flux_data = processed_data[channel].values
                    
                    # Clean data: replace invalid values
                    flux_data = np.where(np.isfinite(flux_data) & (flux_data > 0), flux_data, 1e-10)
                    
                    # Convert to log scale for processing
                    log_flux = np.log10(flux_data + 1e-10)
                    
                    # Additional cleaning for baseline correction
                    log_flux = np.where(np.isfinite(log_flux), log_flux, -10.0)
                    
                    # Baseline correction using Asymmetric Least Squares (AsLS)
                    try:
                        baseline_corrected, _ = self.baseline_corrector.asls(
                            log_flux, 
                            lam=1e4,      # Smoothness parameter
                            p=0.001,      # Asymmetry parameter
                            max_iter=50
                        )
                        baseline_removed = log_flux - baseline_corrected
                        print(f"✅ Baseline correction applied to {channel}")
                    except Exception as e:
                        print(f"⚠️ Baseline correction failed for {channel}: {e}")
                        baseline_removed = log_flux
                    
                    # Savitzky-Golay smoothing
                    try:
                        # Choose window length based on data sampling
                        window_length = min(51, len(baseline_removed) // 4)
                        if window_length % 2 == 0:
                            window_length += 1
                        if window_length < 5:
                            window_length = 5
                        
                        smoothed = savgol_filter(
                            baseline_removed, 
                            window_length=window_length, 
                            polyorder=3,
                            mode='nearest'
                        )
                        print(f"✅ Savitzky-Golay smoothing applied to {channel}")
                    except Exception as e:
                        print(f"⚠️ Savitzky-Golay smoothing failed for {channel}: {e}")
                        smoothed = baseline_removed
                    
                    # Convert back to linear scale
                    processed_data[f'{channel}_processed'] = 10**smoothed
                
        return processed_data
    
    def _create_sequences_and_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences and multi-task labels for training
        """
        # Use processed channels if available, otherwise original
        channel_cols = [col for col in data.columns if 'processed' in col]
        if len(channel_cols) < 2:
            # Detect channel columns
            if 'A_FLUX' in data.columns and 'B_FLUX' in data.columns:
                channel_cols = ['A_FLUX', 'B_FLUX']
            elif 'xrsa' in data.columns and 'xrsb' in data.columns:
                channel_cols = ['xrsa', 'xrsb']
            elif 'xrsa_flux_observed' in data.columns and 'xrsb_flux_observed' in data.columns:
                channel_cols = ['xrsa_flux_observed', 'xrsb_flux_observed']
            else:
                raise ValueError(f"No recognized flux columns found. Available: {data.columns.tolist()}")
        
        # Create feature matrix
        features = []
        for col in channel_cols[:2]:  # Use first 2 channels
            if col in data.columns:
                features.append(data[col].values)
        
        if len(features) < 2:
            raise ValueError("Insufficient data channels found")
        
        feature_matrix = np.column_stack(features)
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(len(feature_matrix) - self.sequence_length - 24):  # 24 steps ahead prediction
            # Input sequence
            seq = feature_matrix[i:i + self.sequence_length]
            
            # Future data for labeling
            future_data = feature_matrix[i + self.sequence_length:i + self.sequence_length + 24]
            
            # Multi-task labels
            label = self._create_multitask_labels(seq, future_data)
            
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def _create_multitask_labels(self, current_seq: np.ndarray, future_data: np.ndarray) -> np.ndarray:
        """
        Create multi-task labels: flare detection, intensity, storm prediction, time to peak
        """
        # Use B channel (index 1) for primary analysis
        current_flux = current_seq[:, 1]
        future_flux = future_data[:, 1]
        
        # 1. Flare Detection (binary)
        current_max = np.max(current_flux)
        future_max = np.max(future_flux)
        threshold = 1e-6  # C-class threshold
        flare_detected = float(future_max > threshold and future_max > current_max * 2)
        
        # 2. Flare Intensity (continuous)
        flare_intensity = np.log10(future_max + 1e-10)
        
        # 3. Storm Prediction (probability)
        # Higher intensity flares more likely to cause storms
        storm_probability = min(1.0, max(0.0, (np.log10(future_max + 1e-10) + 6) / 2))
        
        # 4. Time to Peak (if flare detected)
        time_to_peak = 0.0
        if flare_detected > 0.5:
            peak_idx = np.argmax(future_flux)
            time_to_peak = float(peak_idx) / 24.0  # Normalized time
        
        return np.array([flare_detected, flare_intensity, storm_probability, time_to_peak])
    
    def visualize_preprocessing(self, data: pd.DataFrame, channel: str = None, 
                               start_idx: int = 0, length: int = 2000, save_path: str = None):
        """
        Visualize preprocessing steps with professional aesthetics
        """
        # Auto-detect the best channel to use if not specified
        if channel is None:
            if 'B_FLUX' in data.columns:
                channel = 'B_FLUX'
            elif 'xrsb' in data.columns:
                channel = 'xrsb'
            elif 'xrsb_flux_observed' in data.columns:
                channel = 'xrsb_flux_observed'
            else:
                # Use the first flux column available
                flux_cols = [col for col in data.columns if 'flux' in col.lower() or 'xrs' in col.lower()]
                channel = flux_cols[0] if flux_cols else data.columns[1]
        
        # Set professional theme
        plt.style.use('default')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), facecolor='white')
        fig.suptitle('🌟 Advanced Solar Flare Data Preprocessing Pipeline', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Extract data segment
        end_idx = start_idx + length
        time_data = data['time_tag'].iloc[start_idx:end_idx] if 'time_tag' in data.columns else range(length)
        
        # Original data
        if channel not in data.columns:
            print(f"⚠️ Channel '{channel}' not found. Available: {data.columns.tolist()}")
            return
            
        original_channel = channel
        original_data = data[original_channel].iloc[start_idx:end_idx]
        
        # Processed data
        processed_channel = f'{original_channel}_processed'
        processed_data = data[processed_channel].iloc[start_idx:end_idx] if processed_channel in data.columns else original_data
        
        # 1. Original vs Processed Comparison
        ax1 = axes[0, 0]
        ax1.semilogy(time_data, original_data, alpha=0.7, linewidth=1.5, 
                    label='Original Data', color='lightcoral')
        ax1.semilogy(time_data, processed_data, linewidth=2, 
                    label='Baseline Corrected + Smoothed', color='darkblue')
        ax1.set_title('📊 Original vs Processed Data', fontsize=14, fontweight='bold')
        ax1.set_ylabel('X-ray Flux (W/m²)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Log-scale Difference
        ax2 = axes[0, 1]
        log_diff = np.log10(processed_data + 1e-10) - np.log10(original_data + 1e-10)
        ax2.plot(time_data, log_diff, color='green', linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_title('📈 Log-scale Preprocessing Effect', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Log₁₀(Processed/Original)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution Comparison
        ax3 = axes[1, 0]
        sns.histplot(np.log10(original_data + 1e-10), alpha=0.6, 
                    label='Original', color='lightcoral', ax=ax3, stat='density')
        sns.histplot(np.log10(processed_data + 1e-10), alpha=0.8, 
                    label='Processed', color='darkblue', ax=ax3, stat='density')
        ax3.set_title('📊 Data Distribution Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Log₁₀(X-ray Flux)', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.legend()
        
        # 4. Flare Event Enhancement
        ax4 = axes[1, 1]
        # Find potential flare events (peaks above threshold)
        threshold = np.percentile(processed_data, 95)
        flare_events = processed_data > threshold
        
        ax4.semilogy(time_data, processed_data, color='darkblue', linewidth=2, alpha=0.8)
        ax4.semilogy(time_data[flare_events], processed_data[flare_events], 
                    'ro', markersize=4, alpha=0.8, label='Potential Flares')
        ax4.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, label='95th Percentile')
        ax4.set_title('🔥 Enhanced Flare Event Detection', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time', fontsize=12)
        ax4.set_ylabel('X-ray Flux (W/m²)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def run_complete_pipeline(self, train_years=[2023, 2024], test_years=[2025], 
                             epochs=1, batch_size=64, save_results=True):
        """
        Run the complete solar flare and storm prediction pipeline
        """
        print("🌟 Starting Solar Flare & Storm Prediction Pipeline...")
        print("=" * 60)
        
        # 1. Load and preprocess data
        X_train, y_train, X_test, y_test = self.load_and_preprocess_data(train_years, test_years)
        
        # 2. Create validation split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train[:, 0].astype(int)
        )
        
        print(f"📊 Final data shapes:")
        print(f"   Training: {X_train.shape}")
        print(f"   Validation: {X_val.shape}")
        print(f"   Testing: {X_test.shape}")
        
        # 3. Build and train model
        print("\n🚀 Building and training model...")
        self.build_model()
        print(f"📋 Model summary:")
        print(f"   Parameters: {self.model.count_params():,}")
        
        # 4. Train the model
        history = self.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
        
        # 5. Evaluate model
        print("\n📊 Evaluating model performance...")
        predictions = self.evaluate_model(X_test, y_test, 
                                        save_path='evaluation_dashboard.png' if save_results else None)
        
        # 6. Save model and results
        if save_results:
            print("\n💾 Saving model and results...")
            self.save_model('solar_flare_storm_model.h5')
            
            # Plot training history
            self.plot_training_history(history, save_path='training_history.png')
            
            # Visualize attention patterns
            self.visualize_attention(X_test[:5], save_path='attention_analysis.png')
        
        print("\n✅ Pipeline completed successfully!")
        return history, predictions
    
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
            self.ff_dim, activation="gelu", name=f"{name_prefix}_ffn1"
        )(attention_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(
            self.d_model, name=f"{name_prefix}_ffn2"
        )(ffn_output)
        
        ffn_output = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_ln2"
        )(attention_output + ffn_output)
        
        return ffn_output
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """
        Save the trained model and scalers
        
        Parameters
        ----------
        model_path : str
            Path to save the model
        scaler_path : str, optional
            Path to save the scalers (default: model_path with _scalers suffix)
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        self.model.save(model_path)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scalers.pkl')
        
        import pickle
        scalers_dict = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers_dict, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scalers saved to: {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Load a trained model and scalers
        
        Parameters
        ----------
        model_path : str
            Path to the saved model
        scaler_path : str, optional
            Path to the saved scalers (default: model_path with _scalers suffix)
        """
        self.model = keras.models.load_model(model_path)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scalers.pkl')
        
        import pickle
        try:
            with open(scaler_path, 'rb') as f:
                scalers_dict = pickle.load(f)
                self.scaler_X = scalers_dict['scaler_X']
                self.scaler_y = scalers_dict['scaler_y']
            print(f"Model loaded from: {model_path}")
            print(f"Scalers loaded from: {scaler_path}")
        except FileNotFoundError:
            print(f"Warning: Scalers file not found at {scaler_path}")
            print("Using default scalers - ensure to fit them on appropriate data before prediction")


# Keep the existing TransformerFlareModel and ConvolutionalTransformerModel classes
        """
        Build the transformer model architecture for solar flare and storm prediction
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features), name='xrs_input')
        
        # Input projection to d_model dimensions
        x = layers.Dense(self.d_model, name='input_projection')(inputs)
        
        # Positional encoding
        x = PositionalEncoding(self.sequence_length, self.d_model, name='positional_encoding')(x)
        
        # Transformer blocks
        for i in range(self.num_transformer_blocks):
            x = self.create_transformer_block(x, name_prefix=f"transformer_block_{i}")
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='global_pooling')(x)
        
        # Dense layers for feature extraction
        x = layers.Dense(256, activation='gelu', name='dense_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='gelu', name='dense_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Multi-task outputs
        flare_detection = layers.Dense(1, activation='sigmoid', name='flare_detection')(x)
        flare_intensity = layers.Dense(1, activation='linear', name='flare_intensity')(x)
        storm_prediction = layers.Dense(1, activation='sigmoid', name='storm_prediction')(x)
        time_to_peak = layers.Dense(1, activation='sigmoid', name='time_to_peak')(x)
        
        model = keras.Model(
            inputs=inputs, 
            outputs=[flare_detection, flare_intensity, storm_prediction, time_to_peak],
            name='SolarFlareStormDetector'
        )
        
        # Compile with multiple losses
        model.compile(
            optimizer=optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=1e-4
            ),
            loss={
                'flare_detection': 'binary_crossentropy',
                'flare_intensity': 'mse',
                'storm_prediction': 'binary_crossentropy',
                'time_to_peak': 'mse'
            },
            loss_weights=self.loss_weights,
            metrics={
                'flare_detection': ['accuracy', 'precision', 'recall'],
                'flare_intensity': ['mae'],
                'storm_prediction': ['accuracy', 'precision', 'recall'],
                'time_to_peak': ['mae']
            }
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=1, batch_size=64, verbose=1):
        """
        Train the solar flare and storm prediction model
        """
        if self.model is None:
            self.build_model()
        
        # Prepare multi-task labels
        y_train_dict = {
            'flare_detection': y_train[:, 0],
            'flare_intensity': y_train[:, 1],
            'storm_prediction': y_train[:, 2],
            'time_to_peak': y_train[:, 3]
        }
        
        y_val_dict = {
            'flare_detection': y_val[:, 0],
            'flare_intensity': y_val[:, 1],
            'storm_prediction': y_val[:, 2],
            'time_to_peak': y_val[:, 3]
        }
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=1e-8,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_solar_flare_storm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(
                'training_log.csv',
                append=True
            )
        ]
        
        # Train model
        print("🚀 Starting training...")
        self.history = self.model.fit(
            X_train, y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        print("✅ Training completed!")
        return self.history
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """
        Save the trained model and scaler
        
        Parameters
        ----------
        model_path : str
            Path to save the model
        scaler_path : str, optional
            Path to save the scaler (default: model_path with _scaler suffix)
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        self.model.save(model_path)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
        
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler_X, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Load a trained model and scaler
        
        Parameters
        ----------
        model_path : str
            Path to the saved model
        scaler_path : str, optional
            Path to the saved scaler (default: model_path with _scaler suffix)
        """
        self.model = keras.models.load_model(model_path)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
        
        import pickle
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler_X = pickle.load(f)
            print(f"Model loaded from: {model_path}")
            print(f"Scaler loaded from: {scaler_path}")
        except FileNotFoundError:
            print(f"Warning: Scaler file not found at {scaler_path}")
            print("Using default scaler - ensure to fit it on appropriate data before prediction")
    
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
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 loss_weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid CNN-Transformer model
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        n_classes : int
            Number of flare classes
        conv_filters : list
            List of filter sizes for convolutional layers
        d_model : int
            Dimension of transformer embeddings
        num_heads : int
            Number of attention heads
        num_transformer_blocks : int
            Number of transformer blocks
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        loss_weights : Dict[str, float], optional
            Weights for multi-task losses
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.conv_filters = conv_filters
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights or {
            'flare_class': 1.0,
            'flare_magnitude': 0.5
        }
        
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
        
        # Positional encoding using unified layer
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)
        
        # Transformer blocks
        for i in range(self.num_transformer_blocks):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model,
                name=f"conv_transformer_attention_{i}"
            )(x, x)
            
            attention_output = layers.Dropout(self.dropout_rate)(attention_output)
            attention_output = layers.LayerNormalization(
                epsilon=1e-6
            )(x + attention_output)
            
            # Feed-forward
            ffn_output = layers.Dense(256, activation="relu")(attention_output)
            ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
            ffn_output = layers.Dense(self.d_model)(ffn_output)
            
            x = layers.LayerNormalization(
                epsilon=1e-6
            )(attention_output + ffn_output)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Outputs - 4 tasks for solar flare and storm prediction
        flare_detection = layers.Dense(1, activation='sigmoid', 
                                     name='flare_detection')(x)
        flare_intensity = layers.Dense(1, activation='linear', 
                                     name='flare_intensity')(x)
        storm_prediction = layers.Dense(1, activation='sigmoid', 
                                      name='storm_prediction')(x)
        time_to_peak = layers.Dense(1, activation='linear', 
                                  name='time_to_peak')(x)
        
        model = keras.Model(inputs=inputs, 
                           outputs=[flare_detection, flare_intensity, storm_prediction, time_to_peak])
        
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=self.learning_rate),
            loss={
                'flare_detection': 'binary_crossentropy',
                'flare_intensity': 'mse',
                'storm_prediction': 'binary_crossentropy',
                'time_to_peak': 'mse'
            },
            loss_weights={
                'flare_detection': 1.0,
                'flare_intensity': 0.5,
                'storm_prediction': 0.8,
                'time_to_peak': 0.3
            },
            metrics={
                'flare_detection': ['accuracy'],
                'flare_intensity': ['mae'],
                'storm_prediction': ['accuracy'],
                'time_to_peak': ['mae']
            }
        )
        
        self.model = model
        return model
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """
        Save the trained model and scaler
        
        Parameters
        ----------
        model_path : str
            Path to save the model
        scaler_path : str, optional
            Path to save the scaler (default: model_path with _scaler suffix)
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        self.model.save(model_path)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
        
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler_X, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Load a trained model and scaler
        
        Parameters
        ----------
        model_path : str
            Path to the saved model
        scaler_path : str, optional
            Path to the saved scaler (default: model_path with _scaler suffix)
        """
        self.model = keras.models.load_model(model_path)
        
        if scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
        
        import pickle
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler_X = pickle.load(f)
            print(f"Model loaded from: {model_path}")
            print(f"Scaler loaded from: {scaler_path}")
        except FileNotFoundError:
            print(f"Warning: Scaler file not found at {scaler_path}")
            print("Using default scaler - ensure to fit it on appropriate data before prediction")
