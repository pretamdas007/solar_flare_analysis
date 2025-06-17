"""
Enhanced Machine Learning models for advanced solar flare analysis
Includes nanoflare detection, energy estimation, and statistical analysis
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats, optimize
from scipy.signal import find_peaks, peak_widths, peak_prominences
import warnings
warnings.filterwarnings('ignore')


class NanoflareDetector:
    """
    Specialized detector for identifying nanoflares in solar data
    """
    
    def __init__(self, min_energy_threshold=1e-9, alpha_threshold=2.0):
        """
        Initialize nanoflare detector
        
        Parameters
        ----------
        min_energy_threshold : float
            Minimum energy threshold for nanoflare detection
        alpha_threshold : float
            Alpha parameter threshold (|α| > 2 implies nano-flares)
        """
        self.min_energy_threshold = min_energy_threshold
        self.alpha_threshold = alpha_threshold
        self.scaler = RobustScaler()
        
    def detect_nanoflares(self, data, sampling_rate=1.0):
        """
        Detect nanoflares in time series data
        
        Parameters
        ----------
        data : array-like
            Time series data
        sampling_rate : float
            Sampling rate of the data (Hz)
            
        Returns
        -------
        dict
            Dictionary containing detected nanoflares and their properties
        """
        # Preprocess data
        data_smooth = self._preprocess_data(data)
        
        # Find potential nanoflare events
        peaks, properties = self._find_nanoflare_candidates(data_smooth, sampling_rate)
        
        # Calculate alpha parameter for each event
        alpha_values = self._calculate_alpha_parameter(data_smooth, peaks, properties)
        
        # Filter based on alpha threshold
        nanoflare_mask = np.abs(alpha_values) > self.alpha_threshold
        nanoflare_peaks = peaks[nanoflare_mask]
        nanoflare_alpha = alpha_values[nanoflare_mask]
        
        # Calculate energies
        energies = self._estimate_nanoflare_energies(data_smooth, nanoflare_peaks, properties)
        
        return {
            'peaks': nanoflare_peaks,
            'alpha_values': nanoflare_alpha,
            'energies': energies,
            'properties': properties,
            'total_count': len(nanoflare_peaks),
            'total_energy': np.sum(energies) if len(energies) > 0 else 0
        }
    
    def _preprocess_data(self, data):
        """Preprocess data for nanoflare detection"""
        # Remove trend
        detrended = signal.detrend(data)
          # Apply smoothing filter
        butter_result = signal.butter(4, 0.1, btype='low')
        b, a = butter_result[0], butter_result[1]
        smoothed = signal.filtfilt(b, a, detrended)
        
        return smoothed
    
    def _find_nanoflare_candidates(self, data, sampling_rate):
        """Find candidate nanoflare events"""
        # Calculate prominence threshold based on data statistics
        data_std = np.std(data)
        prominence_threshold = 2 * data_std
        
        # Find peaks with minimum prominence
        peaks, properties = find_peaks(
            data,
            prominence=prominence_threshold,
            width=1,
            distance=int(sampling_rate * 10)  # Minimum 10 seconds between peaks
        )
        
        return peaks, properties
    
    def _calculate_alpha_parameter(self, data, peaks, properties):
        """Calculate alpha parameter for energy distribution"""
        alpha_values = []
        
        for peak in peaks:
            # Get local region around peak
            window = min(50, len(data) // 10)
            start = max(0, peak - window)
            end = min(len(data), peak + window)
            
            local_data = data[start:end]
            local_energies = self._calculate_local_energies(local_data)
            
            # Fit power law to energy distribution
            alpha = self._fit_power_law(local_energies)
            alpha_values.append(alpha)
        
        return np.array(alpha_values)
    
    def _calculate_local_energies(self, data):
        """Calculate local energy distribution"""
        # Use sliding window to calculate energy values
        window_size = 5
        energies = []
        
        for i in range(len(data) - window_size + 1):
            window_data = data[i:i + window_size]
            energy = np.sum(window_data**2)
            energies.append(energy)
        
        return np.array(energies)
    
    def _fit_power_law(self, energies):
        """Fit power law to energy distribution"""
        try:
            # Remove zero and negative values
            positive_energies = energies[energies > 0]
            if len(positive_energies) < 3:
                return 0.0
            
            # Log-linear fit
            log_energies = np.log10(positive_energies)
            log_counts = np.log10(np.arange(1, len(positive_energies) + 1))
              # Linear regression in log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_energies, log_counts)
            
            return float(-slope)  # Alpha is negative slope
        except:
            return 0.0
    
    def _estimate_nanoflare_energies(self, data, peaks, properties):
        """Estimate energy of detected nanoflares"""
        energies = []
        
        for i, peak in enumerate(peaks):
            # Get peak width
            if 'widths' in properties:
                width = properties['widths'][i]
            else:
                width = 10  # Default width
            
            # Calculate energy in the peak region
            start = max(0, int(peak - width))
            end = min(len(data), int(peak + width))
            
            peak_data = data[start:end]
            baseline = np.median(data)  # Simple baseline estimation
            
            # Energy as integral above baseline
            energy = np.sum(np.maximum(0, peak_data - baseline))
            energies.append(energy)
        
        return np.array(energies)


class EnhancedFlareDecompositionModel:
    """
    Enhanced neural network model for advanced flare decomposition and analysis
    """
    
    def __init__(self, sequence_length=256, n_features=2, max_flares=5, 
                 dropout_rate=0.3, attention_units=64):
        """
        Initialize the enhanced flare decomposition model
        
        Parameters
        ----------
        sequence_length : int
            Length of input time series sequences
        n_features : int
            Number of input features (e.g., XRS-A, XRS-B channels)
        max_flares : int
            Maximum number of overlapping flares to decompose
        dropout_rate : float
            Dropout rate for regularization
        attention_units : int
            Number of units in attention mechanism
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.max_flares = max_flares
        self.dropout_rate = dropout_rate
        self.attention_units = attention_units
        self.model = None
        self.history = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.nanoflare_detector = NanoflareDetector()
        
    def build_enhanced_model(self):
        """
        Build enhanced neural network with attention mechanism and multi-output
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Feature extraction with residual connections
        x = self._build_feature_extractor(inputs)
        
        # Attention mechanism
        x = self._build_attention_layer(x)
        
        # Multi-task outputs
        flare_params = self._build_flare_parameter_head(x)
        energy_estimates = self._build_energy_estimation_head(x)
        classification = self._build_classification_head(x)
        
        # Create model
        self.model = models.Model(
            inputs=inputs,
            outputs={
                'flare_params': flare_params,
                'energy_estimates': energy_estimates,
                'classification': classification
            }
        )
          # Compile with multiple losses
        self.model.compile(
            optimizer='adam',
            loss={
                'flare_params': 'mse',
                'energy_estimates': 'mse',
                'classification': 'binary_crossentropy'
            },
            loss_weights={
                'flare_params': 1.0,
                'energy_estimates': 0.5,
                'classification': 0.3
            },
            metrics={
                'flare_params': ['mae'],
                'energy_estimates': ['mae'],
                'classification': ['accuracy']
            }
        )
        
        return self.model
    
    def _build_feature_extractor(self, inputs):
        """Build feature extraction layers with residual connections"""
        # First convolutional block
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual blocks
        for filters in [128, 256, 512]:
            x = self._residual_block(x, filters)
        
        return x
    
    def _residual_block(self, x, filters):
        """Residual block for better gradient flow"""
        shortcut = x
        
        # First conv layer
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second conv layer
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def _build_attention_layer(self, x):
        """Build attention mechanism for focusing on relevant time steps"""
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=self.attention_units,
            dropout=self.dropout_rate
        )(x, x)
        
        # Add & norm
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff = layers.Dense(self.attention_units * 4, activation='relu')(x)
        ff = layers.Dropout(self.dropout_rate)(ff)
        ff = layers.Dense(x.shape[-1])(ff)
        
        # Add & norm
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
        
        return x
    
    def _build_flare_parameter_head(self, x):
        """Build head for flare parameter prediction"""
        x_flare = layers.GlobalAveragePooling1D()(x)
        x_flare = layers.Dense(256, activation='relu')(x_flare)
        x_flare = layers.Dropout(self.dropout_rate)(x_flare)
        x_flare = layers.Dense(128, activation='relu')(x_flare)
        x_flare = layers.Dropout(self.dropout_rate)(x_flare)
        
        # Output: amplitude, peak_time, rise_time, decay_time, background for each flare
        flare_params = layers.Dense(
            self.max_flares * 5,
            activation='linear',
            name='flare_params'
        )(x_flare)
        
        return flare_params
    
    def _build_energy_estimation_head(self, x):
        """Build head for energy estimation"""
        x_energy = layers.GlobalMaxPooling1D()(x)
        x_energy = layers.Dense(128, activation='relu')(x_energy)
        x_energy = layers.Dropout(self.dropout_rate)(x_energy)
        x_energy = layers.Dense(64, activation='relu')(x_energy)
        
        # Output: energy estimate for each flare
        energy_estimates = layers.Dense(
            self.max_flares,
            activation='relu',
            name='energy_estimates'
        )(x_energy)
        
        return energy_estimates
    
    def _build_classification_head(self, x):
        """Build head for flare classification (nanoflare vs regular)"""
        x_class = layers.GlobalAveragePooling1D()(x)
        x_class = layers.Dense(64, activation='relu')(x_class)
        x_class = layers.Dropout(self.dropout_rate)(x_class)
        
        # Output: probability of containing nanoflares
        classification = layers.Dense(
            1,
            activation='sigmoid',
            name='classification'
        )(x_class)
        
        return classification
    
    def generate_enhanced_synthetic_data(self, n_samples=2000, noise_level=0.05):
        """
        Generate enhanced synthetic data with realistic flare characteristics
        """
        X = np.zeros((n_samples, self.sequence_length, self.n_features))
        y_params = np.zeros((n_samples, self.max_flares * 5))
        y_energy = np.zeros((n_samples, self.max_flares))
        y_class = np.zeros((n_samples, 1))
        
        # Time array
        t = np.linspace(0, 1, self.sequence_length)
        
        for i in range(n_samples):
            # Randomly decide flare characteristics
            n_flares = np.random.randint(1, self.max_flares + 1)
            has_nanoflares = np.random.random() < 0.3  # 30% chance of nanoflares
            
            combined_signal = np.zeros((self.sequence_length, self.n_features))
            
            for j in range(n_flares):
                # Generate flare parameters
                if has_nanoflares and j >= n_flares - 2:
                    # Generate nanoflare parameters
                    amplitude = np.random.uniform(0.01, 0.1)
                    rise_time = np.random.uniform(0.005, 0.02)
                    decay_time = np.random.uniform(0.01, 0.05)
                else:
                    # Generate regular flare parameters
                    amplitude = np.random.uniform(0.1, 1.0)
                    rise_time = np.random.uniform(0.02, 0.1)
                    decay_time = np.random.uniform(0.05, 0.3)
                
                peak_pos = np.random.uniform(0.2, 0.8)
                background = np.random.uniform(0.0, 0.05)
                
                # Store parameters
                y_params[i, j*5:(j+1)*5] = [amplitude, peak_pos, rise_time, decay_time, background]
                
                # Generate flare profile for both channels
                for ch in range(self.n_features):
                    # Channel-specific scaling
                    ch_amplitude = amplitude * (1.0 if ch == 0 else 0.7)
                    flare_profile = self._generate_realistic_flare_profile(
                        t, peak_pos, ch_amplitude, rise_time, decay_time, background
                    )
                    combined_signal[:, ch] += flare_profile
                
                # Calculate energy (simplified)
                energy = amplitude * (rise_time + decay_time) * 1000
                y_energy[i, j] = energy
            
            # Add correlated noise between channels
            noise = self._generate_correlated_noise(noise_level)
            combined_signal += noise
            
            # Store data
            X[i] = combined_signal
            y_class[i, 0] = 1.0 if has_nanoflares else 0.0
        
        return X, {
            'flare_params': y_params,
            'energy_estimates': y_energy,
            'classification': y_class
        }
    
    def _generate_realistic_flare_profile(self, t, peak_pos, amplitude, rise_time, decay_time, background):
        """Generate realistic flare profile with proper physics"""
        peak_idx = int(peak_pos * len(t))
        profile = np.zeros_like(t)
        
        for k, time_val in enumerate(t):
            if k <= peak_idx:
                # Exponential rise
                profile[k] = amplitude * (1 - np.exp(-(peak_idx - k) / (rise_time * len(t))))
            else:
                # Exponential decay
                profile[k] = amplitude * np.exp(-(k - peak_idx) / (decay_time * len(t)))
        
        # Add background and small variations
        profile += background
        profile += np.random.normal(0, amplitude * 0.01, len(profile))
        
        return profile
    
    def _generate_correlated_noise(self, noise_level):
        """Generate correlated noise between channels"""
        # Base noise
        base_noise = np.random.normal(0, noise_level, self.sequence_length)
        
        # Create correlated noise matrix
        noise = np.zeros((self.sequence_length, self.n_features))
        correlation = 0.8  # High correlation between XRS channels
        
        for ch in range(self.n_features):
            independent_noise = np.random.normal(0, noise_level, self.sequence_length)
            noise[:, ch] = correlation * base_noise + (1 - correlation) * independent_noise
        
        return noise
    
    def train_enhanced_model(self, X, y_dict, validation_split=0.2, epochs=150, 
                           batch_size=32, patience=15):
        """
        Train the enhanced model with multiple outputs
        """
        # Prepare data
        X_scaled = self._prepare_input_data(X, fit_scaler=True)
        y_scaled = self._prepare_target_data(y_dict, fit_scaler=True)
        
        # Define callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='best_enhanced_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_scaled,
            y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def _prepare_input_data(self, X, fit_scaler=False):
        """Prepare input data for training/prediction"""
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
        
        return X_scaled.reshape(original_shape)
    
    def _prepare_target_data(self, y_dict, fit_scaler=False):
        """Prepare target data for training"""
        y_scaled = {}
        
        for key, y_data in y_dict.items():
            if key == 'classification':
                # No scaling for binary classification
                y_scaled[key] = y_data
            else:
                if fit_scaler:
                    scaler = StandardScaler()
                    y_scaled[key] = scaler.fit_transform(y_data)
                    setattr(self, f'scaler_{key}', scaler)
                else:
                    scaler = getattr(self, f'scaler_{key}')
                    y_scaled[key] = scaler.transform(y_data)
        
        return y_scaled
    
    def predict_enhanced(self, X):
        """Make enhanced predictions with the model"""
        X_scaled = self._prepare_input_data(X)
        predictions = self.model.predict(X_scaled)
        
        # Unscale predictions
        predictions_unscaled = {}
        for key, pred in predictions.items():
            if key == 'classification':
                predictions_unscaled[key] = pred
            else:
                scaler = getattr(self, f'scaler_{key}')
                predictions_unscaled[key] = scaler.inverse_transform(pred)
        
        return predictions_unscaled
    
    def analyze_flare_statistics(self, predictions, time_series_length):
        """
        Perform statistical analysis of detected flares
        """
        flare_params = predictions['flare_params']
        energies = predictions['energy_estimates']
        
        statistics = {
            'total_flares': 0,
            'nanoflare_count': 0,
            'energy_distribution': [],
            'frequency_analysis': {},
            'power_law_fit': {}
        }
        
        # Count valid flares (amplitude > threshold)
        amplitude_threshold = 0.05
        for i in range(flare_params.shape[0]):
            for j in range(0, flare_params.shape[1], 5):
                amplitude = flare_params[i, j]
                if amplitude > amplitude_threshold:
                    statistics['total_flares'] += 1
                    
                    # Check if it's a nanoflare
                    energy = energies[i, j//5]
                    if energy < 1e-6:  # Energy threshold for nanoflares
                        statistics['nanoflare_count'] += 1
                    
                    statistics['energy_distribution'].append(energy)
        
        # Analyze energy distribution
        if len(statistics['energy_distribution']) > 0:
            energies_array = np.array(statistics['energy_distribution'])
            
            # Fit power law
            try:
                alpha, x_min = self._fit_power_law_distribution(energies_array)
                statistics['power_law_fit'] = {
                    'alpha': alpha,
                    'x_min': x_min,
                    'is_nanoflare_dominated': abs(alpha) > 2.0
                }
            except:
                statistics['power_law_fit'] = {'alpha': None, 'x_min': None}
            
            # Calculate frequency
            observation_time = time_series_length / 3600  # Assume hourly data
            statistics['frequency_analysis'] = {
                'flares_per_hour': statistics['total_flares'] / observation_time,
                'nanoflares_per_hour': statistics['nanoflare_count'] / observation_time,
                'mean_energy': np.mean(energies_array),
                'total_energy': np.sum(energies_array)
            }
        
        return statistics
    
    def _fit_power_law_distribution(self, energies):
        """Fit power law distribution to energy data"""
        # Remove zeros and sort
        positive_energies = energies[energies > 0]
        positive_energies = np.sort(positive_energies)
        
        if len(positive_energies) < 10:
            raise ValueError("Not enough data points for power law fit")
        
        # Use maximum likelihood estimation
        x_min = np.min(positive_energies)
        
        # Calculate alpha using MLE
        n = len(positive_energies)
        alpha = 1 + n / np.sum(np.log(positive_energies / x_min))
        
        return alpha, x_min
    def plot_enhanced_training_history(self):
        """
        Enhanced training history visualization with professional seaborn aesthetics
        """
        if self.history is None:
            print("No training history available.")
            return None
        
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        # Create comprehensive training dashboard
        fig = plt.figure(figsize=(22, 16), facecolor='white')
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        epochs = range(1, len(self.history.history['loss']) + 1)
        
        # 1. Total Loss Evolution
        ax1 = fig.add_subplot(gs[0, :2])
        loss_data = []
        for epoch, (train_loss, val_loss) in enumerate(zip(
            self.history.history['loss'],
            self.history.history.get('val_loss', [])), 1):
            loss_data.append({'Epoch': epoch, 'Loss': train_loss, 'Type': 'Training'})
            if val_loss is not None:
                loss_data.append({'Epoch': epoch, 'Loss': val_loss, 'Type': 'Validation'})
        
        loss_df = pd.DataFrame(loss_data)
        sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Type', 
                    ax=ax1, marker='o', linewidth=3, markersize=6, alpha=0.8)
        ax1.set_title('🎯 Total Training Loss Evolution', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='semibold')
        ax1.set_ylabel('Loss Value', fontsize=12, fontweight='semibold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Multi-task Loss Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Prepare multi-task loss data
        multitask_data = []
        task_losses = {
            'Flare Params': 'flare_params_loss',
            'Energy Est.': 'energy_estimates_loss', 
            'Classification': 'classification_loss'
        }
        
        for task_name, loss_key in task_losses.items():
            if loss_key in self.history.history:
                for epoch, loss in enumerate(self.history.history[loss_key], 1):
                    multitask_data.append({'Epoch': epoch, 'Loss': loss, 'Task': task_name})
        
        if multitask_data:
            multitask_df = pd.DataFrame(multitask_data)
            sns.lineplot(data=multitask_df, x='Epoch', y='Loss', hue='Task', 
                        ax=ax2, marker='s', linewidth=2.5, markersize=5)
            ax2.set_title('📊 Multi-task Loss Breakdown', fontsize=16, fontweight='bold')
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'Multi-task\nLoss Data\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, fontweight='bold')
            ax2.set_title('Multi-task Loss Analysis', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        
        # 3. Validation Metrics Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        
        val_metrics_data = []
        val_metrics = {
            'Total Loss': 'val_loss',
            'Flare Params': 'val_flare_params_loss',
            'Energy Est.': 'val_energy_estimates_loss',
            'Classification': 'val_classification_loss'
        }
        
        for metric_name, metric_key in val_metrics.items():
            if metric_key in self.history.history:
                for epoch, value in enumerate(self.history.history[metric_key], 1):
                    val_metrics_data.append({'Epoch': epoch, 'Value': value, 'Metric': metric_name})
        
        if val_metrics_data:
            val_df = pd.DataFrame(val_metrics_data)
            sns.lineplot(data=val_df, x='Epoch', y='Value', hue='Metric', 
                        ax=ax3, marker='d', linewidth=2.5, markersize=5, alpha=0.8)
            ax3.set_title('📈 Validation Metrics Evolution', fontsize=16, fontweight='bold')
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'Validation\nMetrics\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, fontweight='bold')
            ax3.set_title('Validation Metrics', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        
        # 4. Classification Performance
        ax4 = fig.add_subplot(gs[1, 2:])
        
        if 'classification_accuracy' in self.history.history:
            acc_data = []
            for epoch, (train_acc, val_acc) in enumerate(zip(
                self.history.history['classification_accuracy'],
                self.history.history.get('val_classification_accuracy', [])), 1):
                acc_data.append({'Epoch': epoch, 'Accuracy': train_acc, 'Type': 'Training'})
                if val_acc is not None:
                    acc_data.append({'Epoch': epoch, 'Accuracy': val_acc, 'Type': 'Validation'})
            
            acc_df = pd.DataFrame(acc_data)
            sns.lineplot(data=acc_df, x='Epoch', y='Accuracy', hue='Type', 
                        ax=ax4, marker='o', linewidth=3, markersize=6, alpha=0.8)
            ax4.set_title('🎯 Classification Accuracy Progress', fontsize=16, fontweight='bold')
            ax4.set_ylim(0, 1.05)
        else:
            ax4.text(0.5, 0.5, 'Classification\nAccuracy\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, fontweight='bold')
            ax4.set_title('Classification Performance', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=True, fancybox=True, shadow=True)
        
        # 5. Learning Rate and Optimization
        ax5 = fig.add_subplot(gs[2, 0])
        
        if 'lr' in self.history.history:
            sns.lineplot(x=epochs, y=self.history.history['lr'], ax=ax5,
                        marker='v', linewidth=3, markersize=6, color='purple')
            ax5.fill_between(epochs, self.history.history['lr'], alpha=0.3, color='purple')
            ax5.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Learning Rate')
            ax5.set_yscale('log')
        else:
            ax5.text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12, fontweight='bold')
            ax5.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Loss Distribution Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        
        final_losses = []
        loss_types = []
        for task_name, loss_key in task_losses.items():
            if loss_key in self.history.history:
                final_losses.append(self.history.history[loss_key][-1])
                loss_types.append(task_name)
        
        if final_losses:
            loss_dist_df = pd.DataFrame({'Task': loss_types, 'Final_Loss': final_losses})
            sns.barplot(data=loss_dist_df, x='Task', y='Final_Loss', ax=ax6, palette='viridis')
            ax6.set_title('Final Loss by Task', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Final Loss Value')
            ax6.set_yscale('log')
            
            # Add value labels on bars
            for i, v in enumerate(final_losses):
                ax6.text(i, v * 1.1, f'{v:.4f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
        else:
            ax6.text(0.5, 0.5, 'Loss Distribution\nNot Available', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=12, fontweight='bold')
            ax6.set_title('Loss Distribution', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Training Progress Summary
        ax7 = fig.add_subplot(gs[2, 2:])
        ax7.axis('off')
        
        # Calculate comprehensive statistics
        final_loss = self.history.history['loss'][-1]
        best_loss = min(self.history.history['loss'])
        best_epoch = self.history.history['loss'].index(best_loss) + 1
        total_epochs = len(self.history.history['loss'])
        
        final_val_loss = self.history.history.get('val_loss', [0])[-1] if self.history.history.get('val_loss') else 0
        final_accuracy = self.history.history.get('classification_accuracy', [0])[-1] if self.history.history.get('classification_accuracy') else 0
        
        summary_text = f"""📊 ENHANCED FLARE ANALYSIS TRAINING SUMMARY
        
🏆 Overall Performance:
• Final Training Loss: {final_loss:.6f}
• Best Training Loss: {best_loss:.6f} (Epoch {best_epoch})
• Final Val Loss: {final_val_loss:.6f}
• Total Epochs: {total_epochs}

🎯 Multi-task Results:
• Classification Accuracy: {final_accuracy:.4f}
• Flare Detection: Advanced CNN
• Energy Estimation: Regression Head
• Parameter Extraction: Multi-output

⚡ Model Configuration:
• Architecture: Enhanced Multi-task CNN
• Tasks: Classification + Regression
• Nanoflare Detection: ✓
• Energy Analysis: ✓

📈 Training Quality:
• Loss Improvement: {(self.history.history['loss'][0] / final_loss):.2f}x
• Convergence: {'Good' if best_epoch < total_epochs * 0.8 else 'Late'}
• Overfitting: {'Low' if abs(final_loss - final_val_loss) < final_loss * 0.1 else 'Moderate'}
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9,
                         edgecolor='teal', linewidth=2))
        
        fig.suptitle('🚀 Professional Enhanced Flare Analysis Training Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
        axes[1, 2].plot(self.history.history['lr'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        return fig


class FlareEnergyAnalyzer:
    """
    Advanced analyzer for flare energy distribution and statistical properties
    """
    
    def __init__(self):
        self.energy_distributions = {}
        self.power_law_fits = {}
    def analyze_energy_distribution(self, flare_data, time_data=None):
        """
        Comprehensive analysis of flare energy distribution
        
        Parameters
        ----------
        flare_data : dict or array-like
            Dictionary containing flare parameters and energies, or array of energy values
        time_data : array-like, optional
            Time stamps for temporal analysis
            
        Returns
        -------
        dict
            Comprehensive analysis results
        """
        results = {
            'basic_statistics': {},
            'power_law_analysis': {},
            'temporal_analysis': {},
            'nanoflare_analysis': {},
            'corona_heating_assessment': {}
        }
        
        # Extract energies - handle both dict and array inputs
        if isinstance(flare_data, dict):
            energies = flare_data.get('energies', [])
        else:
            # Assume it's an array-like object
            energies = np.array(flare_data).flatten()
            
        if len(energies) == 0:
            return results
        
        energies = np.array(energies)
        positive_energies = energies[energies > 0]
        
        # Basic statistics
        results['basic_statistics'] = {
            'total_events': len(energies),
            'total_energy': np.sum(positive_energies),
            'mean_energy': np.mean(positive_energies),
            'median_energy': np.median(positive_energies),
            'std_energy': np.std(positive_energies),
            'min_energy': np.min(positive_energies),
            'max_energy': np.max(positive_energies),
            'energy_range': np.max(positive_energies) - np.min(positive_energies)
        }
        
        # Power law analysis
        results['power_law_analysis'] = self._analyze_power_law(positive_energies)
        
        # Nanoflare analysis
        results['nanoflare_analysis'] = self._analyze_nanoflares(positive_energies)
        
        # Corona heating assessment
        results['corona_heating_assessment'] = self._assess_corona_heating(
            results['power_law_analysis'], results['nanoflare_analysis']
        )
        
        # Temporal analysis if time data provided
        if time_data is not None:
            results['temporal_analysis'] = self._analyze_temporal_patterns(
                energies, time_data
            )
        
        return results
    
    def _analyze_power_law(self, energies):
        """Analyze power law distribution of energies"""
        try:
            # Log-binning for better power law visualization
            log_energies = np.log10(energies)
            n_bins = min(50, len(energies) // 10)
            
            hist, bin_edges = np.histogram(log_energies, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Remove zero counts
            nonzero_mask = hist > 0
            hist = hist[nonzero_mask]
            bin_centers = bin_centers[nonzero_mask]
            
            if len(hist) < 5:
                return {'alpha': None, 'r_squared': None, 'fit_quality': 'poor'}
            
            # Linear fit in log-log space
            log_hist = np.log10(hist)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                bin_centers, log_hist
            )
            alpha = float(-slope)
            r_squared = float(r_value**2)
            
            # Assess fit quality
            if r_squared > 0.8:
                fit_quality = 'excellent'
            elif r_squared > 0.6:
                fit_quality = 'good'
            elif r_squared > 0.4:
                fit_quality = 'fair'
            else:
                fit_quality = 'poor'
            
            return {
                'alpha': alpha,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_error': std_err,
                'fit_quality': fit_quality,
                'slope': slope,
                'intercept': intercept
            }
        
        except Exception as e:
            return {'alpha': None, 'error': str(e)}
    
    def _analyze_nanoflares(self, energies):
        """Analyze nanoflare characteristics"""
        # Define nanoflare energy threshold (typically < 10^-6 ergs)
        nanoflare_threshold = 1e-6
        
        nanoflares = energies[energies < nanoflare_threshold]
        regular_flares = energies[energies >= nanoflare_threshold]
        
        return {
            'nanoflare_count': len(nanoflares),
            'regular_flare_count': len(regular_flares),
            'nanoflare_fraction': len(nanoflares) / len(energies) if len(energies) > 0 else 0,
            'nanoflare_total_energy': np.sum(nanoflares),
            'nanoflare_mean_energy': np.mean(nanoflares) if len(nanoflares) > 0 else 0,
            'energy_threshold': nanoflare_threshold
        }
    
    def _assess_corona_heating(self, power_law_results, nanoflare_results):
        """Assess corona heating contribution based on power law and nanoflares"""
        assessment = {
            'heating_mechanism': 'unknown',
            'confidence': 'low',
            'nanoflare_heating_potential': False,
            'power_law_significance': False
        }
        
        alpha = power_law_results.get('alpha')
        if alpha is not None:
            assessment['power_law_significance'] = abs(alpha) > 1.5
            
            # Check if alpha > 2 (steeper than -2), indicating nanoflare dominance
            if alpha > 2.0 and power_law_results.get('r_squared', 0) > 0.6:
                assessment['heating_mechanism'] = 'nanoflare_dominated'
                assessment['confidence'] = 'high'
                assessment['nanoflare_heating_potential'] = True
            elif 1.5 < alpha <= 2.0:
                assessment['heating_mechanism'] = 'mixed'
                assessment['confidence'] = 'medium'
                assessment['nanoflare_heating_potential'] = True
            else:
                assessment['heating_mechanism'] = 'large_flare_dominated'
                assessment['confidence'] = 'medium'
        
        # Enhance assessment with nanoflare fraction
        nanoflare_fraction = nanoflare_results.get('nanoflare_fraction', 0)
        if nanoflare_fraction > 0.7:
            assessment['nanoflare_heating_potential'] = True
            if assessment['heating_mechanism'] == 'unknown':
                assessment['heating_mechanism'] = 'nanoflare_dominated'
        
        return assessment
    
    def _analyze_temporal_patterns(self, energies, time_data):
        """Analyze temporal patterns in flare occurrence"""
        if len(energies) != len(time_data):
            return {'error': 'Energy and time data length mismatch'}
        
        # Calculate inter-flare intervals
        intervals = np.diff(time_data)
        
        # Waiting time distribution
        waiting_times = intervals[intervals > 0]
        
        return {
            'mean_interval': np.mean(waiting_times) if len(waiting_times) > 0 else 0,
            'median_interval': np.median(waiting_times) if len(waiting_times) > 0 else 0,
            'min_interval': np.min(waiting_times) if len(waiting_times) > 0 else 0,
            'max_interval': np.max(waiting_times) if len(waiting_times) > 0 else 0,
            'flare_rate': len(energies) / (time_data[-1] - time_data[0]) if len(time_data) > 1 else 0
        }
    def plot_comprehensive_analysis(self, analysis_results, energies):
        """Create comprehensive visualization of the analysis with professional seaborn aesthetics"""
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Solar Flare Energy Analysis', fontsize=16, fontweight='bold', y=0.98)
          # Energy histogram with seaborn styling
        valid_energies = energies[energies > 0]
        log_energies = np.log10(valid_energies)
        
        sns.histplot(log_energies, bins=30, alpha=0.7, ax=axes[0, 0], 
                    color='skyblue', kde=True, stat='density')
        axes[0, 0].set_xlabel('Log₁₀(Energy)', fontsize=12)
        axes[0, 0].set_ylabel('Density', fontsize=12)
        axes[0, 0].set_title('Energy Distribution', fontsize=14, fontweight='bold')
        
        # Add statistics annotation
        mean_log = np.mean(log_energies)
        std_log = np.std(log_energies)
        axes[0, 0].axvline(mean_log, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_log:.2f}')
        axes[0, 0].legend()
          # Power law fit with enhanced visualization
        if analysis_results['power_law_analysis']['alpha'] is not None:
            hist, bin_edges = np.histogram(log_energies, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            nonzero_mask = hist > 0
            x_data = 10**bin_centers[nonzero_mask]
            y_data = hist[nonzero_mask]
            
            # Plot data points with seaborn styling
            axes[0, 1].loglog(x_data, y_data, 'o', alpha=0.7, color='steelblue',
                             markersize=6, label='Data')
            
            # Plot fit line
            alpha = analysis_results['power_law_analysis']['alpha']
            intercept = analysis_results['power_law_analysis']['intercept']
            y_fit = 10**(intercept) * x_data**(-alpha)
            axes[0, 1].loglog(x_data, y_fit, '-', linewidth=3, color='crimson',
                             label=f'Power Law (α = {alpha:.2f})')
            
            axes[0, 1].set_xlabel('Energy', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].set_title('Power Law Analysis', fontsize=14, fontweight='bold')
            axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
            
            # Add goodness of fit annotation
            r_squared = analysis_results['power_law_analysis'].get('r_squared', 'N/A')
            axes[0, 1].text(0.05, 0.95, f'R² = {r_squared}', transform=axes[0, 1].transAxes,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           fontsize=10, verticalalignment='top')
          # Cumulative distribution with enhanced styling
        sorted_energies = np.sort(valid_energies)
        cumulative = 1 - np.arange(len(sorted_energies)) / len(sorted_energies)
        
        axes[0, 2].loglog(sorted_energies, cumulative, '-', linewidth=2.5, 
                         color='darkgreen', alpha=0.8, label='Cumulative Distribution')
        axes[0, 2].set_xlabel('Energy', fontsize=12)
        axes[0, 2].set_ylabel('Cumulative Probability', fontsize=12)
        axes[0, 2].set_title('Cumulative Energy Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        
        # Add median and quartile lines
        median_energy = np.median(sorted_energies)
        q75_energy = np.percentile(sorted_energies, 75)
        axes[0, 2].axvline(median_energy, color='red', linestyle='--', alpha=0.7, 
                          label=f'Median: {median_energy:.2e}')
        axes[0, 2].axvline(q75_energy, color='orange', linestyle=':', alpha=0.7,
                          label=f'Q75: {q75_energy:.2e}')
        axes[0, 2].legend()
          # Energy vs time with enhanced scatter plot
        if 'temporal_analysis' in analysis_results and len(energies) > 1:
            time_index = np.arange(len(energies))
            
            # Create scatter plot with color coding based on energy levels
            scatter = axes[1, 0].scatter(time_index, energies, alpha=0.7, s=30,
                                       c=np.log10(energies), cmap='plasma',
                                       edgecolors='black', linewidth=0.5)
            
            axes[1, 0].set_xlabel('Time Index', fontsize=12)
            axes[1, 0].set_ylabel('Energy', fontsize=12)
            axes[1, 0].set_title('Energy Time Series', fontsize=14, fontweight='bold')
            axes[1, 0].set_yscale('log')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 0], shrink=0.8)
            cbar.set_label('Log₁₀(Energy)', fontsize=10)
            
            # Add trend line if enough points
            if len(energies) > 10:
                z = np.polyfit(time_index, np.log10(energies), 1)
                p = np.poly1d(z)
                axes[1, 0].plot(time_index, 10**p(time_index), '--', 
                               color='red', alpha=0.8, linewidth=2, label='Trend')
                axes[1, 0].legend()
          # Statistics summary with enhanced design
        stats = analysis_results['basic_statistics']
        stats_text = f"""📊 Statistical Summary
        
Total Events: {stats['total_events']:,}
Total Energy: {stats['total_energy']:.2e} J
Mean Energy: {stats['mean_energy']:.2e} J
Median Energy: {stats['median_energy']:.2e} J
Energy Range: {stats['energy_range']:.2e} J

Distribution Metrics:
• Skewness: {stats.get('skewness', 'N/A')}
• Kurtosis: {stats.get('kurtosis', 'N/A')}"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                                alpha=0.9, edgecolor='navy', linewidth=1.5))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistical Analysis', fontsize=14, fontweight='bold')
          # Corona heating assessment with enhanced styling
        heating = analysis_results['corona_heating_assessment']
        heating_text = f"""🔥 Corona Heating Analysis
        
Heating Mechanism: {heating['heating_mechanism']}
Confidence Level: {heating['confidence']}
Nanoflare Potential: {heating['nanoflare_heating_potential']}
Power Law Significance: {heating['power_law_significance']}

Physical Parameters:"""
        
        if analysis_results['power_law_analysis']['alpha'] is not None:
            alpha_val = analysis_results['power_law_analysis']['alpha']
            heating_text += f"\n• Power Law Index (α): {alpha_val:.3f}"
            
            # Interpret alpha value
            if alpha_val < 1.5:
                interpretation = "Steep - Dominated by large events"
            elif alpha_val < 2.0:
                interpretation = "Moderate - Mixed population"
            else:
                interpretation = "Shallow - Dominated by small events"
            heating_text += f"\n• Interpretation: {interpretation}"
        
        # Determine box color based on heating potential
        box_color = 'lightgreen' if heating['nanoflare_heating_potential'] == 'High' else 'lightyellow'
        edge_color = 'darkgreen' if heating['nanoflare_heating_potential'] == 'High' else 'orange'
        
        axes[1, 2].text(0.05, 0.95, heating_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, 
                                alpha=0.9, edgecolor=edge_color, linewidth=1.5))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Corona Heating Assessment')
        
        plt.tight_layout()
        return fig
