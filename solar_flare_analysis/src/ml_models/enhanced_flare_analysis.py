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
        """Plot comprehensive training history"""
        if self.history is None:
            print("No training history available.")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss plots
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Flare parameters loss
        axes[0, 1].plot(self.history.history['flare_params_loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_flare_params_loss'], label='Validation')
        axes[0, 1].set_title('Flare Parameters Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Energy estimation loss
        axes[0, 2].plot(self.history.history['energy_estimates_loss'], label='Training')
        axes[0, 2].plot(self.history.history['val_energy_estimates_loss'], label='Validation')
        axes[0, 2].set_title('Energy Estimation Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Classification metrics
        axes[1, 0].plot(self.history.history['classification_loss'], label='Training')
        axes[1, 0].plot(self.history.history['val_classification_loss'], label='Validation')
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Accuracy
        axes[1, 1].plot(self.history.history['classification_accuracy'], label='Training')
        axes[1, 1].plot(self.history.history['val_classification_accuracy'], label='Validation')
        axes[1, 1].set_title('Classification Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        if 'lr' in self.history.history:
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
        """Create comprehensive visualization of the analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Energy histogram
        axes[0, 0].hist(np.log10(energies[energies > 0]), bins=30, alpha=0.7)
        axes[0, 0].set_xlabel('Log10(Energy)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Energy Distribution')
        axes[0, 0].grid(True)
        
        # Power law fit
        if analysis_results['power_law_analysis']['alpha'] is not None:
            log_energies = np.log10(energies[energies > 0])
            hist, bin_edges = np.histogram(log_energies, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            nonzero_mask = hist > 0
            axes[0, 1].loglog(10**bin_centers[nonzero_mask], hist[nonzero_mask], 'bo', alpha=0.7)
            
            # Plot fit line
            alpha = analysis_results['power_law_analysis']['alpha']
            intercept = analysis_results['power_law_analysis']['intercept']
            x_fit = 10**bin_centers[nonzero_mask]
            y_fit = 10**(intercept) * x_fit**(-alpha)
            axes[0, 1].loglog(x_fit, y_fit, 'r-', linewidth=2, 
                             label=f'α = {alpha:.2f}')
            
            axes[0, 1].set_xlabel('Energy')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Power Law Fit')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Cumulative distribution
        sorted_energies = np.sort(energies[energies > 0])
        cumulative = 1 - np.arange(len(sorted_energies)) / len(sorted_energies)
        axes[0, 2].loglog(sorted_energies, cumulative, 'b-', alpha=0.7)
        axes[0, 2].set_xlabel('Energy')
        axes[0, 2].set_ylabel('Cumulative Frequency')
        axes[0, 2].set_title('Cumulative Distribution')
        axes[0, 2].grid(True)
        
        # Energy vs time (if temporal data available)
        if 'temporal_analysis' in analysis_results and len(energies) > 1:
            time_index = np.arange(len(energies))
            axes[1, 0].scatter(time_index, energies, alpha=0.6, s=20)
            axes[1, 0].set_xlabel('Time Index')
            axes[1, 0].set_ylabel('Energy')
            axes[1, 0].set_title('Energy vs Time')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Statistics summary
        stats = analysis_results['basic_statistics']
        stats_text = f"""Total Events: {stats['total_events']}
Total Energy: {stats['total_energy']:.2e}
Mean Energy: {stats['mean_energy']:.2e}
Median Energy: {stats['median_energy']:.2e}
Energy Range: {stats['energy_range']:.2e}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Basic Statistics')
        
        # Corona heating assessment
        heating = analysis_results['corona_heating_assessment']
        heating_text = f"""Heating Mechanism: {heating['heating_mechanism']}
Confidence: {heating['confidence']}
Nanoflare Potential: {heating['nanoflare_heating_potential']}
Power Law Significant: {heating['power_law_significance']}"""
        
        if analysis_results['power_law_analysis']['alpha'] is not None:
            heating_text += f"\nAlpha: {analysis_results['power_law_analysis']['alpha']:.2f}"
        
        axes[1, 2].text(0.1, 0.5, heating_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Corona Heating Assessment')
        
        plt.tight_layout()
        return fig
