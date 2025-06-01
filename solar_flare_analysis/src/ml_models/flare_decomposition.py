"""
Machine Learning models for flare separation and detection
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


class FlareDecompositionModel:
    """
    Neural network model for decomposing overlapping solar flares
    """
    
    def __init__(self, sequence_length=128, n_features=1, max_flares=3, dropout_rate=0.2):
        """
        Initialize the flare decomposition model.
        
        Parameters
        ----------
        sequence_length : int, optional
            Length of input time series sequences
        n_features : int, optional
            Number of input features per time step
        max_flares : int, optional
            Maximum number of overlapping flares to decompose
        dropout_rate : float, optional
            Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.max_flares = max_flares
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def build_model(self):
        """
        Build and compile the neural network model.
        
        Returns
        -------
        tensorflow.keras.Model
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder layers (extract features from time series)
        x = layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        
        # Bidirectional LSTM layer for sequence modeling
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers for prediction
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layers - for each potential flare, predict parameters
        # For each flare: [amplitude, peak_position, rise_time, decay_time, background]
        flare_params = 5
        outputs = layers.Dense(self.max_flares * flare_params, activation='linear')(x)
        
        # Create and compile the model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X, y=None, fit_scalers=False):
        """
        Prepare data for training or prediction.
        
        Parameters
        ----------
        X : array-like
            Input time series data
        y : array-like, optional
            Target flare parameters
        fit_scalers : bool, optional
            If True, fit the scalers on the data
            
        Returns
        -------
        tuple
            Scaled X and y data
        """
        # Ensure X is the right shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], self.n_features)
        
        # Scale the data
        if fit_scalers:
            X_scaled = self.scaler_X.fit_transform(X.reshape(-1, self.n_features))
        else:
            X_scaled = self.scaler_X.transform(X.reshape(-1, self.n_features))
        
        X_scaled = X_scaled.reshape(X.shape)
        
        if y is not None:
            if fit_scalers:
                y_scaled = self.scaler_y.fit_transform(y)
            else:
                y_scaled = self.scaler_y.transform(y)
            return X_scaled, y_scaled
        
        return X_scaled
    
    def generate_synthetic_data(self, n_samples=1000, noise_level=0.05):
        """
        Generate synthetic data for training and testing.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate
        noise_level : float, optional
            Level of noise to add to synthetic data
            
        Returns
        -------
        tuple
            X (time series) and y (flare parameters) data
        """
        X = np.zeros((n_samples, self.sequence_length, self.n_features))
        y = np.zeros((n_samples, self.max_flares * 5))  # 5 parameters per flare
        
        # Time array for generating flare profiles
        t = np.linspace(0, 1, self.sequence_length)
        
        for i in range(n_samples):
            # Randomly decide how many overlapping flares (1 to max_flares)
            n_flares = np.random.randint(1, self.max_flares + 1)
            
            # Initialize the combined signal
            combined_signal = np.zeros(self.sequence_length)
            
            for j in range(n_flares):
                # Generate random parameters for each flare
                amplitude = np.random.uniform(0.2, 1.0)
                peak_pos = np.random.uniform(0.2, 0.8)
                rise_time = np.random.uniform(0.01, 0.1)
                decay_time = np.random.uniform(0.05, 0.3)
                background = np.random.uniform(0.0, 0.1)
                
                # Store parameters in target array
                y[i, j*5:(j+1)*5] = [amplitude, peak_pos, rise_time, decay_time, background]
                
                # Generate flare profile
                peak_idx = int(peak_pos * self.sequence_length)
                flare = np.zeros(self.sequence_length)
                
                # Generate exponential rise and decay
                for k in range(self.sequence_length):
                    if k <= peak_idx:
                        # Rise phase
                        flare[k] = amplitude * np.exp(-(peak_idx - k) / (rise_time * self.sequence_length))
                    else:
                        # Decay phase
                        flare[k] = amplitude * np.exp(-(k - peak_idx) / (decay_time * self.sequence_length))
                
                # Add to combined signal
                combined_signal += flare + background
            
            # Add noise
            noise = np.random.normal(0, noise_level, self.sequence_length)
            combined_signal += noise
            
            # Store in X array
            X[i, :, 0] = combined_signal
        
        return X, y
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, 
              callbacks=None, save_path=None):
        """
        Train the model on provided data.
        
        Parameters
        ----------
        X : array-like
            Input time series data
        y : array-like
            Target flare parameters
        validation_split : float, optional
            Fraction of data to use for validation
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        callbacks : list, optional
            List of Keras callbacks
        save_path : str, optional
            Path to save the best model
            
        Returns
        -------
        tensorflow.keras.callbacks.History
            Training history
        """
        # Prepare data
        X_scaled, y_scaled = self.prepare_data(X, y, fit_scalers=True)
        
        # If no callbacks provided, create default ones
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
            ]
            
            if save_path:
                callbacks.append(
                    ModelCheckpoint(
                        filepath=save_path,
                        save_best_only=True,
                        monitor='val_loss'
                    )
                )
        
        # Train the model
        self.history = self.model.fit(
            X_scaled, y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Parameters
        ----------
        X : array-like
            Input time series data
            
        Returns
        -------
        array
            Predicted flare parameters, unscaled
        """
        X_scaled = self.prepare_data(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        X : array-like
            Input time series data
        y : array-like
            True flare parameters
            
        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        X_scaled, y_scaled = self.prepare_data(X, y)
        
        # Get predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Calculate MSE on scaled data
        mse_scaled = mean_squared_error(y_scaled, y_pred_scaled)
        
        # Unscale predictions and true values
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics on unscaled data
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate parameter-wise metrics
        param_mse = []
        param_r2 = []
        
        for i in range(0, self.max_flares * 5, 5):
            if i + 5 <= y.shape[1]:
                param_mse.append(mean_squared_error(y[:, i:i+5], y_pred[:, i:i+5]))
                param_r2.append(r2_score(y[:, i:i+5], y_pred[:, i:i+5]))
        
        return {
            'mse': mse,
            'mse_scaled': mse_scaled,
            'r2': r2,
            'flare_mse': param_mse,
            'flare_r2': param_r2
        }
    
    def plot_training_history(self):
        """
        Plot the training history of the model.
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the training history plot
        """
        if self.history is None:
            print("No training history available.")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")
    
    def load_model(self, filepath):
        """
        Load a model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def generate_flare_profile(self, params):
        """
        Generate a flare profile from parameters.
        
        Parameters
        ----------
        params : array-like
            Flare parameters [amplitude, peak_pos, rise_time, decay_time, background]
            
        Returns
        -------
        array
            Generated flare profile
        """
        amplitude, peak_pos, rise_time, decay_time, background = params
        
        # Convert peak_pos from [0,1] to an index
        peak_idx = int(peak_pos * self.sequence_length)
        
        # Generate flare profile
        flare = np.zeros(self.sequence_length)
        
        # Generate exponential rise and decay
        for k in range(self.sequence_length):
            if k <= peak_idx:
                # Rise phase
                flare[k] = amplitude * np.exp(-(peak_idx - k) / (rise_time * self.sequence_length))
            else:
                # Decay phase
                flare[k] = amplitude * np.exp(-(k - peak_idx) / (decay_time * self.sequence_length))
        
        # Add background
        flare += background
        
        return flare


def reconstruct_flares(model, time_series, window_size=128, step=32, plot=False):
    """
    Apply the flare decomposition model to a continuous time series.
    
    Parameters
    ----------
    model : FlareDecompositionModel
        Trained model for flare decomposition
    time_series : array-like
        Input time series data
    window_size : int, optional
        Size of the sliding window
    step : int, optional
        Step size for the sliding window
    plot : bool, optional
        If True, plot the results
        
    Returns
    -------
    tuple
        Original time series and decomposed flares
    """
    if len(time_series.shape) == 1:
        time_series = time_series.reshape(-1, 1)
    
    # Check if window size matches model's sequence length
    if window_size != model.sequence_length:
        print(f"Warning: window_size ({window_size}) doesn't match model's sequence_length ({model.sequence_length})")
    
    # Generate windows
    n_windows = (len(time_series) - window_size) // step + 1
    windows = np.zeros((n_windows, window_size, 1))
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        windows[i, :, 0] = time_series[start_idx:end_idx, 0]
    
    # Make predictions
    predictions = model.predict(windows)
    
    # Initialize arrays for reconstructed flares
    combined_flares = np.zeros((len(time_series), 1))
    individual_flares = np.zeros((len(time_series), model.max_flares))
    
    # Reconstruct flares
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        
        window_flares = np.zeros((window_size, model.max_flares))
        
        # For each predicted flare in the window
        for j in range(model.max_flares):
            params = predictions[i, j*5:(j+1)*5]
            
            # Skip if amplitude is too small
            if params[0] < 0.05:
                continue
                
            # Generate flare profile
            flare = model.generate_flare_profile(params)
            window_flares[:, j] = flare
        
        # Add to the reconstruction arrays with overlap handling
        weight = np.ones((window_size, 1))
        if i > 0:
            # Apply linear fade-in for overlap with previous window
            fade_in = np.linspace(0, 1, step)
            weight[:step, 0] = fade_in
        
        if i < n_windows - 1:
            # Apply linear fade-out for overlap with next window
            fade_out = np.linspace(1, 0, step)
            weight[-step:, 0] = fade_out
        
        for j in range(model.max_flares):
            individual_flares[start_idx:end_idx, j] += window_flares[:, j] * weight[:, 0]
        
        combined_flares[start_idx:end_idx] += np.sum(window_flares, axis=1, keepdims=True) * weight
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_series, 'k-', label='Original')
        plt.plot(combined_flares, 'r--', label='Reconstructed')
        plt.title('Original vs Reconstructed Signal')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        for j in range(model.max_flares):
            plt.plot(individual_flares[:, j], label=f'Flare {j+1}')
        plt.title('Decomposed Individual Flares')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return time_series, individual_flares, combined_flares
