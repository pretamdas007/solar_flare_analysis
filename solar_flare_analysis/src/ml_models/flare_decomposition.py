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
import seaborn as sns
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
        Enhanced training history visualization with seaborn        
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the professional training history dashboard
        """
        if self.history is None:
            print("No training history available.")
            return None
        
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        # Create comprehensive training dashboard
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        epochs = range(1, len(self.history.history['loss']) + 1)
        
        # 1. Loss Evolution with Enhanced Styling
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Prepare data for seaborn
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
        
        # Add trend analysis
        if len(epochs) > 5:
            train_trend = np.polyfit(epochs, self.history.history['loss'], 1)
            ax1.plot(epochs, np.poly1d(train_trend)(epochs), '--', 
                    color='blue', alpha=0.5, linewidth=2, label='Training Trend')
        
        ax1.set_title('🎯 Training & Validation Loss Evolution', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='semibold')
        ax1.set_ylabel('Loss Value', fontsize=12, fontweight='semibold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Loss Distribution Analysis
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Create loss distribution comparison
        all_losses = self.history.history['loss'] + self.history.history.get('val_loss', [])
        loss_stats = pd.DataFrame({
            'Training': self.history.history['loss'],
            'Validation': self.history.history.get('val_loss', [0] * len(self.history.history['loss']))
        })
        
        loss_melted = loss_stats.melt(var_name='Type', value_name='Loss')
        sns.boxplot(data=loss_melted, x='Type', y='Loss', ax=ax2, palette='Set2')
        sns.stripplot(data=loss_melted, x='Type', y='Loss', ax=ax2, 
                     color='black', alpha=0.6, size=4)
        
        ax2.set_title('Loss Distribution Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss Value', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Learning Progress Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate improvement rate
        train_losses = self.history.history['loss']
        improvement_rate = []
        window_size = max(3, len(train_losses) // 10)
        
        for i in range(window_size, len(train_losses)):
            recent_avg = np.mean(train_losses[i-window_size:i])
            current = train_losses[i]
            improvement = (recent_avg - current) / recent_avg * 100
            improvement_rate.append(improvement)
        
        if improvement_rate:
            sns.lineplot(x=range(window_size+1, len(train_losses)+1), y=improvement_rate, 
                        ax=ax3, marker='s', linewidth=2.5, markersize=4, color='green')
            ax3.fill_between(range(window_size+1, len(train_losses)+1), improvement_rate, 
                           alpha=0.3, color='green')
            ax3.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        ax3.set_title('Learning Progress Rate (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Improvement Rate (%)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate convergence metrics
        if len(train_losses) > 10:
            # Moving average for smoothing
            window = max(5, len(train_losses) // 20)
            smooth_loss = pd.Series(train_losses).rolling(window=window).mean()
            
            sns.lineplot(x=epochs, y=train_losses, ax=ax4, alpha=0.5, 
                        linewidth=1, color='lightblue', label='Raw Loss')
            sns.lineplot(x=epochs, y=smooth_loss, ax=ax4, 
                        linewidth=3, color='darkblue', label='Smoothed Loss')
            
            # Mark convergence point (when improvement becomes minimal)
            derivatives = np.diff(smooth_loss.dropna())
            convergence_point = np.where(np.abs(derivatives) < np.std(derivatives) * 0.1)[0]
            if len(convergence_point) > 0:
                conv_epoch = convergence_point[0] + window
                ax4.axvline(conv_epoch, color='red', linestyle=':', linewidth=2, 
                           label=f'Convergence ~Epoch {conv_epoch}')
        
        ax4.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Training Summary Dashboard
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate comprehensive statistics
        final_loss = self.history.history['loss'][-1]
        best_loss = min(self.history.history['loss'])
        best_epoch = self.history.history['loss'].index(best_loss) + 1
        total_epochs = len(self.history.history['loss'])
        
        val_final = self.history.history.get('val_loss', [0])[-1] if self.history.history.get('val_loss') else 0
        val_best = min(self.history.history.get('val_loss', [999])) if self.history.history.get('val_loss') else 0
        
        summary_text = f"""📊 TRAINING SUMMARY
        
🏆 Performance Metrics:
• Final Training Loss: {final_loss:.6f}
• Best Training Loss: {best_loss:.6f}
• Best Epoch: {best_epoch}
• Total Epochs: {total_epochs}

📈 Validation Metrics:
• Final Val Loss: {val_final:.6f}
• Best Val Loss: {val_best:.6f}

🎯 Model Configuration:
• Architecture: Flare Decomposition
• Sequence Length: {self.sequence_length}
• Max Flares: {self.max_flares}
• Features: {self.n_features}

⚡ Training Stats:
• Avg Improvement: {(self.history.history['loss'][0] - final_loss) / self.history.history['loss'][0] * 100:.1f}%
• Loss Reduction: {self.history.history['loss'][0] / final_loss:.2f}x
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.9,
                         edgecolor='navy', linewidth=2))
        
        fig.suptitle('🚀 Professional Flare Decomposition Training Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
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
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        # Create comprehensive decomposition analysis
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Original vs Reconstructed (Main comparison)
        ax1 = fig.add_subplot(gs[0, :2])
        time_steps = np.arange(len(time_series))
        
        # Enhanced line plots with seaborn styling
        sns.lineplot(x=time_steps, y=time_series.flatten(), ax=ax1, 
                    label='Original Signal', linewidth=2.5, color='black', alpha=0.8)
        sns.lineplot(x=time_steps, y=combined_flares.flatten(), ax=ax1, 
                    label='Reconstructed Signal', linewidth=2, color='red', linestyle='--', alpha=0.9)
        
        ax1.fill_between(time_steps, time_series.flatten(), alpha=0.3, color='lightblue', label='Original')
        ax1.fill_between(time_steps, combined_flares.flatten(), alpha=0.3, color='lightcoral', label='Reconstructed')
        
        ax1.set_title('🔍 Original vs Reconstructed Signal Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Steps', fontsize=12, fontweight='semibold')
        ax1.set_ylabel('Signal Intensity', fontsize=12, fontweight='semibold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # 2. Reconstruction Quality Metrics
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Calculate quality metrics
        mse = mean_squared_error(time_series, combined_flares)
        r2 = r2_score(time_series, combined_flares)
        correlation = np.corrcoef(time_series.flatten(), combined_flares.flatten())[0, 1]
        
        metrics_data = pd.DataFrame({
            'Metric': ['MSE', 'R² Score', 'Correlation'],
            'Value': [mse, r2, correlation],
            'Color': ['red', 'green', 'blue']
        })
        
        sns.barplot(data=metrics_data, x='Metric', y='Value', ax=ax2, palette=metrics_data['Color'])
        ax2.set_title('Reconstruction Quality', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(metrics_data['Value']):
            ax2.text(i, v + max(metrics_data['Value']) * 0.01, f'{v:.4f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Individual Flare Decomposition
        ax3 = fig.add_subplot(gs[1, :])
        
        # Prepare data for seaborn
        flare_data = []
        colors = sns.color_palette("husl", model.max_flares)
        
        for j in range(model.max_flares):
            flare_intensity = np.max(individual_flares[:, j])
            if flare_intensity > 0.01:  # Only show significant flares
                for t, intensity in enumerate(individual_flares[:, j]):
                    flare_data.append({
                        'Time': t,
                        'Intensity': intensity,
                        'Flare': f'Flare {j+1}',
                        'Peak': flare_intensity
                    })
        
        if flare_data:
            flare_df = pd.DataFrame(flare_data)
            sns.lineplot(data=flare_df, x='Time', y='Intensity', hue='Flare', 
                        ax=ax3, linewidth=2.5, marker='o', markersize=3, alpha=0.8)
            
            # Add fill areas for better visualization
            for j, flare_id in enumerate(flare_df['Flare'].unique()):
                flare_subset = flare_df[flare_df['Flare'] == flare_id]
                ax3.fill_between(flare_subset['Time'], flare_subset['Intensity'], 
                               alpha=0.3, color=colors[j])
        else:
            ax3.text(0.5, 0.5, 'No Significant Flares Detected', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
        
        ax3.set_title('📊 Individual Flare Decomposition Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Time Steps', fontsize=12, fontweight='semibold')
        ax3.set_ylabel('Flare Intensity', fontsize=12, fontweight='semibold')
        ax3.legend(frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.05, 1))
        ax3.grid(True, alpha=0.3)
        
        # 4. Residual Analysis
        ax4 = fig.add_subplot(gs[2, 0])
        residuals = time_series.flatten() - combined_flares.flatten()
        
        sns.histplot(residuals, kde=True, ax=ax4, color='purple', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Line')
        ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Residuals', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Temporal Error Analysis
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Calculate rolling error
        window_size = max(10, len(residuals) // 20)
        rolling_error = pd.Series(np.abs(residuals)).rolling(window=window_size).mean()
        
        sns.lineplot(x=range(len(rolling_error)), y=rolling_error, ax=ax5, 
                    linewidth=2.5, color='orange', marker='o', markersize=3)
        ax5.fill_between(range(len(rolling_error)), rolling_error, alpha=0.3, color='orange')
        ax5.set_title('Rolling Mean Absolute Error', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time Steps', fontsize=11)
        ax5.set_ylabel('MAE', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # 6. Decomposition Summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Calculate summary statistics
        total_flares = np.sum([1 for j in range(model.max_flares) 
                              if np.max(individual_flares[:, j]) > 0.01])
        peak_original = np.max(time_series)
        peak_reconstructed = np.max(combined_flares)
        
        summary_text = f"""📊 DECOMPOSITION SUMMARY
        
🎯 Model Performance:
• MSE: {mse:.6f}
• R² Score: {r2:.4f}
• Correlation: {correlation:.4f}

🔥 Flare Analysis:
• Detected Flares: {total_flares}/{model.max_flares}
• Original Peak: {peak_original:.4f}
• Reconstructed Peak: {peak_reconstructed:.4f}

📈 Signal Statistics:
• Mean Residual: {np.mean(residuals):.6f}
• Std Residual: {np.std(residuals):.6f}
• Max Error: {np.max(np.abs(residuals)):.6f}

⚡ Model Config:
• Sequence Length: {len(time_series)}
• Max Flares: {model.max_flares}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9,
                         edgecolor='navy', linewidth=2))
        
        fig.suptitle('🚀 Professional Flare Decomposition Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    return time_series, individual_flares, combined_flares
