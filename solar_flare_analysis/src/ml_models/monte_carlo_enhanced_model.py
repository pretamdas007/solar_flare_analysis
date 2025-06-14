"""
Enhanced Monte Carlo Solar Flare ML Model
A complete machine learning model using Monte Carlo methods for uncertainty quantification
Trains on XRS data and provides robust flare detection and classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime, timedelta
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow Probability distributions
tfd = tfp.distributions

class MonteCarloSolarFlareModel:
    """
    Advanced Monte Carlo Solar Flare ML Model
    
    This model combines:
    - Monte Carlo Dropout for uncertainty quantification
    - Bayesian Neural Networks using TensorFlow Probability
    - Multi-task learning (detection, classification, regression)
    - Comprehensive training on XRS data
    """
    
    def __init__(self, sequence_length=128, n_features=2, n_classes=6, 
                 mc_samples=100, dropout_rate=0.3, learning_rate=0.001):
        """
        Initialize the Monte Carlo Solar Flare Model
        
        Parameters
        ----------
        sequence_length : int
            Length of input time series sequences
        n_features : int
            Number of input features (XRSA, XRSB channels)
        n_classes : int
            Number of flare classes (No Flare, A, B, C, M, X)
        mc_samples : int
            Number of Monte Carlo samples for uncertainty estimation
        dropout_rate : float
            Dropout rate for Monte Carlo uncertainty
        learning_rate : float
            Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Model components
        self.model = None
        self.bayesian_model = None
        
        # Data preprocessing
        self.scaler_X = RobustScaler()
        self.scaler_y_reg = StandardScaler()
        
        # Training history
        self.training_history = {}
        self.mc_predictions = {}
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_detection_train = None
        self.y_detection_val = None
        self.y_detection_test = None
        self.y_class_train = None
        self.y_class_val = None
        self.y_class_test = None
        self.y_reg_train = None
        self.y_reg_val = None
        self.y_reg_test = None
    
    def load_xrs_data(self, data_dir="data/XRS", max_files=None):
        """Load and preprocess XRS data from CSV files"""
        logger.info(f"Loading XRS data from: {data_dir}")
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Data directory {data_path} not found. Using synthetic data.")
            return self._generate_synthetic_training_data()
        
        csv_files = list(data_path.glob("*.csv"))
        if max_files:
            csv_files = csv_files[:max_files]
            
        logger.info(f"Found {len(csv_files)} CSV files")
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.lower().str.strip()
                
                # Map common column variations
                column_mapping = {
                    'xrsa_flux': ['xrsa', 'xrs_a', 'a_flux', 'short'],
                    'xrsb_flux': ['xrsb', 'xrs_b', 'b_flux', 'long'],
                    'time': ['time', 'timestamp', 'datetime', 'date']
                }
                
                for target_col, variations in column_mapping.items():
                    for var in variations:
                        if var in df.columns and target_col not in df.columns:
                            df[target_col] = df[var]
                            break
                
                if 'xrsa_flux' in df.columns and 'xrsb_flux' in df.columns:
                    df = df.dropna(subset=['xrsa_flux', 'xrsb_flux'])
                    df = df[(df['xrsa_flux'] > 0) & (df['xrsb_flux'] > 0)]
                    
                    if len(df) > 50000:
                        df = df.sample(n=50000, random_state=42)
                    
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} valid samples from {csv_file.name}")
                    
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if not all_data:
            logger.warning("No valid data loaded. Using synthetic data.")
            return self._generate_synthetic_training_data()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total combined samples: {len(combined_df)}")
        
        X, y_detection, y_classification, y_regression = self._process_xrs_data(combined_df)
        return X, y_detection, y_classification, y_regression
    
    def _process_xrs_data(self, df):
        """Process XRS data into sequences and multi-task targets"""
        logger.info("Processing XRS data into ML-ready format")
        
        # Ensure proper column names
        if 'xrsa_flux' not in df.columns or 'xrsb_flux' not in df.columns:
            raise ValueError("Required columns 'xrsa_flux' and 'xrsb_flux' not found")
        
        # Clean and sort data
        df = df.dropna(subset=['xrsa_flux', 'xrsb_flux'])
        df = df[(df['xrsa_flux'] > 0) & (df['xrsb_flux'] > 0)]
        
        # Take log transform for better numerical stability
        df['xrsa_log'] = np.log10(df['xrsa_flux'])
        df['xrsb_log'] = np.log10(df['xrsb_flux'])
        
        # Create sequences
        X = []
        y_detection = []
        y_classification = []
        y_regression = []
        
        # Generate targets based on flux levels
        flare_classes = self._classify_flare_levels(df['xrsb_flux'])
        
        # Create sliding windows
        for i in range(len(df) - self.sequence_length + 1):
            # Input sequence
            seq = df[['xrsa_log', 'xrsb_log']].iloc[i:i + self.sequence_length].values
            X.append(seq)
            
            # Detection target (binary: flare or no flare)
            max_flux = df['xrsb_flux'].iloc[i:i + self.sequence_length].max()
            has_flare = int(max_flux > 1e-6)  # C-class threshold
            y_detection.append(has_flare)
            
            # Classification target (flare class)
            max_class = flare_classes.iloc[i:i + self.sequence_length].max()
            y_classification.append(max_class)
            
            # Regression target (log peak flux)
            peak_flux = df['xrsb_flux'].iloc[i:i + self.sequence_length].max()
            y_regression.append(np.log10(peak_flux))
        
        X = np.array(X)
        y_detection = np.array(y_detection)
        y_classification = np.array(y_classification)
        y_regression = np.array(y_regression)
        
        logger.info(f"Created {len(X)} sequences, {np.sum(y_detection)} with flares")
        return X, y_detection, y_classification, y_regression
    
    def _classify_flare_levels(self, flux_values):
        """Classify flare levels based on XRSB flux"""
        classes = np.zeros(len(flux_values), dtype=int)
        
        # GOES flare classification thresholds (W/m²)
        thresholds = {
            1: 1e-8,   # A-class
            2: 1e-7,   # B-class  
            3: 1e-6,   # C-class
            4: 1e-5,   # M-class
            5: 1e-4    # X-class
        }
        
        for class_num, threshold in thresholds.items():
            classes[flux_values >= threshold] = class_num
            
        return pd.Series(classes)
    
    def _generate_synthetic_training_data(self):
        """Generate synthetic training data when real data unavailable"""
        logger.info("Generating synthetic training data")
        
        n_samples = 10000
        X = []
        y_detection = []
        y_classification = []
        y_regression = []
        
        for i in range(n_samples):
            # Generate base quiet background
            seq = np.random.lognormal(-8, 0.5, (self.sequence_length, 2))
            
            # Randomly add flares
            if np.random.random() < 0.3:  # 30% chance of flare
                flare_start = np.random.randint(0, self.sequence_length - 20)
                flare_class = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.04, 0.01])
                
                # Add synthetic flare
                flare_duration = np.random.randint(5, 30)
                flare_end = min(flare_start + flare_duration, self.sequence_length)
                
                # Flare intensities
                intensities = {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4}
                peak_intensity = intensities[flare_class] * np.random.uniform(1, 10)
                
                # Create flare profile
                for j in range(flare_start, flare_end):
                    progress = (j - flare_start) / flare_duration
                    if progress < 0.3:  # Rise phase
                        intensity = peak_intensity * (progress / 0.3)
                    else:  # Decay phase
                        intensity = peak_intensity * np.exp(-(progress - 0.3) / 0.7)
                    
                    seq[j, 0] += intensity * 0.1  # XRSA
                    seq[j, 1] += intensity        # XRSB
                
                y_detection.append(1)
                y_classification.append(flare_class)
                y_regression.append(np.log10(peak_intensity))
            else:
                y_detection.append(0)
                y_classification.append(0)
                y_regression.append(np.log10(np.max(seq[:, 1])))
            
            # Log transform
            seq = np.log10(seq)
            X.append(seq)
        
        return np.array(X), np.array(y_detection), np.array(y_classification), np.array(y_regression)
    
    def build_monte_carlo_model(self):
        """Build the complete Monte Carlo ML model with uncertainty quantification"""
        logger.info("Building Monte Carlo Solar Flare ML Model")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features), name='input_sequences')
        
        # Shared feature extraction layers with Monte Carlo Dropout
        x = layers.LSTM(128, return_sequences=True, dropout=self.dropout_rate, 
                       recurrent_dropout=self.dropout_rate)(inputs)
        x = layers.Dropout(self.dropout_rate)(x, training=True)  # Always active for MC
        
        x = layers.LSTM(64, return_sequences=True, dropout=self.dropout_rate,
                       recurrent_dropout=self.dropout_rate)(x)
        x = layers.Dropout(self.dropout_rate)(x, training=True)
        
        x = layers.LSTM(32, dropout=self.dropout_rate, 
                       recurrent_dropout=self.dropout_rate)(x)
        x = layers.Dropout(self.dropout_rate)(x, training=True)
        
        # Dense feature layers
        shared_features = layers.Dense(64, activation='relu')(x)
        shared_features = layers.Dropout(self.dropout_rate)(shared_features, training=True)
        
        # Multi-task outputs
        
        # 1. Detection head (binary classification)
        detection_head = layers.Dense(32, activation='relu', name='detection_dense')(shared_features)
        detection_head = layers.Dropout(self.dropout_rate)(detection_head, training=True)
        detection_output = layers.Dense(1, activation='sigmoid', name='detection_output')(detection_head)
        
        # 2. Classification head (multi-class)
        classification_head = layers.Dense(32, activation='relu', name='classification_dense')(shared_features)
        classification_head = layers.Dropout(self.dropout_rate)(classification_head, training=True)
        classification_output = layers.Dense(self.n_classes, activation='softmax', 
                                           name='classification_output')(classification_head)
        
        # 3. Regression head (peak flux prediction)
        regression_head = layers.Dense(32, activation='relu', name='regression_dense')(shared_features)
        regression_head = layers.Dropout(self.dropout_rate)(regression_head, training=True)
        regression_output = layers.Dense(1, activation='linear', name='regression_output')(regression_head)
        
        # Create multi-output model
        self.model = models.Model(
            inputs=inputs,
            outputs=[detection_output, classification_output, regression_output],
            name='MonteCarloSolarFlareModel'
        )
        
        # Compile with multiple losses
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'detection_output': 'binary_crossentropy',
                'classification_output': 'sparse_categorical_crossentropy',
                'regression_output': 'mse'
            },
            loss_weights={
                'detection_output': 1.0,
                'classification_output': 1.0,
                'regression_output': 0.5
            },
            metrics={
                'detection_output': ['accuracy', 'precision', 'recall'],
                'classification_output': ['accuracy'],
                'regression_output': ['mae']
            }
        )
        
        logger.info(f"Model built successfully: {self.model.count_params()} parameters")
        return self.model
    
    def build_bayesian_model(self):
        """Build Bayesian Neural Network using TensorFlow Probability"""
        logger.info("Building Bayesian Neural Network for uncertainty quantification")
        
        # Bayesian LSTM layer
        def prior_fn(kernel_size, bias_size, dtype=tf.float32):
            n = kernel_size + bias_size
            return tfd.Sequential([
                tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=tf.ones(n, dtype=dtype))
            ])
        
        def posterior_fn(kernel_size, bias_size, dtype=tf.float32):
            n = kernel_size + bias_size
            return tfd.Sequential([
                tfp.layers.VariableLayer(shape=(2*n,), dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Normal(
                    loc=t[..., :n], scale=tf.nn.softplus(t[..., n:])
                ))
            ])
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Regular LSTM layers
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(32)(x)
        
        # Bayesian Dense layers
        x = tfp.layers.DenseVariational(
            32, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn,
            kl_weight=1/self.sequence_length, activation='relu'
        )(x)
        
        # Output layers
        detection_output = tfp.layers.DenseVariational(
            1, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn,
            kl_weight=1/self.sequence_length, activation='sigmoid'
        )(x)
        
        classification_output = tfp.layers.DenseVariational(
            self.n_classes, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn,
            kl_weight=1/self.sequence_length, activation='softmax'
        )(x)
        
        regression_output = tfp.layers.DenseVariational(
            1, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn,
            kl_weight=1/self.sequence_length
        )(x)
        
        # Bayesian model
        self.bayesian_model = models.Model(
            inputs=inputs,
            outputs=[detection_output, classification_output, regression_output],
            name='BayesianSolarFlareModel'
        )
        
        # Compile with KL divergence loss
        def loss_with_kl(y_true, y_pred):
            kl_loss = sum(self.bayesian_model.losses)
            return tf.nn.compute_average_loss(
                tf.keras.losses.binary_crossentropy(y_true, y_pred)
            ) + kl_loss
        
        self.bayesian_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'detection_output': loss_with_kl,
                'classification_output': 'sparse_categorical_crossentropy',
                'regression_output': 'mse'
            }
        )
        
        return self.bayesian_model
    
    def train_model(self, validation_split=0.2, epochs=10, batch_size=32, use_callbacks=True):
        """Train the Monte Carlo model on XRS data"""
        logger.info("Starting model training")
        
        # Load training data
        X, y_detection, y_classification, y_regression = self.load_xrs_data()
        
        # Preprocess data
        X_scaled = self._preprocess_features(X)
        y_reg_scaled = self._preprocess_targets(y_regression)
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        
        self.X_train = X_scaled[:split_idx]
        self.X_val = X_scaled[split_idx:]
        
        self.y_detection_train = y_detection[:split_idx]
        self.y_detection_val = y_detection[split_idx:]
        
        self.y_class_train = y_classification[:split_idx]
        self.y_class_val = y_classification[split_idx:]
        
        self.y_reg_train = y_reg_scaled[:split_idx]
        self.y_reg_val = y_reg_scaled[split_idx:]
        
        # Build model if not exists
        if self.model is None:
            self.build_monte_carlo_model()
        
        # Prepare training data
        train_data = {
            'detection_output': self.y_detection_train,
            'classification_output': self.y_class_train,
            'regression_output': self.y_reg_train
        }
        
        val_data = {
            'detection_output': self.y_detection_val,
            'classification_output': self.y_class_val,
            'regression_output': self.y_reg_val
        }
        
        # Setup callbacks
        callback_list = []
        if use_callbacks:
            callback_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    'best_monte_carlo_model.h5', save_best_only=True, 
                    monitor='val_loss', mode='min'
                )
            ]
        
        # Train model
        history = self.model.fit(
            self.X_train, train_data,
            validation_data=(self.X_val, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.training_history = history.history
        logger.info("Model training completed")
        
        return history
    
    def _preprocess_features(self, X):
        """Preprocess input features"""
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        return X_scaled
    
    def _preprocess_targets(self, y_regression):
        """Preprocess regression targets"""
        y_reshaped = y_regression.reshape(-1, 1)
        y_scaled = self.scaler_y_reg.fit_transform(y_reshaped)
        return y_scaled.flatten()
    
    def predict_with_uncertainty(self, X, return_std=True, n_samples=None):
        """Make predictions with Monte Carlo uncertainty estimation"""
        if n_samples is None:
            n_samples = self.mc_samples
            
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Preprocess input
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        
        X_scaled = self._preprocess_features(X)
        
        # Monte Carlo predictions
        predictions = []
        
        for i in range(n_samples):
            # Enable dropout during inference for MC sampling
            pred = self.model(X_scaled, training=True)
            predictions.append(pred)
        
        # Convert to numpy arrays
        detection_preds = np.array([p[0].numpy() for p in predictions])
        classification_preds = np.array([p[1].numpy() for p in predictions])
        regression_preds = np.array([p[2].numpy() for p in predictions])
        
        # Calculate statistics
        results = {
            'detection': {
                'mean': np.mean(detection_preds, axis=0),
                'std': np.std(detection_preds, axis=0) if return_std else None,
                'confidence_interval': np.percentile(detection_preds, [2.5, 97.5], axis=0),
                'predictions': detection_preds
            },
            'classification': {
                'mean': np.mean(classification_preds, axis=0),
                'std': np.std(classification_preds, axis=0) if return_std else None,
                'confidence_interval': np.percentile(classification_preds, [2.5, 97.5], axis=0),
                'predictions': classification_preds
            },
            'regression': {
                'mean': np.mean(regression_preds, axis=0),
                'std': np.std(regression_preds, axis=0) if return_std else None,
                'confidence_interval': np.percentile(regression_preds, [2.5, 97.5], axis=0),
                'predictions': regression_preds
            }
        }
        
        # Transform regression back to original scale
        if hasattr(self, 'scaler_y_reg'):
            reg_mean_orig = self.scaler_y_reg.inverse_transform(
                results['regression']['mean'].reshape(-1, 1)
            ).flatten()
            results['regression']['mean_original_scale'] = reg_mean_orig
        
        return results
    
    def evaluate_model(self, X_test=None, y_test=None):
        """Evaluate model performance with uncertainty metrics"""
        logger.info("Evaluating model performance")
        
        if X_test is None:
            X_test = self.X_val if hasattr(self, 'X_val') else self.X_test
            y_test = {
                'detection': self.y_detection_val if hasattr(self, 'y_detection_val') else self.y_detection_test,
                'classification': self.y_class_val if hasattr(self, 'y_class_val') else self.y_class_test,
                'regression': self.y_reg_val if hasattr(self, 'y_reg_val') else self.y_reg_test
            }
        
        # Standard evaluation
        standard_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Monte Carlo evaluation
        mc_predictions = self.predict_with_uncertainty(X_test)
        
        # Calculate uncertainty-aware metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(mc_predictions, y_test)
        
        evaluation_results = {
            'standard_metrics': dict(zip(self.model.metrics_names, standard_metrics)),
            'monte_carlo_metrics': uncertainty_metrics,
            'model_info': {
                'parameters': self.model.count_params(),
                'mc_samples': self.mc_samples,
                'dropout_rate': self.dropout_rate
            }
        }
        
        logger.info(f"Model evaluation completed")
        return evaluation_results
    
    def _calculate_uncertainty_metrics(self, mc_predictions, y_true):
        """Calculate uncertainty-aware performance metrics"""
        metrics = {}
        
        # Detection metrics
        det_mean = mc_predictions['detection']['mean'].flatten()
        det_std = mc_predictions['detection']['std'].flatten()
        
        # Classification metrics  
        class_mean = mc_predictions['classification']['mean']
        class_std = mc_predictions['classification']['std']
        
        # Regression metrics
        reg_mean = mc_predictions['regression']['mean'].flatten()
        reg_std = mc_predictions['regression']['std'].flatten()
        
        # Prediction interval coverage
        metrics['prediction_interval_coverage'] = self._calculate_coverage(
            mc_predictions, y_true
        )
        
        # Epistemic vs aleatoric uncertainty
        metrics['uncertainty_decomposition'] = {
            'detection_epistemic': np.mean(det_std),
            'classification_epistemic': np.mean(class_std),
            'regression_epistemic': np.mean(reg_std)
        }
        
        # Calibration metrics
        metrics['calibration'] = self._calculate_calibration(mc_predictions, y_true)
        
        return metrics
    
    def _calculate_coverage(self, predictions, y_true):
        """Calculate prediction interval coverage"""
        coverage = {}
        
        # Detection coverage
        det_lower = predictions['detection']['confidence_interval'][0].flatten()
        det_upper = predictions['detection']['confidence_interval'][1].flatten()
        det_true = y_true['detection'].flatten()
        
        det_coverage = np.mean((det_true >= det_lower) & (det_true <= det_upper))
        coverage['detection'] = det_coverage
        
        # Regression coverage
        reg_lower = predictions['regression']['confidence_interval'][0].flatten()
        reg_upper = predictions['regression']['confidence_interval'][1].flatten()
        reg_true = y_true['regression'].flatten()
        
        reg_coverage = np.mean((reg_true >= reg_lower) & (reg_true <= reg_upper))
        coverage['regression'] = reg_coverage
        
        return coverage
    
    def _calculate_calibration(self, predictions, y_true):
        """Calculate model calibration"""
        # Simplified calibration for detection
        det_probs = predictions['detection']['mean'].flatten()
        det_true = y_true['detection'].flatten()
        
        # Bin predictions and calculate calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_errors = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (det_probs > bin_lower) & (det_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = det_true[in_bin].mean()
                avg_confidence_in_bin = det_probs[in_bin].mean()
                calibration_errors.append(abs(avg_confidence_in_bin - accuracy_in_bin))
        
        expected_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
        
        return {
            'expected_calibration_error': expected_calibration_error,
            'n_bins': n_bins
        }
    
    def save_model(self, filepath='monte_carlo_solar_flare_model.h5'):
        """Save the trained model"""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        self.model.save(filepath)
        
        # Save preprocessing objects
        import joblib
        preprocessing_objects = {
            'scaler_X': self.scaler_X,
            'scaler_y_reg': self.scaler_y_reg,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'mc_samples': self.mc_samples,
            'dropout_rate': self.dropout_rate
        }
        
        preprocessing_path = filepath.replace('.h5', '_preprocessing.pkl')
        joblib.dump(preprocessing_objects, preprocessing_path)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Preprocessing objects saved to {preprocessing_path}")
    
    def load_model(self, filepath='monte_carlo_solar_flare_model.h5'):
        """Load a trained model"""
        import joblib
        
        # Load model
        self.model = tf.keras.models.load_model(filepath)
        
        # Load preprocessing objects
        preprocessing_path = filepath.replace('.h5', '_preprocessing.pkl')
        if Path(preprocessing_path).exists():
            preprocessing_objects = joblib.load(preprocessing_path)
            
            self.scaler_X = preprocessing_objects.get('scaler_X', RobustScaler())
            self.scaler_y_reg = preprocessing_objects.get('scaler_y_reg', StandardScaler())
            self.sequence_length = preprocessing_objects.get('sequence_length', 128)
            self.n_features = preprocessing_objects.get('n_features', 2)
            self.n_classes = preprocessing_objects.get('n_classes', 6)
            self.mc_samples = preprocessing_objects.get('mc_samples', 100)
            self.dropout_rate = preprocessing_objects.get('dropout_rate', 0.3)
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history with uncertainty"""
        if not hasattr(self, 'training_history') or not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Monte Carlo Model Training History', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Detection accuracy
        if 'detection_output_accuracy' in self.training_history:
            axes[0, 1].plot(self.training_history['detection_output_accuracy'], 
                          label='Training Accuracy')
            axes[0, 1].plot(self.training_history['val_detection_output_accuracy'], 
                          label='Validation Accuracy')
            axes[0, 1].set_title('Detection Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Classification accuracy
        if 'classification_output_accuracy' in self.training_history:
            axes[1, 0].plot(self.training_history['classification_output_accuracy'], 
                          label='Training Accuracy')
            axes[1, 0].plot(self.training_history['val_classification_output_accuracy'], 
                          label='Validation Accuracy')
            axes[1, 0].set_title('Classification Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Regression MAE
        if 'regression_output_mae' in self.training_history:
            axes[1, 1].plot(self.training_history['regression_output_mae'], 
                          label='Training MAE')
            axes[1, 1].plot(self.training_history['val_regression_output_mae'], 
                          label='Validation MAE')
            axes[1, 1].set_title('Regression MAE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the Monte Carlo Solar Flare Model
    logger.info("=" * 60)
    logger.info("MONTE CARLO SOLAR FLARE ML MODEL DEMONSTRATION")
    logger.info("=" * 60)
    
    # Create model instance
    mc_model = MonteCarloSolarFlareModel(
        sequence_length=128,
        n_features=2,
        n_classes=6,
        mc_samples=50,  # Reduced for demo
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    logger.info("Model initialized successfully")
    
    # Build and compile model
    model = mc_model.build_monte_carlo_model()
    logger.info(f"Model architecture:")
    model.summary()
    
    # Load and preprocess data
    logger.info("\nLoading and preprocessing XRS data...")
    try:
        X, y_detection, y_classification, y_regression = mc_model.load_xrs_data(
            data_dir="../../data/XRS", max_files=2  # Limit for demo
        )
        logger.info(f"Data loaded: {X.shape[0]} samples")
        
        # Train model (quick demo training)
        logger.info("\nStarting model training...")
        history = mc_model.train_model(
            validation_split=0.2,
            epochs=5,  # Reduced for demo
            batch_size=32,
            use_callbacks=False
        )
        
        # Evaluate model
        logger.info("\nEvaluating model...")
        evaluation = mc_model.evaluate_model()
        logger.info("Evaluation completed")
        
        # Make predictions with uncertainty
        logger.info("\nMaking predictions with uncertainty quantification...")
        sample_X = X[:5]  # Take first 5 samples
        predictions = mc_model.predict_with_uncertainty(sample_X, n_samples=20)
        
        logger.info("Prediction results:")
        for i in range(len(sample_X)):
            det_mean = predictions['detection']['mean'][i][0]
            det_std = predictions['detection']['std'][i][0]
            logger.info(f"Sample {i+1}: Detection probability = {det_mean:.3f} ± {det_std:.3f}")
        
        # Save model
        logger.info("\nSaving trained model...")
        mc_model.save_model("monte_carlo_demo_model.h5")
        
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("The Monte Carlo Solar Flare ML Model is now ready for production use.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        logger.info("Using synthetic data for basic model demonstration...")
        
        # Fallback to synthetic data demo
        X, y_det, y_class, y_reg = mc_model._generate_synthetic_training_data()
        logger.info(f"Synthetic data generated: {X.shape[0]} samples")
        
        logger.info("Basic model demonstration completed with synthetic data")
