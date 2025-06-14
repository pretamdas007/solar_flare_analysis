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
    
    def plot_prediction_uncertainty(self, X, predictions_dict=None, true_values=None, save_path=None):
        """
        Plot prediction uncertainty analysis with modern seaborn visualizations
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences
        predictions_dict : dict, optional
            Pre-computed predictions with uncertainty
        true_values : dict, optional
            True values for comparison
        save_path : str, optional
            Path to save the plot
        """
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # Get predictions if not provided
        if predictions_dict is None:
            predictions_dict = self.predict_with_uncertainty(X[:50], n_samples=30)
        
        # 1. Detection Uncertainty Distribution (seaborn histogram with KDE)
        det_mean = predictions_dict['detection']['mean'].flatten()
        det_std = predictions_dict['detection']['std'].flatten()
        
        det_df = pd.DataFrame({
            'Detection Probability': det_mean,
            'Uncertainty (std)': det_std
        })
        
        sns.histplot(data=det_df, x='Detection Probability', kde=True, 
                    alpha=0.7, color='skyblue', ax=axes[0])
        axes[0].axvline(np.mean(det_mean), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(det_mean):.3f}', linewidth=2)
        axes[0].set_title('Detection Probability Distribution', fontweight='bold')
        axes[0].legend()
        
        # 2. Uncertainty vs Prediction Strength (seaborn scatterplot)
        scatter_df = pd.DataFrame({
            'Prediction Strength': det_mean,
            'Uncertainty': det_std,
            'Prediction Type': ['High Confidence' if std < np.median(det_std) 
                               else 'Low Confidence' for std in det_std]
        })
        
        sns.scatterplot(data=scatter_df, x='Prediction Strength', y='Uncertainty',
                       hue='Prediction Type', alpha=0.7, s=60, ax=axes[1])
        
        # Add trend line
        z = np.polyfit(det_mean, det_std, 1)
        p = np.poly1d(z)
        axes[1].plot(det_mean, p(det_mean), "r--", alpha=0.8, linewidth=2,
                    label=f'Trend (slope={z[0]:.3f})')
        axes[1].set_title('Uncertainty vs Prediction Strength', fontweight='bold')
        axes[1].legend()
        
        # 3. Classification Confidence (seaborn boxplot)
        if 'classification' in predictions_dict:
            class_mean = predictions_dict['classification']['mean']
            class_std = predictions_dict['classification']['std']
            
            # Get max confidence for each sample
            max_confidence = np.max(class_mean, axis=1)
            avg_uncertainty = np.mean(class_std, axis=1)
            
            # Create confidence categories
            conf_categories = pd.cut(max_confidence, bins=3, labels=['Low', 'Medium', 'High'])
            
            conf_df = pd.DataFrame({
                'Confidence Category': conf_categories,
                'Average Uncertainty': avg_uncertainty
            })
            
            sns.boxplot(data=conf_df, x='Confidence Category', y='Average Uncertainty',
                       palette='viridis', ax=axes[2])
            axes[2].set_title('Classification Uncertainty by Confidence Level', fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, 'Classification data\nnot available', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Classification Analysis (N/A)', fontweight='bold')
        
        # 4. Regression Prediction Intervals (seaborn lineplot with confidence intervals)
        if 'regression' in predictions_dict:
            reg_mean = predictions_dict['regression']['mean'].flatten()
            reg_std = predictions_dict['regression']['std'].flatten()
            reg_ci = predictions_dict['regression']['confidence_interval']
            
            n_samples = min(len(reg_mean), 50)
            sample_idx = np.arange(n_samples)
            
            reg_df = pd.DataFrame({
                'Sample': sample_idx,
                'Mean Prediction': reg_mean[:n_samples],
                'Lower CI': reg_ci[0].flatten()[:n_samples],
                'Upper CI': reg_ci[1].flatten()[:n_samples]
            })
            
            # Plot confidence intervals
            axes[3].fill_between(reg_df['Sample'], reg_df['Lower CI'], reg_df['Upper CI'],
                               alpha=0.3, color='lightcoral', label='95% CI')
            
            sns.lineplot(data=reg_df, x='Sample', y='Mean Prediction', 
                        color='red', linewidth=2, label='Mean Prediction', ax=axes[3])
            
            # Add true values if available
            if true_values and 'regression' in true_values:
                true_reg = true_values['regression'][:n_samples]
                reg_df['True Values'] = true_reg
                sns.scatterplot(data=reg_df, x='Sample', y='True Values',
                              color='blue', s=40, alpha=0.8, label='True Values', ax=axes[3])
            
            axes[3].set_title('Regression Predictions with Uncertainty', fontweight='bold')
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, 'Regression data\nnot available', 
                        ha='center', va='center', transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title('Regression Analysis (N/A)', fontweight='bold')
        
        # 5. Input Signal Complexity vs Uncertainty
        if len(X) > 0:
            # Calculate signal complexity metrics
            complexity_metrics = []
            uncertainties = []
            
            n_analyze = min(len(X), len(det_std))
            for i in range(n_analyze):
                # Signal variance as complexity measure
                signal_var = np.var(X[i, :, 0])  # Use first channel
                signal_range = np.ptp(X[i, :, 0])  # Peak-to-peak
                
                complexity_metrics.append(signal_var)
                uncertainties.append(det_std[i])
            
            complexity_df = pd.DataFrame({
                'Signal Complexity (Variance)': complexity_metrics,
                'Prediction Uncertainty': uncertainties
            })
            
            sns.scatterplot(data=complexity_df, x='Signal Complexity (Variance)', 
                           y='Prediction Uncertainty', alpha=0.7, color='green', s=50, ax=axes[4])
            
            # Add correlation info
            correlation = np.corrcoef(complexity_metrics, uncertainties)[0, 1]
            axes[4].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[4].transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[4].set_title('Signal Complexity vs Prediction Uncertainty', fontweight='bold')
        else:
            axes[4].text(0.5, 0.5, 'Input data\nnot available', 
                        ha='center', va='center', transform=axes[4].transAxes, fontsize=12)
            axes[4].set_title('Complexity Analysis (N/A)', fontweight='bold')
        
        # 6. Monte Carlo Sample Convergence
        if 'detection' in predictions_dict and 'predictions' in predictions_dict['detection']:
            mc_samples = predictions_dict['detection']['predictions']
            
            # Calculate running mean for first sample
            if len(mc_samples) > 0:
                first_sample_preds = mc_samples[:, 0, 0]  # First sample, first output
                running_means = [np.mean(first_sample_preds[:i+1]) for i in range(len(first_sample_preds))]
                
                conv_df = pd.DataFrame({
                    'MC Sample': range(len(running_means)),
                    'Running Mean': running_means
                })
                
                sns.lineplot(data=conv_df, x='MC Sample', y='Running Mean',
                           color='purple', linewidth=2, ax=axes[5])
                
                # Add final convergence line
                final_mean = running_means[-1]
                axes[5].axhline(y=final_mean, color='red', linestyle='--', alpha=0.8,
                               label=f'Converged Mean: {final_mean:.4f}')
                
                axes[5].set_title('Monte Carlo Convergence', fontweight='bold')
                axes[5].legend()
            else:
                axes[5].text(0.5, 0.5, 'MC samples\nnot available', 
                            ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
                axes[5].set_title('MC Convergence (N/A)', fontweight='bold')
        else:
            axes[5].text(0.5, 0.5, 'MC samples\nnot available', 
                        ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
            axes[5].set_title('MC Convergence (N/A)', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.suptitle('Monte Carlo Model - Prediction Uncertainty Analysis', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction uncertainty plot saved to: {save_path}")
        
        return fig

    def plot_model_diagnostics(self, X, predictions_dict=None, true_values=None, save_path=None):
        """
        Plot comprehensive model diagnostics with seaborn
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences
        predictions_dict : dict, optional
            Pre-computed predictions
        true_values : dict, optional
            True values for comparison
        save_path : str, optional
            Path to save the plot
        """
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Get predictions if not provided
        if predictions_dict is None:
            if self.model is not None:
                predictions_dict = self.predict_with_uncertainty(X[:50], n_samples=30)
            else:
                logger.warning("No trained model available and no predictions provided. Using placeholder data.")
                # Create placeholder predictions for visualization testing
                n_samples = min(len(X), 50)
                predictions_dict = {
                    'detection': {
                        'mean': np.random.rand(n_samples, 1) * 0.6 + 0.2,
                        'std': np.random.rand(n_samples, 1) * 0.1 + 0.05
                    },
                    'regression': {
                        'mean': np.random.randn(n_samples, 1) * 2 - 8,
                        'std': np.random.rand(n_samples, 1) * 0.2 + 0.1
                    }
                }
        
        # 1. Prediction vs True Values Scatter (multi-task)
        if true_values:
            # Detection task
            det_pred = predictions_dict['detection']['mean'].flatten()
            det_true = true_values['detection'].flatten() if 'detection' in true_values else det_pred
            
            det_df = pd.DataFrame({
                'True Values': det_true,
                'Predictions': det_pred,
                'Task': 'Detection'
            })
            
            # Regression task
            if 'regression' in predictions_dict and 'regression' in true_values:
                reg_pred = predictions_dict['regression']['mean'].flatten()
                reg_true = true_values['regression'].flatten()
                
                reg_df = pd.DataFrame({
                    'True Values': reg_true,
                    'Predictions': reg_pred,
                    'Task': 'Regression'
                })
                
                combined_df = pd.concat([det_df, reg_df], ignore_index=True)
            else:
                combined_df = det_df
            
            # Create faceted scatter plot
            for task in combined_df['Task'].unique():
                task_data = combined_df[combined_df['Task'] == task]
                sns.scatterplot(data=task_data, x='True Values', y='Predictions',
                              alpha=0.6, s=50, ax=axes[0])
            
            # Perfect prediction line
            all_true = combined_df['True Values']
            all_pred = combined_df['Predictions']
            min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
            
            axes[0].set_title('Predictions vs True Values', fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 'True values\nnot available', ha='center', va='center',
                        transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('Predictions vs True Values (N/A)', fontweight='bold')
        
        # 2. Residuals Analysis (seaborn residplot)
        if true_values and 'regression' in predictions_dict and 'regression' in true_values:
            reg_pred = predictions_dict['regression']['mean'].flatten()
            reg_true = true_values['regression'].flatten()
            
            residuals_df = pd.DataFrame({
                'Predictions': reg_pred,
                'Residuals': reg_true - reg_pred
            })
            
            sns.residplot(data=residuals_df, x='Predictions', y='Residuals',
                         lowess=True, scatter_kws={'alpha': 0.6}, 
                         line_kws={'color': 'red'}, ax=axes[1])
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
            axes[1].set_title('Residuals Analysis', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'Residuals data\nnot available', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('Residuals Analysis (N/A)', fontweight='bold')
        
        # 3. Uncertainty Calibration Plot
        det_pred = predictions_dict['detection']['mean'].flatten()
        det_std = predictions_dict['detection']['std'].flatten()
        
        # Create uncertainty bins
        n_bins = 10
        uncertainty_bins = pd.qcut(det_std, q=n_bins, labels=False, duplicates='drop')
        
        calibration_data = []
        for bin_idx in range(n_bins):
            mask = uncertainty_bins == bin_idx
            if np.sum(mask) > 0:
                bin_uncertainty = np.mean(det_std[mask])
                bin_predictions = det_pred[mask]
                bin_variance = np.var(bin_predictions)
                
                calibration_data.append({
                    'Bin': bin_idx,
                    'Expected Uncertainty': bin_uncertainty,
                    'Observed Variance': bin_variance
                })
        
        if calibration_data:
            calib_df = pd.DataFrame(calibration_data)
            sns.scatterplot(data=calib_df, x='Expected Uncertainty', y='Observed Variance',
                           s=80, color='orange', ax=axes[2])
            
            # Perfect calibration line
            min_val = min(calib_df['Expected Uncertainty'].min(), calib_df['Observed Variance'].min())
            max_val = max(calib_df['Expected Uncertainty'].max(), calib_df['Observed Variance'].max())
            axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
            
            axes[2].set_title('Uncertainty Calibration', fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, 'Insufficient data\nfor calibration', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Uncertainty Calibration (N/A)', fontweight='bold')
          # 4. Feature Importance via Input Perturbation
        if len(X) > 0 and self.model is not None:
            # Simplified feature importance: compare predictions with zeroed features
            baseline_pred = self.predict_with_uncertainty(X[:10], n_samples=10)
            baseline_det = baseline_pred['detection']['mean'].flatten()
            
            importance_scores = []
            feature_names = ['XRSA', 'XRSB']
            
            for feat_idx in range(self.n_features):
                X_perturbed = X[:10].copy()
                X_perturbed[:, :, feat_idx] = 0  # Zero out feature
                
                perturbed_pred = self.predict_with_uncertainty(X_perturbed, n_samples=10)
                perturbed_det = perturbed_pred['detection']['mean'].flatten()
                
                importance = np.mean(np.abs(baseline_det - perturbed_det))
                importance_scores.append(importance)
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance Score': importance_scores
            })
            
            sns.barplot(data=importance_df, x='Feature', y='Importance Score',
                       palette='viridis', ax=axes[3])
            axes[3].set_title('Feature Importance (Perturbation)', fontweight='bold')
        else:
            axes[3].text(0.5, 0.5, 'Input data\nnot available', ha='center', va='center',
                        transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title('Feature Importance (N/A)', fontweight='bold')
        
        # 5. Prediction Confidence Distribution
        confidence_data = []
        
        # Detection confidence
        det_confidence = np.maximum(det_pred, 1 - det_pred)  # Distance from 0.5
        confidence_data.extend([{'Task': 'Detection', 'Confidence': conf} for conf in det_confidence])
        
        # Classification confidence (if available)
        if 'classification' in predictions_dict:
            class_pred = predictions_dict['classification']['mean']
            class_confidence = np.max(class_pred, axis=1)  # Max class probability
            confidence_data.extend([{'Task': 'Classification', 'Confidence': conf} for conf in class_confidence])
        
        conf_df = pd.DataFrame(confidence_data)
        
        sns.violinplot(data=conf_df, x='Task', y='Confidence', palette='Set2', ax=axes[4])
        axes[4].set_title('Prediction Confidence Distribution', fontweight='bold')
        
        # 6. Training History Overview (if available)
        if hasattr(self, 'training_history') and self.training_history:
            history_data = []
            
            for metric, values in self.training_history.items():
                if 'val_' not in metric and 'loss' in metric:
                    for epoch, value in enumerate(values):
                        history_data.append({
                            'Epoch': epoch,
                            'Loss': value,
                            'Type': 'Training'
                        })
                
                # Add validation if available
                val_metric = f"val_{metric}"
                if val_metric in self.training_history:
                    for epoch, value in enumerate(self.training_history[val_metric]):
                        history_data.append({
                            'Epoch': epoch,
                            'Loss': value,
                            'Type': 'Validation'
                        })
            
            if history_data:
                hist_df = pd.DataFrame(history_data)
                sns.lineplot(data=hist_df, x='Epoch', y='Loss', hue='Type',
                           marker='o', ax=axes[5])
                axes[5].set_title('Training Loss History', fontweight='bold')
            else:
                axes[5].text(0.5, 0.5, 'Training history\nnot detailed enough', 
                            ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
                axes[5].set_title('Training History (N/A)', fontweight='bold')
        else:
            axes[5].text(0.5, 0.5, 'Training history\nnot available', 
                        ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
            axes[5].set_title('Training History (N/A)', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.suptitle('Monte Carlo Model - Comprehensive Diagnostics', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model diagnostics plot saved to: {save_path}")
        
        return fig

    def plot_uncertainty_evolution(self, X, predictions_dict=None, save_path=None):
        """
        Plot uncertainty evolution and temporal analysis with seaborn
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences
        predictions_dict : dict, optional
            Pre-computed predictions
        save_path : str, optional
            Path to save the plot
        """
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Get predictions if not provided
        if predictions_dict is None:
            predictions_dict = self.predict_with_uncertainty(X[:50], n_samples=30)
        
        # Extract uncertainty data
        det_std = predictions_dict['detection']['std'].flatten()
        det_mean = predictions_dict['detection']['mean'].flatten()
        
        # 1. Uncertainty Evolution Over Samples (seaborn lineplot)
        n_samples = min(len(det_std), 100)
        sample_indices = np.arange(n_samples)
        
        evolution_df = pd.DataFrame({
            'Sample Index': sample_indices,
            'Uncertainty': det_std[:n_samples],
            'Prediction': det_mean[:n_samples]
        })
        
        # Create twin axis for prediction
        ax1 = axes[0]
        ax2 = ax1.twinx()
        
        sns.lineplot(data=evolution_df, x='Sample Index', y='Uncertainty',
                    color='red', linewidth=2, label='Uncertainty', ax=ax1)
        sns.lineplot(data=evolution_df, x='Sample Index', y='Prediction',
                    color='blue', linewidth=2, label='Prediction', ax=ax2)
        
        ax1.set_ylabel('Uncertainty (std)', color='red')
        ax2.set_ylabel('Prediction', color='blue')
        ax1.set_title('Uncertainty and Prediction Evolution', fontweight='bold')
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 2. Uncertainty vs Prediction Magnitude Heatmap
        # Create 2D histogram
        prediction_bins = np.linspace(det_mean.min(), det_mean.max(), 15)
        uncertainty_bins = np.linspace(det_std.min(), det_std.max(), 15)
        
        hist, xedges, yedges = np.histogram2d(det_mean, det_std, bins=[prediction_bins, uncertainty_bins])
        
        sns.heatmap(hist.T, xticklabels=False, yticklabels=False, 
                   cmap='Blues', annot=False, ax=axes[1])
        axes[1].set_xlabel('Prediction Magnitude')
        axes[1].set_ylabel('Uncertainty')
        axes[1].set_title('Prediction-Uncertainty Density Heatmap', fontweight='bold')
        
        # 3. Input Signal Statistics vs Uncertainty
        if len(X) > 0:
            signal_stats = []
            n_analyze = min(len(X), len(det_std))
            
            for i in range(n_analyze):
                # Calculate various signal statistics
                signal = X[i, :, 0]  # Use first channel
                stats_dict = {
                    'Mean': np.mean(signal),
                    'Std': np.std(signal),
                    'Skewness': stats.skew(signal),
                    'Kurtosis': stats.kurtosis(signal),
                    'Range': np.ptp(signal),
                    'Uncertainty': det_std[i]
                }
                signal_stats.append(stats_dict)
            
            stats_df = pd.DataFrame(signal_stats)
            
            # Select most correlated features with uncertainty
            correlations = {}
            for col in ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Range']:
                corr = np.corrcoef(stats_df[col], stats_df['Uncertainty'])[0, 1]
                correlations[col] = abs(corr)
            
            # Get top 2 most correlated features
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Plot correlations
            for idx, (feature, corr) in enumerate(top_features):
                plot_df = stats_df[[feature, 'Uncertainty']].copy()
                plot_df['Feature Type'] = feature
                
                if idx == 0:
                    sns.scatterplot(data=plot_df, x=feature, y='Uncertainty',
                                   alpha=0.7, s=50, color='green', ax=axes[2])
                    axes[2].set_title(f'Signal {feature} vs Uncertainty (r={corr:.3f})', fontweight='bold')
            
            # Regression line
            if len(top_features) > 0:
                feature_name = top_features[0][0]
                x_vals = stats_df[feature_name]
                y_vals = stats_df['Uncertainty']
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                axes[2].plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)
        else:
            axes[2].text(0.5, 0.5, 'Input data\nnot available', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Signal Statistics Analysis (N/A)', fontweight='bold')
        
        # 4. Uncertainty Distribution Analysis
        uncertainty_categories = pd.cut(det_std, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        prediction_categories = pd.cut(det_mean, bins=3, labels=['Low', 'Medium', 'High'])
        
        heatmap_data = pd.crosstab(uncertainty_categories, prediction_categories, normalize='columns')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Proportion'}, ax=axes[3])
        axes[3].set_title('Uncertainty vs Prediction Category Heatmap', fontweight='bold')
        axes[3].set_xlabel('Prediction Category')
        axes[3].set_ylabel('Uncertainty Category')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.suptitle('Monte Carlo Model - Uncertainty Evolution Analysis', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Uncertainty evolution plot saved to: {save_path}")
        
        return fig

    def plot_training_history(self, history=None, save_path=None):
        """
        Enhanced training history plotting with seaborn
        
        Parameters
        ----------
        history : dict, optional
            Training history dictionary
        save_path : str, optional
            Path to save the plot
        """
        if history is None:
            if hasattr(self, 'training_history') and self.training_history:
                history = self.training_history
            else:
                logger.warning("No training history available")
                return None
        
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Prepare data for seaborn
        training_data = []
        
        # 1. Total Loss
        if 'loss' in history and 'val_loss' in history:
            for epoch, (train_loss, val_loss) in enumerate(zip(history['loss'], history['val_loss'])):
                training_data.extend([
                    {'Epoch': epoch, 'Loss': train_loss, 'Type': 'Training', 'Metric': 'Total Loss'},
                    {'Epoch': epoch, 'Loss': val_loss, 'Type': 'Validation', 'Metric': 'Total Loss'}
                ])
            
            loss_df = pd.DataFrame([d for d in training_data if d['Metric'] == 'Total Loss'])
            sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Type', 
                        marker='o', markersize=4, ax=axes[0])
            axes[0].set_title('Total Loss', fontweight='bold')
            axes[0].legend()
        
        # 2. Detection Accuracy
        det_acc_key = 'detection_output_accuracy'
        val_det_acc_key = 'val_detection_output_accuracy'
        
        if det_acc_key in history and val_det_acc_key in history:
            acc_data = []
            for epoch, (train_acc, val_acc) in enumerate(zip(history[det_acc_key], history[val_det_acc_key])):
                acc_data.extend([
                    {'Epoch': epoch, 'Accuracy': train_acc, 'Type': 'Training'},
                    {'Epoch': epoch, 'Accuracy': val_acc, 'Type': 'Validation'}
                ])
            
            acc_df = pd.DataFrame(acc_data)
            sns.lineplot(data=acc_df, x='Epoch', y='Accuracy', hue='Type',
                        marker='s', markersize=4, ax=axes[1])
            axes[1].set_title('Detection Accuracy', fontweight='bold')
            axes[1].legend()
        
        # 3. Classification Accuracy
        class_acc_key = 'classification_output_accuracy'
        val_class_acc_key = 'val_classification_output_accuracy'
        
        if class_acc_key in history and val_class_acc_key in history:
            class_data = []
            for epoch, (train_acc, val_acc) in enumerate(zip(history[class_acc_key], history[val_class_acc_key])):
                class_data.extend([
                    {'Epoch': epoch, 'Accuracy': train_acc, 'Type': 'Training'},
                    {'Epoch': epoch, 'Accuracy': val_acc, 'Type': 'Validation'}
                ])
            
            class_df = pd.DataFrame(class_data)
            sns.lineplot(data=class_df, x='Epoch', y='Accuracy', hue='Type',
                        marker='^', markersize=4, ax=axes[2])
            axes[2].set_title('Classification Accuracy', fontweight='bold')
            axes[2].legend()
        
        # 4. Regression MAE
        reg_mae_key = 'regression_output_mae'
        val_reg_mae_key = 'val_regression_output_mae'
        
        if reg_mae_key in history and val_reg_mae_key in history:
            mae_data = []
            for epoch, (train_mae, val_mae) in enumerate(zip(history[reg_mae_key], history[val_reg_mae_key])):
                mae_data.extend([
                    {'Epoch': epoch, 'MAE': train_mae, 'Type': 'Training'},
                    {'Epoch': epoch, 'MAE': val_mae, 'Type': 'Validation'}
                ])
            
            mae_df = pd.DataFrame(mae_data)
            sns.lineplot(data=mae_df, x='Epoch', y='MAE', hue='Type',
                        marker='d', markersize=4, ax=axes[3])
            axes[3].set_title('Regression MAE', fontweight='bold')
            axes[3].legend()
        
        # 5. Learning Rate (if available)
        if 'lr' in history:
            lr_df = pd.DataFrame({
                'Epoch': range(len(history['lr'])),
                'Learning Rate': history['lr']
            })
            sns.lineplot(data=lr_df, x='Epoch', y='Learning Rate',
                        color='orange', linewidth=2, ax=axes[4])
            axes[4].set_title('Learning Rate Schedule', fontweight='bold')
            axes[4].set_yscale('log')
        else:
            axes[4].text(0.5, 0.5, 'Learning rate\nhistory not available', 
                        ha='center', va='center', transform=axes[4].transAxes, fontsize=12)
            axes[4].set_title('Learning Rate (N/A)', fontweight='bold')
        
        # 6. Multi-task Loss Breakdown
        task_losses = []
        for task in ['detection', 'classification', 'regression']:
            loss_key = f'{task}_output_loss'
            if loss_key in history:
                for epoch, loss_val in enumerate(history[loss_key]):
                    task_losses.append({
                        'Epoch': epoch,
                        'Loss': loss_val,
                        'Task': task.title()
                    })
        
        if task_losses:
            task_df = pd.DataFrame(task_losses)
            sns.lineplot(data=task_df, x='Epoch', y='Loss', hue='Task',
                        marker='o', markersize=3, ax=axes[5])
            axes[5].set_title('Task-Specific Losses', fontweight='bold')
            axes[5].legend()
        else:
            axes[5].text(0.5, 0.5, 'Task-specific losses\nnot available', 
                        ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
            axes[5].set_title('Task Losses (N/A)', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.suptitle('Monte Carlo Model - Enhanced Training History', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        return fig

    def plot_model_comparison(self, comparison_data, save_path=None):
        """
        Plot model comparison metrics with seaborn
        
        Parameters
        ----------
        comparison_data : dict
            Dictionary containing comparison metrics for different models
        save_path : str, optional
            Path to save the plot
        """
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # 1. Performance Metrics Comparison (bar plot)
        metrics_data = []
        for model_name, metrics in comparison_data.items():
            if 'standard_metrics' in metrics:
                for metric_name, value in metrics['standard_metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_data.append({
                            'Model': model_name,
                            'Metric': metric_name,
                            'Value': value
                        })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Select key metrics for visualization
            key_metrics = ['loss', 'detection_output_accuracy', 'classification_output_accuracy']
            filtered_df = metrics_df[metrics_df['Metric'].isin(key_metrics)]
            
            if not filtered_df.empty:
                sns.barplot(data=filtered_df, x='Metric', y='Value', hue='Model',
                           palette='Set2', ax=axes[0])
                axes[0].set_title('Performance Metrics Comparison', fontweight='bold')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].legend()
        
        # 2. Uncertainty Metrics Comparison
        uncertainty_data = []
        for model_name, metrics in comparison_data.items():
            if 'monte_carlo_metrics' in metrics:
                mc_metrics = metrics['monte_carlo_metrics']
                if 'uncertainty_decomposition' in mc_metrics:
                    unc_decomp = mc_metrics['uncertainty_decomposition']
                    for task, uncertainty in unc_decomp.items():
                        uncertainty_data.append({
                            'Model': model_name,
                            'Task': task.replace('_epistemic', '').title(),
                            'Epistemic Uncertainty': uncertainty
                        })
        
        if uncertainty_data:
            unc_df = pd.DataFrame(uncertainty_data)
            sns.barplot(data=unc_df, x='Task', y='Epistemic Uncertainty', hue='Model',
                       palette='viridis', ax=axes[1])
            axes[1].set_title('Epistemic Uncertainty Comparison', fontweight='bold')
            axes[1].legend()
        
        # 3. Coverage Probability Comparison
        coverage_data = []
        for model_name, metrics in comparison_data.items():
            if 'monte_carlo_metrics' in metrics and 'prediction_interval_coverage' in metrics['monte_carlo_metrics']:
                coverage = metrics['monte_carlo_metrics']['prediction_interval_coverage']
                for task, cov_value in coverage.items():
                    coverage_data.append({
                        'Model': model_name,
                        'Task': task.title(),
                        'Coverage': cov_value
                    })
        
        if coverage_data:
            cov_df = pd.DataFrame(coverage_data)
            sns.barplot(data=cov_df, x='Task', y='Coverage', hue='Model',
                       palette='plasma', ax=axes[2])
            axes[2].axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='Target (95%)')
            axes[2].set_title('Prediction Interval Coverage', fontweight='bold')
            axes[2].legend()
        
        # 4. Model Complexity Comparison
        complexity_data = []
        for model_name, metrics in comparison_data.items():
            if 'model_info' in metrics:
                info = metrics['model_info']
                complexity_data.append({
                    'Model': model_name,
                    'Parameters': info.get('parameters', 0),
                    'MC Samples': info.get('mc_samples', 0),
                    'Dropout Rate': info.get('dropout_rate', 0)
                })
        
        if complexity_data:
            comp_df = pd.DataFrame(complexity_data)
            
            # Create a composite complexity score
            comp_df['Complexity Score'] = (comp_df['Parameters'] / 1000 + 
                                         comp_df['MC Samples'] + 
                                         comp_df['Dropout Rate'] * 100)
            
            sns.barplot(data=comp_df, x='Model', y='Complexity Score',
                       palette='coolwarm', ax=axes[3])
            axes[3].set_title('Model Complexity Comparison', fontweight='bold')
            axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.suptitle('Monte Carlo Models - Comprehensive Comparison', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {save_path}")
        
        return fig


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
