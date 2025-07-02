"""
Prediction Module for Solar Flare Analysis

This module loads trained models and makes predictions on new solar flare data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
import joblib
import warnings
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

# Optional imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class NanoflareDetector:
    """
    Solar nanoflare detection based on alpha power-law analysis
    """
    
    def __init__(self, alpha_threshold=1.63, alpha_uncertainty=0.03):
        """
        Initialize nanoflare detector
        
        Parameters:
        -----------
        alpha_threshold : float
            Alpha threshold for nanoflare detection (default: 1.63)
        alpha_uncertainty : float
            Uncertainty in alpha threshold (default: ±0.03)
        """
        self.alpha_threshold = alpha_threshold
        self.alpha_uncertainty = alpha_uncertainty
        self.alpha_min = alpha_threshold - alpha_uncertainty
        self.alpha_max = alpha_threshold + alpha_uncertainty
    
    def smooth_gaussian_curve(self, time, flux, sigma=2.0):
        """
        Smooth the Gaussian fit curve using Gaussian filter
        
        Parameters:
        -----------
        time : array-like
            Time array
        flux : array-like
            Flux array from Gaussian fit
        sigma : float
            Gaussian filter sigma parameter
            
        Returns:
        --------
        tuple
            (smoothed_time, smoothed_flux)
        """
        # Apply Gaussian smoothing
        smoothed_flux = gaussian_filter1d(flux, sigma=sigma)
        
        return time, smoothed_flux
    
    def calculate_power_law_index(self, energies, frequencies, smooth=True, sigma=1.5):
        """
        Calculate power-law index (alpha) from flare frequency distribution
        
        Parameters:
        -----------
        energies : array-like
            Flare energies (or peak flux values)
        frequencies : array-like
            Frequency of occurrence for each energy bin
        smooth : bool
            Whether to smooth the frequency distribution
        sigma : float
            Smoothing parameter
            
        Returns:
        --------
        dict
            Dictionary containing alpha and fit quality metrics
        """
        # Remove zero frequencies and corresponding energies
        mask = (frequencies > 0) & (energies > 0)
        energies_clean = np.array(energies)[mask]
        frequencies_clean = np.array(frequencies)[mask]
        
        if len(energies_clean) < 3:
            return {
                'alpha': np.nan,
                'alpha_error': np.nan,
                'r_squared': np.nan,
                'fit_quality': 'insufficient_data'
            }
        
        # Smooth frequency distribution if requested
        if smooth and len(frequencies_clean) > 5:
            frequencies_clean = gaussian_filter1d(frequencies_clean, sigma=sigma)
        
        # Take logarithms for power-law fitting
        log_energies = np.log10(energies_clean)
        log_frequencies = np.log10(frequencies_clean)
        
        try:
            # Fit power law: log(frequency) = -alpha * log(energy) + const
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_energies, log_frequencies)
            
            # Alpha is the negative of the slope
            alpha = -slope
            alpha_error = std_err
            r_squared = r_value**2
            
            # Determine fit quality
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
                'alpha_error': alpha_error,
                'r_squared': r_squared,
                'p_value': p_value,
                'fit_quality': fit_quality,
                'slope': slope,
                'intercept': intercept,
                'n_points': len(energies_clean)
            }
            
        except Exception as e:
            return {
                'alpha': np.nan,
                'alpha_error': np.nan,
                'r_squared': np.nan,
                'fit_quality': f'error: {str(e)}'
            }
    
    def detect_nanoflares(self, alpha_result, confidence_level=0.95):
        """
        Detect nanoflares based on alpha threshold criteria
        
        Parameters:
        -----------
        alpha_result : dict
            Result from calculate_power_law_index
        confidence_level : float
            Confidence level for detection
            
        Returns:
        --------
        dict
            Nanoflare detection results
        """
        alpha = alpha_result.get('alpha', np.nan)
        alpha_error = alpha_result.get('alpha_error', np.nan)
        
        if np.isnan(alpha):
            return {
                'is_nanoflare': False,
                'detection_confidence': 0.0,
                'reason': 'alpha_calculation_failed',
                'alpha_value': alpha,
                'threshold_comparison': 'undefined'
            }
        
        # Check if alpha is greater than threshold
        alpha_above_threshold = alpha > self.alpha_threshold
        
        # Calculate confidence considering uncertainties
        if not np.isnan(alpha_error):
            # Use error propagation for confidence calculation
            combined_uncertainty = np.sqrt(alpha_error**2 + self.alpha_uncertainty**2)
            
            # Calculate z-score
            z_score = (alpha - self.alpha_threshold) / combined_uncertainty
            
            # Convert to confidence (using normal distribution)
            from scipy.stats import norm
            if alpha_above_threshold:
                detection_confidence = norm.cdf(z_score)
            else:
                detection_confidence = norm.cdf(-z_score)
        else:
            # Simple threshold comparison without error consideration
            detection_confidence = 0.8 if alpha_above_threshold else 0.2
        
        # Determine if this qualifies as nanoflare detection
        is_nanoflare = (alpha_above_threshold and 
                       detection_confidence >= confidence_level and
                       alpha_result.get('r_squared', 0) > 0.5)
        
        # Prepare detailed comparison
        threshold_comparison = {
            'alpha_value': alpha,
            'alpha_error': alpha_error,
            'threshold': self.alpha_threshold,
            'threshold_uncertainty': self.alpha_uncertainty,
            'difference': alpha - self.alpha_threshold,
            'sigma_difference': (alpha - self.alpha_threshold) / combined_uncertainty if not np.isnan(alpha_error) else np.nan
        }
        
        return {
            'is_nanoflare': is_nanoflare,
            'detection_confidence': detection_confidence,
            'alpha_above_threshold': alpha_above_threshold,
            'reason': self._get_detection_reason(alpha_result, alpha_above_threshold, detection_confidence),
            'threshold_comparison': threshold_comparison,
            'fit_quality': alpha_result.get('fit_quality', 'unknown')
        }
    
    def _get_detection_reason(self, alpha_result, alpha_above_threshold, confidence):
        """Get human-readable reason for detection result"""
        if alpha_result.get('fit_quality') == 'poor':
            return 'poor_power_law_fit'
        elif not alpha_above_threshold:
            return 'alpha_below_threshold'
        elif confidence < 0.95:
            return 'low_confidence'
        else:
            return 'nanoflare_detected'
    
    def analyze_flare_population(self, flare_data, energy_column='peak_flux', 
                               n_bins=20, min_energy=None, max_energy=None):
        """
        Analyze a population of flares to detect nanoflare activity
        
        Parameters:
        -----------
        flare_data : pandas.DataFrame
            DataFrame containing flare data
        energy_column : str
            Column name for energy proxy (e.g., 'peak_flux', 'integrated_flux')
        n_bins : int
            Number of energy bins for frequency distribution
        min_energy : float, optional
            Minimum energy for analysis
        max_energy : float, optional
            Maximum energy for analysis
            
        Returns:
        --------
        dict
            Population analysis results
        """
        if energy_column not in flare_data.columns:
            return {
                'error': f"Column '{energy_column}' not found in data",
                'available_columns': list(flare_data.columns)
            }
        
        # Get energy values
        energies = flare_data[energy_column].dropna()
        
        if len(energies) < 10:
            return {
                'error': 'Insufficient data for population analysis',
                'n_flares': len(energies)
            }
        
        # Filter energy range if specified
        if min_energy is not None:
            energies = energies[energies >= min_energy]
        if max_energy is not None:
            energies = energies[energies <= max_energy]
        
        # Create energy bins and calculate frequency distribution
        if len(energies) > 0:
            # Use log-spaced bins for better power-law analysis
            energy_min = energies.min()
            energy_max = energies.max()
            
            if energy_min <= 0:
                energy_min = energies[energies > 0].min() if any(energies > 0) else 1e-10
            
            energy_bins = np.logspace(np.log10(energy_min), np.log10(energy_max), n_bins + 1)
            frequencies, bin_edges = np.histogram(energies, bins=energy_bins)
            
            # Use bin centers for energy values
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
            
            # Calculate power-law index
            alpha_result = self.calculate_power_law_index(bin_centers, frequencies)
            
            # Detect nanoflares
            nanoflare_result = self.detect_nanoflares(alpha_result)
            
            return {
                'n_flares': len(energies),
                'energy_range': (energy_min, energy_max),
                'energy_bins': bin_centers,
                'frequencies': frequencies,
                'alpha_analysis': alpha_result,
                'nanoflare_detection': nanoflare_result,
                'analysis_successful': True
            }
        else:
            return {
                'error': 'No valid energy data after filtering',
                'n_flares': 0
            }
    
    def plot_power_law_analysis(self, analysis_result, output_path=None, title=None):
        """
        Create visualization of power-law analysis and nanoflare detection
        
        Parameters:
        -----------
        analysis_result : dict
            Result from analyze_flare_population
        output_path : str, optional
            Path to save the plot
        title : str, optional
            Custom title for the plot
        """
        if not analysis_result.get('analysis_successful', False):
            print(f"Cannot plot: {analysis_result.get('error', 'Analysis failed')}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        energies = analysis_result['energy_bins']
        frequencies = analysis_result['frequencies']
        alpha_result = analysis_result['alpha_analysis']
        nanoflare_result = analysis_result['nanoflare_detection']
        
        # Plot 1: Energy distribution (linear scale)
        axes[0, 0].hist(energies, weights=frequencies, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Energy (proxy)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Flare Energy Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Power-law fit (log-log scale)
        mask = frequencies > 0
        log_energies = np.log10(energies[mask])
        log_frequencies = np.log10(frequencies[mask])
        
        axes[0, 1].scatter(log_energies, log_frequencies, alpha=0.7, s=50)
        
        # Add fit line if successful
        if not np.isnan(alpha_result.get('alpha', np.nan)):
            slope = alpha_result['slope']
            intercept = alpha_result['intercept']
            fit_line = slope * log_energies + intercept
            axes[0, 1].plot(log_energies, fit_line, 'r-', linewidth=2, 
                           label=f'α = {alpha_result["alpha"]:.3f} ± {alpha_result.get("alpha_error", 0):.3f}')
            axes[0, 1].legend()
        
        axes[0, 1].set_xlabel('log₁₀(Energy)')
        axes[0, 1].set_ylabel('log₁₀(Frequency)')
        axes[0, 1].set_title(f'Power-Law Fit (R² = {alpha_result.get("r_squared", 0):.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Alpha threshold comparison
        alpha_val = alpha_result.get('alpha', np.nan)
        alpha_err = alpha_result.get('alpha_error', 0)
        
        x_pos = [0, 1]
        y_vals = [self.alpha_threshold, alpha_val if not np.isnan(alpha_val) else 0]
        y_errs = [self.alpha_uncertainty, alpha_err]
        colors = ['red', 'green' if nanoflare_result['is_nanoflare'] else 'orange']
        labels = ['Threshold', 'Measured α']
        
        bars = axes[1, 0].bar(x_pos, y_vals, yerr=y_errs, color=colors, alpha=0.7, 
                             capsize=5, error_kw={'linewidth': 2})
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(labels)
        axes[1, 0].set_ylabel('Alpha Value')
        axes[1, 0].set_title('Alpha Threshold Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val, err) in enumerate(zip(bars, y_vals, y_errs)):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Detection summary
        axes[1, 1].axis('off')
        
        # Create summary text
        summary_text = "NANOFLARE DETECTION SUMMARY\n" + "="*35 + "\n\n"
        summary_text += f"Number of flares analyzed: {analysis_result['n_flares']}\n"
        summary_text += f"Alpha threshold: {self.alpha_threshold:.3f} ± {self.alpha_uncertainty:.3f}\n"
        summary_text += f"Measured alpha: {alpha_val:.3f} ± {alpha_err:.3f}\n"
        summary_text += f"Fit quality: {alpha_result.get('fit_quality', 'unknown')}\n"
        summary_text += f"R²: {alpha_result.get('r_squared', 0):.3f}\n\n"
        
        # Detection result
        if nanoflare_result['is_nanoflare']:
            summary_text += "🟢 NANOFLARES DETECTED\n"
            summary_text += f"Confidence: {nanoflare_result['detection_confidence']:.1%}\n"
        else:
            summary_text += "🔴 NO NANOFLARES DETECTED\n"
            summary_text += f"Reason: {nanoflare_result['reason']}\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Solar Nanoflare Detection Analysis', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Power-law analysis plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


class SolarFlarePredictor:
    """
    Main prediction class for solar flare analysis
    """
    def __init__(self, models_dir, scaler_path=None, alpha_threshold=1.63, alpha_uncertainty=0.03):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.task = None
        
        # Initialize nanoflare detector
        self.nanoflare_detector = NanoflareDetector(alpha_threshold, alpha_uncertainty)
        
        # Load scaler if provided
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        
        self.load_models()
    
    def load_models(self):
        """Load all available trained models"""
        models_dir = self.models_dir
        
        # Load Random Forest
        rf_path = models_dir / 'random_forest_model.pkl'
        if rf_path.exists():
            try:
                self.models['random_forest'] = joblib.load(rf_path)
                print("Loaded Random Forest model")
            except Exception as e:
                print(f"Error loading Random Forest: {e}")
        
        # Load XGBoost
        xgb_path = models_dir / 'xgboost_model.pkl'
        if xgb_path.exists():
            try:
                self.models['xgboost'] = joblib.load(xgb_path)
                print("Loaded XGBoost model")
            except Exception as e:
                print(f"Error loading XGBoost: {e}")
        
        # Load Bayesian Neural Network
        bnn_path = models_dir / 'bayesian_nn_model'
        if bnn_path.exists() and TF_AVAILABLE:
            try:
                self.models['bayesian_nn'] = tf.keras.models.load_model(bnn_path)
                print("Loaded Bayesian Neural Network model")
            except Exception as e:
                print(f"Error loading Bayesian NN: {e}")
        
        # Load training results to get metadata
        results_path = models_dir / 'training_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.training_results = json.load(f)
        else:
            self.training_results = {}
        
        if not self.models:
            raise ValueError(f"No models found in {models_dir}")
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def preprocess_features(self, features):
        """
        Preprocess features for prediction
        
        Parameters:
        -----------
        features : dict or pandas.DataFrame
            Feature dictionary or DataFrame
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed feature array
        """
        if isinstance(features, dict):
            # Convert single feature dict to DataFrame
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Handle missing feature names
        if self.feature_names is not None:
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                print(f"Warning: Missing features {missing_features}. Filling with zeros.")
                for feature in missing_features:
                    features_df[feature] = 0
            
            # Select only the features used during training
            features_df = features_df[self.feature_names]
        
        # Handle infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)  # or use median from training
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_df.values)
        else:
            features_scaled = features_df.values
        
        return features_scaled
    
    def predict_single_model(self, model_name, features, with_uncertainty=False):
        """
        Make prediction with a single model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        features : numpy.ndarray
            Preprocessed features
        with_uncertainty : bool
            Whether to return uncertainty estimates (Bayesian NN only)
            
        Returns:
        --------
        dict
            Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        if model_name == 'bayesian_nn':
            if with_uncertainty:
                # Multiple forward passes for uncertainty estimation
                predictions = []
                n_samples = 100
                for _ in range(n_samples):
                    pred = model(features, training=True)
                    predictions.append(pred.numpy())
                
                predictions = np.array(predictions)
                mean_pred = np.mean(predictions, axis=0).flatten()
                std_pred = np.std(predictions, axis=0).flatten()
                
                return {
                    'prediction': mean_pred,
                    'uncertainty': std_pred,
                    'confidence_interval_95': {
                        'lower': mean_pred - 1.96 * std_pred,
                        'upper': mean_pred + 1.96 * std_pred
                    }
                }
            else:
                prediction = model.predict(features).flatten()
                return {'prediction': prediction}
        
        else:
            # Standard sklearn-like models
            prediction = model.predict(features)
            
            result = {'prediction': prediction}
            
            # Add probability estimates for classification
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features)
                    result['probabilities'] = probabilities
                except:
                    pass
            
            return result
    
    def predict_ensemble(self, features, weights=None):
        """
        Make ensemble prediction using all available models
        
        Parameters:
        -----------
        features : numpy.ndarray
            Preprocessed features
        weights : dict, optional
            Weights for each model in ensemble
            
        Returns:
        --------
        dict
            Ensemble prediction results
        """
        if weights is None:
            # Equal weights for all models
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        for model_name in self.models.keys():
            if model_name in weights:
                pred_result = self.predict_single_model(model_name, features)
                predictions[model_name] = pred_result
                
                weight = weights[model_name]
                weighted_sum += weight * pred_result['prediction']
                total_weight += weight
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'weights': weights
        }
    
    def predict_from_flare_params(self, flare_parameters, model_name=None):
        """
        Make prediction from fitted flare parameters
        
        Parameters:
        -----------
        flare_parameters : dict
            Dictionary containing fitted parameters A, B, C, D
        model_name : str, optional
            Specific model to use. If None, uses ensemble
            
        Returns:
        --------
        dict
            Prediction results
        """
        # Extract features from parameters (basic feature extraction)
        features = self.extract_basic_features(flare_parameters)
        features_df = pd.DataFrame([features])
        
        # Preprocess
        features_processed = self.preprocess_features(features_df)
        
        # Make prediction
        if model_name is not None:
            return self.predict_single_model(model_name, features_processed)
        else:
            return self.predict_ensemble(features_processed)
    
    def extract_basic_features(self, parameters):
        """
        Extract basic features from fitted parameters
        (Simplified version of feature extraction)
        """
        features = {}
        
        # Direct parameters
        features['amplitude'] = parameters.get('A', 0)
        features['peak_time'] = parameters.get('B', 0)
        features['width'] = parameters.get('C', 1)
        features['decay_rate'] = parameters.get('D', 1e-6)
        
        # Derived quantities
        features['decay_time'] = 1.0 / features['decay_rate'] if features['decay_rate'] > 0 else 1000
        features['total_duration'] = features['peak_time'] + 3 * features['decay_time']
        features['peak_flux'] = features['amplitude']
        features['integrated_flux'] = features['amplitude'] * features['width'] * np.sqrt(np.pi)
        features['asymmetry'] = features['decay_time'] / features['peak_time'] if features['peak_time'] > 0 else 1
        features['sharpness'] = features['amplitude'] / features['width'] if features['width'] > 0 else 0
        
        # Logarithmic features
        for param in ['amplitude', 'width', 'decay_rate']:
            if features[param] > 0:
                features[f'log_{param}'] = np.log10(features[param])
            else:
                features[f'log_{param}'] = -10
        
        return features
    
    def batch_predict(self, input_file, output_file=None, model_name=None):
        """
        Make predictions on a batch of flare data
        
        Parameters:
        -----------
        input_file : str
            Path to JSON file containing flare fit results
        output_file : str, optional
            Path to save predictions
        model_name : str, optional
            Specific model to use
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with predictions
        """
        # Load input data
        with open(input_file, 'r') as f:
            flare_data = json.load(f)
        
        if not isinstance(flare_data, list):
            flare_data = [flare_data]
        
        predictions_list = []
        
        for i, flare in enumerate(flare_data):
            if 'parameters' in flare:
                try:
                    pred_result = self.predict_from_flare_params(
                        flare['parameters'], model_name
                    )
                    
                    # Prepare result dictionary
                    result = {
                        'flare_id': flare.get('flare_id', i),
                        'file': flare.get('file', 'unknown')
                    }
                    
                    if 'ensemble_prediction' in pred_result:
                        result['prediction'] = pred_result['ensemble_prediction'][0]
                        # Add individual model predictions
                        for mname, mpred in pred_result['individual_predictions'].items():
                            result[f'{mname}_prediction'] = mpred['prediction'][0]
                    else:
                        result['prediction'] = pred_result['prediction'][0]
                        if 'uncertainty' in pred_result:
                            result['uncertainty'] = pred_result['uncertainty'][0]
                    
                    predictions_list.append(result)
                
                except Exception as e:
                    print(f"Error predicting flare {i}: {e}")
                    continue
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions_list)
        
        # Save if output file specified
        if output_file:
            predictions_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        
        return predictions_df
    
    def visualize_predictions(self, predictions_df, output_dir=None):
        """
        Create visualizations of prediction results
        """
        if predictions_df.empty:
            print("No predictions to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prediction distribution
        axes[0, 0].hist(predictions_df['prediction'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Predicted Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty plot (if available)
        if 'uncertainty' in predictions_df.columns:
            axes[0, 1].scatter(predictions_df['prediction'], predictions_df['uncertainty'], alpha=0.6)
            axes[0, 1].set_xlabel('Prediction')
            axes[0, 1].set_ylabel('Uncertainty')
            axes[0, 1].set_title('Prediction vs Uncertainty')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Uncertainty data\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Uncertainty Analysis')
        
        # Model comparison (if multiple models)
        model_cols = [col for col in predictions_df.columns if col.endswith('_prediction')]
        if len(model_cols) > 1:
            for col in model_cols:
                model_name = col.replace('_prediction', '')
                axes[1, 0].scatter(predictions_df['prediction'], predictions_df[col], 
                                 alpha=0.6, label=model_name)
            axes[1, 0].plot([predictions_df['prediction'].min(), predictions_df['prediction'].max()],
                          [predictions_df['prediction'].min(), predictions_df['prediction'].max()],
                          'k--', alpha=0.5)
            axes[1, 0].set_xlabel('Ensemble Prediction')
            axes[1, 0].set_ylabel('Individual Model Prediction')
            axes[1, 0].set_title('Model Agreement')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Multiple models\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Model Comparison')
        
        # Prediction confidence intervals (if uncertainty available)
        if 'uncertainty' in predictions_df.columns:
            sorted_indices = np.argsort(predictions_df['prediction'])
            x_sorted = predictions_df['prediction'].iloc[sorted_indices]
            uncertainty_sorted = predictions_df['uncertainty'].iloc[sorted_indices]
            
            axes[1, 1].plot(x_sorted, x_sorted, 'b-', label='Prediction')
            axes[1, 1].fill_between(x_sorted, 
                                   x_sorted - 1.96 * uncertainty_sorted,
                                   x_sorted + 1.96 * uncertainty_sorted,
                                   alpha=0.3, label='95% Confidence')
            axes[1, 1].set_xlabel('Sorted Predictions')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Prediction Confidence Intervals')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Summary statistics
            stats_text = f"Count: {len(predictions_df)}\n"
            stats_text += f"Mean: {predictions_df['prediction'].mean():.3f}\n"
            stats_text += f"Std: {predictions_df['prediction'].std():.3f}\n"
            stats_text += f"Min: {predictions_df['prediction'].min():.3f}\n"
            stats_text += f"Max: {predictions_df['prediction'].max():.3f}"
            
            axes[1, 1].text(0.1, 0.7, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_title('Prediction Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / 'prediction_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

    def analyze_nanoflares_from_predictions(self, predictions_df, energy_column='prediction', 
                                          output_dir=None, plot_analysis=True):
        """
        Analyze predictions for nanoflare activity using alpha threshold method
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame with alpha predictions
        energy_column : str
            Column to use as energy proxy
        output_dir : str, optional
            Directory to save analysis results
        plot_analysis : bool
            Whether to create analysis plots
            
        Returns:
        --------
        dict
            Nanoflare analysis results
        """
        print("Analyzing flare population for nanoflare activity...")
        
        # Perform population analysis
        analysis_result = self.nanoflare_detector.analyze_flare_population(
            predictions_df, 
            energy_column=energy_column,
            n_bins=15,  # Reduced bins for better statistics
            min_energy=None,
            max_energy=None
        )
        
        if not analysis_result.get('analysis_successful', False):
            print(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")
            return analysis_result
        
        # Print summary
        alpha_result = analysis_result['alpha_analysis']
        nanoflare_result = analysis_result['nanoflare_detection']
        
        print(f"\n{'='*50}")
        print("SOLAR NANOFLARE DETECTION ANALYSIS")
        print(f"{'='*50}")
        print(f"Number of flares analyzed: {analysis_result['n_flares']}")
        print(f"Energy range: {analysis_result['energy_range'][0]:.3e} to {analysis_result['energy_range'][1]:.3e}")
        print(f"\nPower-law fit results:")
        print(f"  Alpha (power-law index): {alpha_result.get('alpha', 'N/A'):.3f} ± {alpha_result.get('alpha_error', 0):.3f}")
        print(f"  R-squared: {alpha_result.get('r_squared', 0):.3f}")
        print(f"  Fit quality: {alpha_result.get('fit_quality', 'unknown')}")
        print(f"\nNanoflare detection:")
        print(f"  Alpha threshold: {self.nanoflare_detector.alpha_threshold:.3f} ± {self.nanoflare_detector.alpha_uncertainty:.3f}")
        print(f"  Alpha above threshold: {nanoflare_result.get('alpha_above_threshold', False)}")
        print(f"  Detection confidence: {nanoflare_result.get('detection_confidence', 0):.1%}")
        
        if nanoflare_result.get('is_nanoflare', False):
            print("  🟢 NANOFLARES DETECTED!")
        else:
            print(f"  🔴 No nanoflares detected ({nanoflare_result.get('reason', 'unknown reason')})")
        
        # Create visualizations
        if plot_analysis and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            plot_path = output_dir / 'nanoflare_analysis.png'
            self.nanoflare_detector.plot_power_law_analysis(
                analysis_result, 
                output_path=plot_path,
                title="Solar Nanoflare Detection Analysis"
            )
        
        # Save detailed results
        if output_dir:
            output_dir = Path(output_dir)
            results_path = output_dir / 'nanoflare_analysis_results.json'
            
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(analysis_result)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nDetailed results saved to {results_path}")
        
        return analysis_result
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def batch_predict_with_nanoflare_analysis(self, input_file, output_file=None, 
                                            model_name=None, analyze_nanoflares=True,
                                            output_dir=None):
        """
        Enhanced batch prediction with automatic nanoflare analysis
        
        Parameters:
        -----------
        input_file : str
            Path to JSON file containing flare fit results
        output_file : str, optional
            Path to save predictions
        model_name : str, optional
            Specific model to use
        analyze_nanoflares : bool
            Whether to perform nanoflare analysis
        output_dir : str, optional
            Directory for analysis outputs
            
        Returns:
        --------
        tuple
            (predictions_df, nanoflare_analysis_result)
        """
        # Make standard predictions
        predictions_df = self.batch_predict(input_file, output_file, model_name)
        
        nanoflare_analysis = None
        
        if analyze_nanoflares and not predictions_df.empty:
            # Perform nanoflare analysis using predictions as energy proxy
            nanoflare_analysis = self.analyze_nanoflares_from_predictions(
                predictions_df,
                energy_column='prediction',  # Use alpha predictions
                output_dir=output_dir,
                plot_analysis=True
            )
        
        return predictions_df, nanoflare_analysis

def main():
    """
    Main prediction function with nanoflare analysis
    """
    parser = argparse.ArgumentParser(description='Make predictions on solar flare data with nanoflare analysis')
    parser.add_argument('--models_dir', required=True, help='Directory containing trained models')
    parser.add_argument('--input', required=True, help='Input file (JSON with fit results)')
    parser.add_argument('--output', help='Output CSV file for predictions')
    parser.add_argument('--model', choices=['random_forest', 'xgboost', 'bayesian_nn', 'ensemble'],
                       default='ensemble', help='Model to use for prediction')
    parser.add_argument('--visualize', action='store_true', help='Create prediction visualizations')
    parser.add_argument('--output_dir', default='output', help='Output directory for visualizations')
    parser.add_argument('--analyze_nanoflares', action='store_true', 
                       help='Perform nanoflare detection analysis')
    parser.add_argument('--alpha_threshold', type=float, default=1.63,
                       help='Alpha threshold for nanoflare detection (default: 1.63)')
    parser.add_argument('--alpha_uncertainty', type=float, default=0.03,
                       help='Alpha threshold uncertainty (default: ±0.03)')
    
    args = parser.parse_args()
    
    # Initialize predictor with nanoflare detection parameters
    try:
        predictor = SolarFlarePredictor(
            args.models_dir, 
            alpha_threshold=args.alpha_threshold,
            alpha_uncertainty=args.alpha_uncertainty
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Make predictions
    print(f"Making predictions on {args.input}")
    
    model_name = args.model if args.model != 'ensemble' else None
    
    try:
        if args.analyze_nanoflares:
            # Enhanced prediction with nanoflare analysis
            predictions_df, nanoflare_analysis = predictor.batch_predict_with_nanoflare_analysis(
                args.input, 
                output_file=args.output,
                model_name=model_name,
                analyze_nanoflares=True,
                output_dir=args.output_dir
            )
        else:
            # Standard prediction only
            predictions_df = predictor.batch_predict(
                args.input, 
                output_file=args.output,
                model_name=model_name
            )
            nanoflare_analysis = None
        
        print(f"Made predictions for {len(predictions_df)} flares")
        
        # Display prediction summary
        if not predictions_df.empty:
            print(f"\nPrediction Summary:")
            print(f"Mean α: {predictions_df['prediction'].mean():.4f}")
            print(f"Std α:  {predictions_df['prediction'].std():.4f}")
            print(f"Min α:  {predictions_df['prediction'].min():.4f}")
            print(f"Max α:  {predictions_df['prediction'].max():.4f}")
        
        # Create standard visualizations
        if args.visualize:
            predictor.visualize_predictions(predictions_df, args.output_dir)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
