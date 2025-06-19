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
warnings.filterwarnings('ignore')

# Optional imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class SolarFlarePredictor:
    """
    Main prediction class for solar flare analysis
    """
    
    def __init__(self, models_dir, scaler_path=None):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.task = None
        
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

def main():
    """
    Main prediction function
    """
    parser = argparse.ArgumentParser(description='Make predictions on solar flare data')
    parser.add_argument('--models_dir', required=True, help='Directory containing trained models')
    parser.add_argument('--input', required=True, help='Input file (JSON with fit results)')
    parser.add_argument('--output', help='Output CSV file for predictions')
    parser.add_argument('--model', choices=['random_forest', 'xgboost', 'bayesian_nn', 'ensemble'],
                       default='ensemble', help='Model to use for prediction')
    parser.add_argument('--visualize', action='store_true', help='Create prediction visualizations')
    parser.add_argument('--output_dir', default='output', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = SolarFlarePredictor(args.models_dir)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Make predictions
    print(f"Making predictions on {args.input}")
    
    model_name = args.model if args.model != 'ensemble' else None
    
    try:
        predictions_df = predictor.batch_predict(
            args.input, 
            output_file=args.output,
            model_name=model_name
        )
        
        print(f"Made predictions for {len(predictions_df)} flares")
        
        # Display summary
        if not predictions_df.empty:
            print(f"\nPrediction Summary:")
            print(f"Mean: {predictions_df['prediction'].mean():.4f}")
            print(f"Std:  {predictions_df['prediction'].std():.4f}")
            print(f"Min:  {predictions_df['prediction'].min():.4f}")
            print(f"Max:  {predictions_df['prediction'].max():.4f}")
        
        # Create visualizations
        if args.visualize:
            predictor.visualize_predictions(predictions_df, args.output_dir)
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
