"""
Machine Learning Training Module for Solar Flare Analysis

This module implements various ML models for predicting flare energy distribution
power-law index (α) or nanoflare classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced models
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Bayesian Neural Networks will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Advanced neural networks will be disabled.")

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network using TensorFlow Probability
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1, task='regression'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task = task
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build the Bayesian neural network model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for Bayesian Neural Network")
        
        # Define prior distribution
        def prior_fn(dtype, shape, name, trainable, add_variable_fn):
            """Prior distribution for weights"""
            return tfp.distributions.Normal(
                loc=tf.zeros(shape, dtype), 
                scale=tf.ones(shape, dtype)
            )
        
        # Define posterior distribution
        def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
            """Posterior distribution for weights"""
            loc = add_variable_fn(
                name=name + '_loc',
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                initializer='zeros'
            )
            scale = add_variable_fn(
                name=name + '_scale',
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                initializer=lambda shape, dtype: tf.random.normal(shape, mean=-3, stddev=0.1, dtype=dtype)
            )
            return tfp.distributions.Normal(loc=loc, scale=tf.nn.softplus(scale))
        
        # Build model
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(self.input_dim,)))
        
        # Hidden layers with variational weights
        for hidden_dim in self.hidden_dims:
            model.add(tfp.layers.DenseVariational(
                hidden_dim,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_weight=1/1000,  # Adjust based on dataset size
                activation='relu'
            ))
            model.add(keras.layers.Dropout(0.1))
        
        # Output layer
        if self.task == 'regression':
            model.add(tfp.layers.DenseVariational(
                self.output_dim,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_weight=1/1000
            ))
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            model.add(tfp.layers.DenseVariational(
                self.output_dim,
                make_prior_fn=prior_fn,
                make_posterior_fn=posterior_fn,
                kl_weight=1/1000,
                activation='sigmoid' if self.output_dim == 1 else 'softmax'
            ))
            loss = 'binary_crossentropy' if self.output_dim == 1 else 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the Bayesian neural network"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Make predictions with uncertainty estimates"""
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std

class XGBoostModel:
    """
    XGBoost model wrapper for regression and classification
    """
    
    def __init__(self, task='regression', **kwargs):
        self.task = task
        self.model = None
        self.best_params = None
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.params = default_params
    
    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """Tune hyperparameters using grid search"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        if self.task == 'regression':
            model = xgb.XGBRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        else:
            model = xgb.XGBClassifier(random_state=42)
            scoring = 'accuracy'
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return self.best_params
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        if self.model is None:
            if self.task == 'regression':
                self.model = xgb.XGBRegressor(**self.params)
            else:
                self.model = xgb.XGBClassifier(**self.params)
        
        # Add validation set for early stopping if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=False
        )
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.feature_importances_

class RandomForestModel:
    """
    Random Forest model wrapper
    """
    
    def __init__(self, task='regression', **kwargs):
        self.task = task
        self.model = None
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.params = default_params
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        if self.task == 'regression':
            self.model = RandomForestRegressor(**self.params)
        else:
            self.model = RandomForestClassifier(**self.params)
        
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions (classification only)"""
        if self.task == 'classification':
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba only available for classification")
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.feature_importances_

class ModelTrainer:
    """
    Main class for training and evaluating multiple models
    """
    
    def __init__(self, task='regression'):
        self.task = task
        self.models = {}
        self.results = {}
        self.feature_names = None
    
    def load_data(self, data_path, target='alpha'):
        """Load preprocessed data"""
        data = np.load(data_path)
        
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.feature_names = data['feature_names']
        
        print(f"Loaded data: {self.X_train.shape[0]} training, {self.X_test.shape[0]} test samples")
        print(f"Features: {self.X_train.shape[1]}")
    
    def train_random_forest(self, **kwargs):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        rf_model = RandomForestModel(task=self.task, **kwargs)
        rf_model.train(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(self.X_train)
        y_pred_test = rf_model.predict(self.X_test)
        
        # Evaluate
        if self.task == 'regression':
            train_score = r2_score(self.y_train, y_pred_train)
            test_score = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            results = {
                'train_r2': train_score,
                'test_r2': test_score,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'feature_importance': rf_model.get_feature_importance()
            }
        else:
            train_score = accuracy_score(self.y_train, y_pred_train)
            test_score = accuracy_score(self.y_test, y_pred_test)
            
            results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'feature_importance': rf_model.get_feature_importance()
            }
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = results
        
        print(f"Random Forest - Test Score: {test_score:.4f}")
        return rf_model, results
    
    def train_xgboost(self, tune_hyperparams=True, **kwargs):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        xgb_model = XGBoostModel(task=self.task, **kwargs)
        
        if tune_hyperparams:
            print("Tuning hyperparameters...")
            best_params = xgb_model.tune_hyperparameters(self.X_train, self.y_train)
            print(f"Best parameters: {best_params}")
        else:
            xgb_model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Predictions
        y_pred_train = xgb_model.predict(self.X_train)
        y_pred_test = xgb_model.predict(self.X_test)
        
        # Evaluate
        if self.task == 'regression':
            train_score = r2_score(self.y_train, y_pred_train)
            test_score = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            results = {
                'train_r2': train_score,
                'test_r2': test_score,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'feature_importance': xgb_model.get_feature_importance()
            }
        else:
            train_score = accuracy_score(self.y_train, y_pred_train)
            test_score = accuracy_score(self.y_test, y_pred_test)
            
            results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'feature_importance': xgb_model.get_feature_importance()
            }
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = results
        
        print(f"XGBoost - Test Score: {test_score:.4f}")
        return xgb_model, results
    
    def train_bayesian_nn(self, hidden_dims=[64, 32], epochs=100):
        """Train Bayesian Neural Network"""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping Bayesian Neural Network.")
            return None, None
        
        print("Training Bayesian Neural Network...")
        
        output_dim = 1 if self.task == 'regression' or len(np.unique(self.y_train)) == 2 else len(np.unique(self.y_train))
        
        bnn_model = BayesianNeuralNetwork(
            input_dim=self.X_train.shape[1],
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            task=self.task
        )
        
        # Split training data for validation
        split_idx = int(0.8 * len(self.X_train))
        X_train_split = self.X_train[:split_idx]
        y_train_split = self.y_train[:split_idx]
        X_val_split = self.X_train[split_idx:]
        y_val_split = self.y_train[split_idx:]
        
        history = bnn_model.train(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=epochs
        )
        
        # Predictions with uncertainty
        y_pred_mean, y_pred_std = bnn_model.predict_with_uncertainty(self.X_test)
        y_pred_mean = y_pred_mean.flatten()
        y_pred_std = y_pred_std.flatten()
        
        # Evaluate
        if self.task == 'regression':
            test_score = r2_score(self.y_test, y_pred_mean)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_mean))
            
            results = {
                'test_r2': test_score,
                'test_rmse': test_rmse,
                'prediction_uncertainty': np.mean(y_pred_std),
                'history': history.history
            }
        else:
            y_pred_binary = (y_pred_mean > 0.5).astype(int)
            test_score = accuracy_score(self.y_test, y_pred_binary)
            
            results = {
                'test_accuracy': test_score,
                'prediction_uncertainty': np.mean(y_pred_std),
                'history': history.history
            }
        
        self.models['bayesian_nn'] = bnn_model
        self.results['bayesian_nn'] = results
        
        print(f"Bayesian NN - Test Score: {test_score:.4f}, Uncertainty: {np.mean(y_pred_std):.4f}")
        return bnn_model, results
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("No models trained yet.")
            return
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if self.task == 'regression':
                test_score = results.get('test_r2', 0)
                test_rmse = results.get('test_rmse', float('inf'))
                comparison_data.append({
                    'Model': model_name,
                    'Test R²': test_score,
                    'Test RMSE': test_rmse
                })
            else:
                test_score = results.get('test_accuracy', 0)
                comparison_data.append({
                    'Model': model_name,
                    'Test Accuracy': test_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_results(self, output_dir):
        """Plot training results and comparisons"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Model comparison plot
        if len(self.results) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = list(self.results.keys())
            if self.task == 'regression':
                scores = [self.results[model].get('test_r2', 0) for model in models]
                ylabel = 'Test R²'
                title = 'Model Comparison - R² Scores'
            else:
                scores = [self.results[model].get('test_accuracy', 0) for model in models]
                ylabel = 'Test Accuracy'
                title = 'Model Comparison - Accuracy Scores'
            
            bars = ax.bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral'][:len(models)])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Feature importance plots
        for model_name, results in self.results.items():
            if 'feature_importance' in results and self.feature_names is not None:
                importance = results['feature_importance']
                
                # Sort features by importance
                sorted_idx = np.argsort(importance)[::-1][:20]  # Top 20 features
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(sorted_idx)), importance[sorted_idx])
                plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
                plt.xlabel('Feature Importance')
                plt.title(f'{model_name.title()} - Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(output_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Bayesian NN training history
        if 'bayesian_nn' in self.results and 'history' in self.results['bayesian_nn']:
            history = self.results['bayesian_nn']['history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            ax1.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training History - Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Metric plot
            metric_key = 'mae' if self.task == 'regression' else 'accuracy'
            if metric_key in history:
                ax2.plot(history[metric_key], label=f'Training {metric_key.upper()}')
                if f'val_{metric_key}' in history:
                    ax2.plot(history[f'val_{metric_key}'], label=f'Validation {metric_key.upper()}')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel(metric_key.upper())
                ax2.set_title(f'Training History - {metric_key.upper()}')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'bayesian_nn_training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_models(self, output_dir):
        """Save trained models"""
        output_dir = Path(output_dir)
        models_dir = output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name in ['random_forest', 'xgboost']:
                joblib.dump(model.model, models_dir / f'{model_name}_model.pkl')
            elif model_name == 'bayesian_nn' and model.model is not None:
                model.model.save(models_dir / 'bayesian_nn_model')
        
        # Save results
        with open(models_dir / 'training_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for model_name, results in self.results.items():
                results_copy = results.copy()
                for key, value in results_copy.items():
                    if isinstance(value, np.ndarray):
                        results_copy[key] = value.tolist()
                    elif key == 'history' and isinstance(value, dict):
                        # Handle TensorFlow history
                        history_copy = {}
                        for hist_key, hist_value in value.items():
                            if isinstance(hist_value, list):
                                history_copy[hist_key] = hist_value
                            else:
                                history_copy[hist_key] = float(hist_value) if np.isscalar(hist_value) else hist_value
                        results_copy[key] = history_copy
                results_serializable[model_name] = results_copy
            
            json.dump(results_serializable, f, indent=2)

def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train ML models for solar flare analysis')
    parser.add_argument('--data', required=True, help='Path to preprocessed data (.npz file)')
    parser.add_argument('--task', default='regression', choices=['regression', 'classification'],
                       help='Task type')
    parser.add_argument('--models', nargs='+', default=['random_forest', 'xgboost', 'bayesian_nn'],
                       choices=['random_forest', 'xgboost', 'bayesian_nn'],
                       help='Models to train')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(task=args.task)
    
    # Load data
    trainer.load_data(args.data)
    
    # Train specified models
    if 'random_forest' in args.models:
        trainer.train_random_forest()
    
    if 'xgboost' in args.models:
        trainer.train_xgboost()
    
    if 'bayesian_nn' in args.models:
        trainer.train_bayesian_nn()
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Plot results
    trainer.plot_results(args.output_dir)
    
    # Save models and results
    trainer.save_models(args.output_dir)
    
    print(f"\nTraining complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
