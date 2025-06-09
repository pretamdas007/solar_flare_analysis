#!/usr/bin/env python3
"""
Solar Flare Analysis Backend Server
Provides REST API endpoints for the React frontend to interact with ML models
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from pathlib import Path

# Flask imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
from io import BytesIO

# ML and data processing imports
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
from scipy.stats import chi2
import netCDF4 as nc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
class Config:
    MODEL_DIR = Path(__file__).parent / "models"
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = Path(__file__).parent / "output"
    MAX_DATA_POINTS = 10000  # Limit for performance
    CACHE_DURATION = 3600  # 1 hour cache

# Global variables for caching
_models_cache = {}
_data_cache = {}
_predictions_cache = {}

class ModelManager:
    """Manages loading and inference of ML models"""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.load_models()
    
    def load_models(self):
        """Load all available trained models"""
        try:
            model_files = {
                'binary_classifier': 'binary_flare_classifier.h5',
                'multiclass_classifier': 'multiclass_flare_classifier.h5',
                'energy_estimator': 'energy_regression_model.h5',
                'cnn_detector': 'cnn_flare_detector.h5',
                'minimal_detector': 'minimal_flare_model.h5'
            }
            
            for model_name, filename in model_files.items():
                model_path = Config.MODEL_DIR / filename
                if model_path.exists():
                    try:
                        model = keras.models.load_model(str(model_path))
                        self.models[model_name] = model
                        
                        # Get model info
                        self.model_info[model_name] = {
                            'input_shape': model.input_shape,
                            'output_shape': model.output_shape,
                            'parameters': model.count_params(),
                            'layers': len(model.layers)
                        }
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {e}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            logger.info(f"Loaded {len(self.models)} models successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict_flare_detection(self, data: np.ndarray) -> Dict:
        """Run flare detection using multiple models"""
        results = {}
        
        try:
            # Ensure data is properly shaped
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # Binary classification
            if 'binary_classifier' in self.models:
                binary_pred = self.models['binary_classifier'].predict(data, verbose=0)
                results['binary_prediction'] = {
                    'probability': float(binary_pred[0, 0]),
                    'classification': 'flare' if binary_pred[0, 0] > 0.5 else 'no_flare',
                    'confidence': float(abs(binary_pred[0, 0] - 0.5) * 2)
                }
            
            # Multiclass classification
            if 'multiclass_classifier' in self.models:
                multi_pred = self.models['multiclass_classifier'].predict(data, verbose=0)
                class_names = ['no_flare', 'C_class', 'M_class', 'X_class']
                class_idx = np.argmax(multi_pred[0])
                results['multiclass_prediction'] = {
                    'predicted_class': class_names[class_idx],
                    'probabilities': {
                        class_names[i]: float(multi_pred[0, i]) 
                        for i in range(len(class_names))
                    },
                    'confidence': float(np.max(multi_pred[0]))
                }
            
            # Energy estimation
            if 'energy_estimator' in self.models:
                energy_pred = self.models['energy_estimator'].predict(data, verbose=0)
                results['energy_estimation'] = {
                    'estimated_energy': float(energy_pred[0, 0]),
                    'energy_log10': float(np.log10(max(energy_pred[0, 0], 1e-10)))
                }
            
            # CNN detection (requires 3D input)
            if 'cnn_detector' in self.models:
                # Reshape for CNN (add channel dimension)
                cnn_data = data.reshape(data.shape[0], data.shape[1], 1)
                cnn_pred = self.models['cnn_detector'].predict(cnn_data, verbose=0)
                results['cnn_detection'] = {
                    'detection_score': float(cnn_pred[0, 0]),
                    'detected': bool(cnn_pred[0, 0] > 0.5)
                }
            
            # Minimal detector
            if 'minimal_detector' in self.models:
                minimal_pred = self.models['minimal_detector'].predict(data, verbose=0)
                results['minimal_detection'] = {
                    'detection_score': float(minimal_pred[0, 0]),
                    'detected': bool(minimal_pred[0, 0] > 0.5)
                }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            results['error'] = str(e)
        
        return results

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self):
        self.current_data = None
        self.data_info = {}
    
    def load_goes_data(self, filename: str) -> Dict:
        """Load GOES XRS data from netCDF file"""
        try:
            file_path = Config.DATA_DIR / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {filename}")
            
            with nc.Dataset(str(file_path), 'r') as dataset:
                # Extract key variables
                times = dataset.variables['time'][:]
                xrs_a = dataset.variables['xrsa_flux'][:]  # 0.05-0.4 nm
                xrs_b = dataset.variables['xrsb_flux'][:]  # 0.1-0.8 nm
                
                # Convert time to datetime
                time_units = dataset.variables['time'].units
                base_time = pd.to_datetime('2000-01-01 12:00:00')  # GOES epoch
                timestamps = pd.to_datetime(times, unit='s', origin=base_time)
                
                # Create DataFrame
                data = pd.DataFrame({
                    'timestamp': timestamps,
                    'xrs_a': xrs_a,
                    'xrs_b': xrs_b,
                    'ratio': xrs_a / (xrs_b + 1e-12)  # Avoid division by zero
                })
                
                # Remove invalid data
                data = data.dropna()
                data = data[data['xrs_a'] > 0]
                data = data[data['xrs_b'] > 0]
                
                # Limit data points for performance
                if len(data) > Config.MAX_DATA_POINTS:
                    step = len(data) // Config.MAX_DATA_POINTS
                    data = data.iloc[::step].reset_index(drop=True)
                
                self.current_data = data
                self.data_info = {
                    'filename': filename,
                    'start_time': str(data['timestamp'].min()),
                    'end_time': str(data['timestamp'].max()),
                    'duration_hours': (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600,
                    'data_points': len(data),
                    'xrs_a_range': [float(data['xrs_a'].min()), float(data['xrs_a'].max())],
                    'xrs_b_range': [float(data['xrs_b'].min()), float(data['xrs_b'].max())]
                }
                
                logger.info(f"Loaded GOES data: {len(data)} points from {filename}")
                return self.data_info
                
        except Exception as e:
            logger.error(f"Error loading GOES data: {e}")
            raise
    
    def detect_flares(self, threshold_factor: float = 3.0) -> List[Dict]:
        """Detect flare events in the loaded data"""
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        flares = []
        data = self.current_data.copy()
        
        try:
            # Use XRS-A for flare detection (shorter wavelength, more sensitive)
            flux = data['xrs_a'].values
            times = data['timestamp'].values
            
            # Calculate background (rolling minimum)
            window_size = min(100, len(flux) // 10)
            background = pd.Series(flux).rolling(window=window_size, center=True).min()
            background = background.fillna(method='bfill').fillna(method='ffill')
            
            # Find peaks above threshold
            threshold = background * threshold_factor
            peaks, properties = signal.find_peaks(
                flux, 
                height=threshold.values,
                distance=10,  # Minimum distance between peaks
                width=5       # Minimum width
            )
            
            for i, peak_idx in enumerate(peaks):
                peak_time = times[peak_idx]
                peak_flux = flux[peak_idx]
                peak_background = background.iloc[peak_idx]
                
                # Determine flare class based on peak flux
                if peak_flux >= 1e-4:
                    flare_class = 'X'
                elif peak_flux >= 1e-5:
                    flare_class = 'M'
                elif peak_flux >= 1e-6:
                    flare_class = 'C'
                else:
                    flare_class = 'B'
                
                # Estimate flare duration
                start_idx = max(0, peak_idx - 20)
                end_idx = min(len(flux), peak_idx + 20)
                flare_info = {
                    'id': i + 1,
                    'peak_time': pd.to_datetime(peak_time).isoformat(),
                    'peak_flux': float(peak_flux),
                    'background_flux': float(peak_background),
                    'enhancement_factor': float(peak_flux / peak_background),
                    'class': flare_class,
                    'start_time': pd.to_datetime(times[start_idx]).isoformat(),
                    'end_time': pd.to_datetime(times[end_idx]).isoformat(),
                    'duration_minutes': float((pd.to_datetime(times[end_idx]) - pd.to_datetime(times[start_idx])).total_seconds() / 60),
                    'index': int(peak_idx)
                }
                
                flares.append(flare_info)
            
            logger.info(f"Detected {len(flares)} flare events")
            return flares
        except Exception as e:
            logger.error(f"Error detecting flares: {e}")
            raise
    
    def prepare_ml_input(self, window_size: int = 100) -> np.ndarray:
        """Prepare data for ML model input"""
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        data = self.current_data.copy()
        
        # Normalize the data (log scale)
        xrs_a_log = np.log10(data['xrs_a'].values + 1e-12)
        xrs_b_log = np.log10(data['xrs_b'].values + 1e-12)
        
        # Create sliding windows
        if len(data) < window_size:
            # Pad if necessary
            pad_size = window_size - len(data)
            xrs_a_log = np.pad(xrs_a_log, (pad_size, 0), mode='edge')
            xrs_b_log = np.pad(xrs_b_log, (pad_size, 0), mode='edge')
        
        # Take the last window_size points and create features that match model input (128 features)
        features_a = xrs_a_log[-window_size:]
        features_b = xrs_b_log[-window_size:]
        
        # Combine features to get exactly 128 dimensions
        if window_size >= 64:
            # Use last 64 points from each channel
            combined_features = np.concatenate([features_a[-64:], features_b[-64:]])
        else:
            # Pad to get 128 features total
            combined_features = np.concatenate([features_a, features_b])
            if len(combined_features) < 128:
                pad_size = 128 - len(combined_features)
                combined_features = np.pad(combined_features, (0, pad_size), mode='constant', constant_values=combined_features[-1])
            elif len(combined_features) > 128:
                combined_features = combined_features[:128]
        
        return combined_features.reshape(1, -1)

class PlotGenerator:
    """Generates plots and visualizations"""
    
    @staticmethod
    def create_time_series_plot(data: pd.DataFrame, flares: List[Dict] = None) -> str:
        """Create time series plot of XRS data with flare markers"""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot XRS-A
        ax1.plot(data['timestamp'], data['xrs_a'], 'r-', linewidth=1, label='XRS-A (0.05-0.4 nm)')
        ax1.set_ylabel('Flux (W/m²)', color='white')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('GOES XRS Solar Flare Data Analysis', color='white', fontsize=16)
        
        # Plot XRS-B
        ax2.plot(data['timestamp'], data['xrs_b'], 'b-', linewidth=1, label='XRS-B (0.1-0.8 nm)')
        ax2.set_ylabel('Flux (W/m²)', color='white')
        ax2.set_xlabel('Time (UTC)', color='white')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add flare markers if provided
        if flares:
            for flare in flares:
                peak_time = pd.to_datetime(flare['peak_time'])
                peak_flux = flare['peak_flux']
                
                # Color by flare class
                colors = {'X': 'red', 'M': 'orange', 'C': 'yellow', 'B': 'green'}
                color = colors.get(flare['class'], 'white')
                
                ax1.axvline(peak_time, color=color, linestyle='--', alpha=0.7)
                ax1.scatter(peak_time, peak_flux, color=color, s=50, marker='*', zorder=5)
                
                # Add class label
                ax1.annotate(f"{flare['class']}{flare['id']}", 
                           (peak_time, peak_flux), 
                           xytext=(5, 5), textcoords='offset points',
                           color=color, fontsize=8)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='black', edgecolor='white', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    @staticmethod
    def create_flare_distribution_plot(flares: List[Dict]) -> str:
        """Create flare class distribution plot"""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Flare class distribution
        if flares:
            classes = [flare['class'] for flare in flares]
            class_counts = pd.Series(classes).value_counts()
            
            colors = {'X': 'red', 'M': 'orange', 'C': 'yellow', 'B': 'green'}
            bar_colors = [colors.get(cls, 'white') for cls in class_counts.index]
            
            ax1.bar(class_counts.index, class_counts.values, color=bar_colors, alpha=0.7)
            ax1.set_xlabel('Flare Class', color='white')
            ax1.set_ylabel('Count', color='white')
            ax1.set_title('Flare Class Distribution', color='white')
            ax1.grid(True, alpha=0.3)
            
            # Enhancement factor distribution
            enhancement_factors = [flare['enhancement_factor'] for flare in flares]
            ax2.hist(enhancement_factors, bins=20, color='cyan', alpha=0.7, edgecolor='white')
            ax2.set_xlabel('Enhancement Factor', color='white')
            ax2.set_ylabel('Count', color='white')
            ax2.set_title('Flare Enhancement Factor Distribution', color='white')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No flares detected', ha='center', va='center', 
                    transform=ax1.transAxes, color='white')
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax2.transAxes, color='white')
        
        plt.tight_layout()
        
        # Convert to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='black', edgecolor='white', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    @staticmethod
    def create_ml_predictions_plot(predictions: Dict, flares: List[Dict]) -> str:
        """Create ML predictions visualization"""
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Binary classification confidence
        if 'binary_prediction' in predictions:
            binary_data = predictions['binary_prediction']
            ax1.bar(['No Flare', 'Flare'], 
                   [1 - binary_data['probability'], binary_data['probability']], 
                   color=['green', 'red'], alpha=0.7)
            ax1.set_title('Binary Classification', color='white')
            ax1.set_ylabel('Probability', color='white')
            ax1.grid(True, alpha=0.3)
        
        # Multiclass probabilities
        if 'multiclass_prediction' in predictions:
            multi_data = predictions['multiclass_prediction']
            classes = list(multi_data['probabilities'].keys())
            probs = list(multi_data['probabilities'].values())
            colors = ['green', 'yellow', 'orange', 'red']
            
            ax2.bar(classes, probs, color=colors, alpha=0.7)
            ax2.set_title('Multiclass Classification', color='white')
            ax2.set_ylabel('Probability', color='white')
            ax2.grid(True, alpha=0.3)
        
        # Flare timeline
        if flares:
            times = [pd.to_datetime(flare['peak_time']) for flare in flares]
            fluxes = [flare['peak_flux'] for flare in flares]
            classes = [flare['class'] for flare in flares]
            
            color_map = {'X': 'red', 'M': 'orange', 'C': 'yellow', 'B': 'green'}
            colors = [color_map.get(cls, 'white') for cls in classes]
            
            ax3.scatter(times, fluxes, c=colors, s=50, alpha=0.7)
            ax3.set_yscale('log')
            ax3.set_title('Detected Flares Timeline', color='white')
            ax3.set_ylabel('Peak Flux (W/m²)', color='white')
            ax3.grid(True, alpha=0.3)
        
        # Model performance summary
        performance_text = "ML Model Summary:\n"
        if 'binary_prediction' in predictions:
            performance_text += f"Binary Confidence: {predictions['binary_prediction']['confidence']:.2f}\n"
        if 'multiclass_prediction' in predictions:
            performance_text += f"Multiclass Confidence: {predictions['multiclass_prediction']['confidence']:.2f}\n"
        if 'energy_estimation' in predictions:
            performance_text += f"Estimated Energy: {predictions['energy_estimation']['estimated_energy']:.2e}\n"
        
        ax4.text(0.1, 0.9, performance_text, transform=ax4.transAxes, 
                color='white', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Performance Summary', color='white')
        
        plt.tight_layout()
        
        # Convert to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='black', edgecolor='white', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data

# Initialize global instances
model_manager = ModelManager()
data_processor = DataProcessor()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(model_manager.models),
        'available_models': list(model_manager.models.keys()),
        'ml_models_loaded': len(model_manager.models) > 0,
        'data_loaded': data_processor.current_data is not None
    })

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """Get information about loaded models"""
    return jsonify({
        'models': model_manager.model_info,
        'total_models': len(model_manager.models),
        'status': 'success'
    })

@app.route('/api/data/load', methods=['POST'])
def load_data():
    """Load GOES data file"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        data_info = data_processor.load_goes_data(filename)
        
        return jsonify({
            'status': 'success',
            'data_info': data_info,
            'message': f'Successfully loaded {filename}'
        })
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/available', methods=['GET'])
def get_available_data():
    """Get list of available data files"""
    try:
        data_files = []
        if Config.DATA_DIR.exists():
            for file_path in Config.DATA_DIR.glob('*.nc'):
                file_info = {
                    'filename': file_path.name,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                data_files.append(file_info)
        
        return jsonify({
            'status': 'success',
            'files': data_files,
            'count': len(data_files)
        })
        
    except Exception as e:
        logger.error(f"Error getting available data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/detect-flares', methods=['POST'])
def detect_flares():
    """Detect flares in loaded data"""
    try:
        data = request.get_json()
        threshold_factor = data.get('threshold_factor', 3.0)
        
        flares = data_processor.detect_flares(threshold_factor)
        
        return jsonify({
            'status': 'success',
            'flares': flares,
            'count': len(flares),
            'threshold_factor': threshold_factor
        })
        
    except Exception as e:
        logger.error(f"Error detecting flares: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/ml-predict', methods=['POST'])
def ml_predict():
    """Run ML predictions on current data"""
    try:
        data = request.get_json()
        window_size = data.get('window_size', 100)
        
        # Prepare input data
        ml_input = data_processor.prepare_ml_input(window_size)
        
        # Run predictions
        predictions = model_manager.predict_flare_detection(ml_input)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'input_shape': ml_input.shape,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plots/time-series', methods=['POST'])
def generate_time_series_plot():
    """Generate time series plot"""
    try:
        data = request.get_json()
        include_flares = data.get('include_flares', True)
        
        flares = None
        if include_flares:
            threshold_factor = data.get('threshold_factor', 3.0)
            flares = data_processor.detect_flares(threshold_factor)
        
        plot_data = PlotGenerator.create_time_series_plot(
            data_processor.current_data, flares
        )
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'format': 'base64_png'
        })
        
    except Exception as e:
        logger.error(f"Error generating time series plot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plots/flare-distribution', methods=['POST'])
def generate_flare_distribution_plot():
    """Generate flare distribution plot"""
    try:
        data = request.get_json()
        threshold_factor = data.get('threshold_factor', 3.0)
        
        flares = data_processor.detect_flares(threshold_factor)
        plot_data = PlotGenerator.create_flare_distribution_plot(flares)
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'format': 'base64_png',
            'flare_count': len(flares)
        })
        
    except Exception as e:
        logger.error(f"Error generating flare distribution plot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plots/ml-predictions', methods=['POST'])
def generate_ml_predictions_plot():
    """Generate ML predictions visualization"""
    try:
        data = request.get_json()
        window_size = data.get('window_size', 100)
        threshold_factor = data.get('threshold_factor', 3.0)
        
        # Get ML predictions
        ml_input = data_processor.prepare_ml_input(window_size)
        predictions = model_manager.predict_flare_detection(ml_input)
        
        # Get flares for context
        flares = data_processor.detect_flares(threshold_factor)
        
        plot_data = PlotGenerator.create_ml_predictions_plot(predictions, flares)
        
        return jsonify({
            'status': 'success',
            'plot': plot_data,
            'format': 'base64_png',
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error generating ML predictions plot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/summary', methods=['GET'])
def get_analysis_summary():
    """Get comprehensive analysis summary"""
    try:
        if data_processor.current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # Detect flares
        flares = data_processor.detect_flares()
        
        # Run ML predictions
        ml_input = data_processor.prepare_ml_input()
        predictions = model_manager.predict_flare_detection(ml_input)
        
        # Calculate statistics
        data = data_processor.current_data
        stats = {
            'data_period': {
                'start': str(data['timestamp'].min()),
                'end': str(data['timestamp'].max()),
                'duration_hours': (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600
            },
            'flux_statistics': {
                'xrs_a': {
                    'min': float(data['xrs_a'].min()),
                    'max': float(data['xrs_a'].max()),
                    'mean': float(data['xrs_a'].mean()),
                    'std': float(data['xrs_a'].std())
                },
                'xrs_b': {
                    'min': float(data['xrs_b'].min()),
                    'max': float(data['xrs_b'].max()),
                    'mean': float(data['xrs_b'].mean()),
                    'std': float(data['xrs_b'].std())
                }
            },
            'flare_statistics': {
                'total_flares': len(flares),
                'class_distribution': pd.Series([f['class'] for f in flares]).value_counts().to_dict(),
                'average_enhancement': float(np.mean([f['enhancement_factor'] for f in flares])) if flares else 0
            }
        }
        
        return jsonify({
            'status': 'success',
            'summary': stats,
            'flares': flares[:10],  # Return first 10 flares
            'ml_predictions': predictions,
            'data_info': data_processor.data_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating analysis summary: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure directories exist
    Config.MODEL_DIR.mkdir(exist_ok=True)
    Config.DATA_DIR.mkdir(exist_ok=True)
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    logger.info("Starting Solar Flare Analysis Backend Server")
    logger.info(f"Models available: {list(model_manager.models.keys())}")
    
    # Run the server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
