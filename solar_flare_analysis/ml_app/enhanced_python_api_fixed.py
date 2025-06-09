#!/usr/bin/env python3
"""
Enhanced Production Python API for Solar Flare Analysis
Connects the React frontend with the ML backend using Flask
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import logging
from datetime import datetime
import traceback
import io
import base64

# Add the src directory to the path to import our ML modules
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

try:
    from solar_flare_analysis.src.data_processing.data_loader import GOESDataLoader
    from solar_flare_analysis.src.ml_models.enhanced_flare_analysis import (
        EnhancedFlareDecompositionModel,
        NanoflareDetector,
        FlareEnergyAnalyzer
    )
    from solar_flare_analysis.src.visualization.plotting import FlareVisualization
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML modules not available: {e}. Using mock data.")
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'nc', 'h5', 'hdf5', 'fits', 'csv', 'txt'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ProductionFlareAnalyzer:
    """Production-ready solar flare analyzer"""
    
    def __init__(self):
        self.ml_model = None
        self.nanoflare_detector = None
        self.energy_analyzer = None
        self.data_loader = None
        self.visualization = None
        self.initialized = False
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            logger.info("Initializing ML models...")
            
            # Initialize models
            self.ml_model = EnhancedFlareDecompositionModel(
                sequence_length=512,
                n_features=2,
                max_flares=10
            )
            
            self.nanoflare_detector = NanoflareDetector(
                threshold_alpha=2.0,
                min_prominence=0.1
            )
            
            self.energy_analyzer = FlareEnergyAnalyzer()
            self.data_loader = GOESDataLoader()
            self.visualization = FlareVisualization()
            
            # Build the model
            self.ml_model.build_enhanced_model()
            
            # Try to load pre-trained weights if available
            model_path = project_root / 'models' / 'enhanced_flare_model.h5'
            if model_path.exists():
                self.ml_model.model.load_weights(str(model_path))
                logger.info("Loaded pre-trained model weights")
            else:
                logger.info("No pre-trained weights found, using untrained model")
            
            self.initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            logger.error(traceback.format_exc())
    
    def _generate_mock_analysis(self):
        """Generate mock analysis data when ML is unavailable"""
        logger.info("Generating mock analysis data")
        
        # Generate synthetic flare data
        num_flares = np.random.randint(30, 80)
        flares = []
        
        for i in range(num_flares):
            energy = np.power(10, np.random.uniform(26, 30))
            intensity = np.random.exponential(200) + 50
            alpha = np.random.normal(0, 2)
            
            flare = {
                'timestamp': f"2024-{1 + i//30:02d}-{1 + i%30:02d}T{i%24:02d}:{(i*15)%60:02d}:00Z",
                'intensity': float(intensity),
                'energy': float(energy),
                'alpha': float(alpha),
                'peak_time': float(np.random.random()),
                'rise_time': float(np.random.exponential(0.1)),
                'decay_time': float(np.random.exponential(0.3)),
                'background': float(np.random.normal(50, 10)),
                'confidence': float(np.random.random()),
                'flare_type': np.random.choice(['nano', 'micro', 'minor', 'major', 'X-class'], 
                                            p=[0.5, 0.3, 0.15, 0.04, 0.01])
            }
            flares.append(flare)
        
        # Identify nanoflares
        nanoflares = [f for f in flares if abs(f['alpha']) > 2.0 or f['flare_type'] == 'nano']
        
        # Calculate statistics
        energies = [f['energy'] for f in flares]
        nano_energies = [f['energy'] for f in nanoflares]
        
        return {
            'success': True,
            'separated_flares': flares,
            'nanoflares': nanoflares,
            'energy_analysis': {
                'total_energy': sum(energies),
                'average_energy': np.mean(energies),
                'median_energy': np.median(energies),
                'power_law_index': np.random.uniform(-2.5, -1.5),
                'nanoflare_energy_fraction': sum(nano_energies) / sum(energies) if nano_energies else 0
            },
            'statistics': {
                'total_flares': len(flares),
                'nanoflare_count': len(nanoflares),
                'nanoflare_percentage': len(nanoflares) / len(flares) * 100,
                'average_energy': np.mean(energies),
                'power_law_index': np.random.uniform(-2.5, -1.5)
            },
            'visualizations': {
                'energy_histogram': [{'energy': 10**(27+i*0.3), 'count': max(0, int(20*np.exp(-i/3)))} 
                                   for i in range(10)],
                'flare_timeline': sorted(flares, key=lambda x: x['timestamp'])[:20]
            },
            'metadata': {
                'file_processed': 'mock_data.csv',
                'processing_time': datetime.now().isoformat(),
                'data_points': 1000,
                'model_version': '2.0.0-mock'
            }
        }

# Global analyzer instance
analyzer = ProductionFlareAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_available': ML_AVAILABLE,
        'model_initialized': analyzer.initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Enhanced Flare Decomposition Model',
        'version': '2.0.0',
        'features': [
            'Multi-flare separation',
            'Nanoflare detection',
            'Energy estimation',
            'Power law analysis',
            'Attention mechanism',
            'Residual connections'
        ],
        'initialized': analyzer.initialized,
        'ml_available': ML_AVAILABLE
    })

@app.route('/api/montecarlo/background', methods=['POST'])
def monte_carlo_background():
    """Monte Carlo background noise simulation endpoint"""
    try:
        data = request.get_json()
        
        # Extract configuration
        config = {
            'realizations': data.get('realizations', 1000),
            'duration_hours': data.get('duration_hours', 24),
            'activity_level': data.get('activity_level', 'medium'),
            'background_noise_level': data.get('background_noise_level', 0.1),
            'confidence_level': data.get('confidence_level', 0.95)
        }
        
        logger.info(f"Running Monte Carlo background simulation with {config['realizations']} realizations")
        
        if ML_AVAILABLE:
            # Run actual Monte Carlo simulation using imported modules
            try:
                # Import Monte Carlo module
                sys.path.insert(0, str(src_dir))
                from solar_flare_analysis.src.ml_models.monte_carlo_background_simulation import MonteCarloBackgroundSimulator
                
                # Initialize simulator with proper paths
                models_dir = project_root / 'models'
                data_dir = project_root / 'data'
                
                simulator = MonteCarloBackgroundSimulator(
                    models_dir=models_dir,
                    data_dir=data_dir,
                    n_samples=config['realizations']
                )
                
                # Create scenario parameters based on config
                scenario_params = {
                    config['activity_level']: {
                        'noise_level': config['background_noise_level'],
                        'duration_hours': config['duration_hours'],
                        'add_flares': config['activity_level'] in ['high', 'mixed']
                    }
                }
                
                results = simulator.simulate_background_scenarios(scenario_params)
                
                return jsonify({
                    'success': True,
                    'source': 'ml_backend',
                    'results': results,
                    'config': config,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"ML Monte Carlo failed, using fallback: {e}")
        
        # Fallback: Generate mock results
        mock_results = generate_mock_monte_carlo_background(config)
        
        return jsonify({
            'success': True,
            'source': 'mock_backend',
            'results': mock_results,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'message': 'Using mock Monte Carlo results - ML modules not fully available'
        })
        
    except Exception as e:
        logger.error(f"Monte Carlo background endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Monte Carlo background simulation failed',
            'details': str(e)
        }), 500

@app.route('/api/montecarlo/cross-validation', methods=['POST'])
def monte_carlo_cross_validation():
    """Monte Carlo cross-validation endpoint"""
    try:
        data = request.get_json()
        
        config = {
            'cv_folds': data.get('cv_folds', 5),
            'realizations': data.get('realizations', 1000),
            'activity_level': data.get('activity_level', 'medium'),
            'confidence_level': data.get('confidence_level', 0.95)
        }
        
        logger.info(f"Running Monte Carlo cross-validation with {config['cv_folds']} folds")
        
        if ML_AVAILABLE:
            try:
                # Import Monte Carlo module
                sys.path.insert(0, str(src_dir))
                from solar_flare_analysis.src.ml_models.monte_carlo_background_simulation import MonteCarloBackgroundSimulator
                
                # Initialize simulator
                models_dir = project_root / 'models'
                data_dir = project_root / 'data'
                
                simulator = MonteCarloBackgroundSimulator(
                    models_dir=models_dir,
                    data_dir=data_dir,
                    n_samples=config['realizations']
                )
                
                results = simulator.run_cross_validation_simulation(
                    cv_folds=config['cv_folds']
                )
                
                return jsonify({
                    'success': True,
                    'source': 'ml_backend',
                    'results': results,
                    'config': config,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"ML Monte Carlo CV failed, using fallback: {e}")
        
        # Fallback: Generate mock results
        mock_results = generate_mock_monte_carlo_cv(config)
        
        return jsonify({
            'success': True,
            'source': 'mock_backend',
            'results': mock_results,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'message': 'Using mock Monte Carlo CV results - ML modules not fully available'
        })
        
    except Exception as e:
        logger.error(f"Monte Carlo CV endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Monte Carlo cross-validation failed',
            'details': str(e)
        }), 500

@app.route('/api/montecarlo/augmentation', methods=['POST'])
def monte_carlo_augmentation():
    """Monte Carlo data augmentation endpoint"""
    try:
        data = request.get_json()
        
        config = {
            'augmentation_factor': data.get('augmentation_factor', 3),
            'realizations': data.get('realizations', 1000),
            'background_noise_level': data.get('background_noise_level', 0.1),
            'activity_level': data.get('activity_level', 'medium')
        }
        
        logger.info(f"Running Monte Carlo data augmentation with factor {config['augmentation_factor']}")
        
        if ML_AVAILABLE:
            try:
                # Import Monte Carlo module
                sys.path.insert(0, str(src_dir))
                from solar_flare_analysis.src.ml_models.monte_carlo_background_simulation import MonteCarloBackgroundSimulator
                
                # Initialize simulator
                models_dir = project_root / 'models'
                data_dir = project_root / 'data'
                
                simulator = MonteCarloBackgroundSimulator(
                    models_dir=models_dir,
                    data_dir=data_dir,
                    n_samples=config['realizations']
                )
                
                # Create augmentation parameters
                augmentation_params = {
                    'noise_levels': [0.05, 0.1, 0.15, 0.2],
                    'scaling_factors': [0.8, 0.9, 1.1, 1.2]
                }
                
                results = simulator.run_data_augmentation_simulation(
                    augmentation_params=augmentation_params
                )
                
                return jsonify({
                    'success': True,
                    'source': 'ml_backend',
                    'results': results,
                    'config': config,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"ML Monte Carlo augmentation failed, using fallback: {e}")
        
        # Fallback: Generate mock results
        mock_results = generate_mock_monte_carlo_augmentation(config)
        
        return jsonify({
            'success': True,
            'source': 'mock_backend',
            'results': mock_results,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'message': 'Using mock Monte Carlo augmentation results - ML modules not fully available'
        })
        
    except Exception as e:
        logger.error(f"Monte Carlo augmentation endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Monte Carlo data augmentation failed',
            'details': str(e)
        }), 500

@app.route('/api/bayesian/analysis', methods=['POST'])
def bayesian_analysis():
    """Bayesian uncertainty analysis endpoint"""
    try:
        data = request.get_json()
        
        config = {
            'data': data.get('data', []),
            'n_chains': data.get('n_chains', 4),
            'n_iterations': data.get('n_iterations', 2000),
            'target_acceptance': data.get('target_acceptance', 0.8),
            'regularization_strength': data.get('regularization_strength', 0.01)
        }
        
        logger.info(f"Running Bayesian analysis with {config['n_chains']} chains, {config['n_iterations']} iterations")
        
        if ML_AVAILABLE:
            try:
                # Import Bayesian analysis module
                sys.path.insert(0, str(src_dir))
                from solar_flare_analysis.src.ml_models.bayesian_flare_analysis import BayesianFlareAnalyzer, BayesianFlareEnergyEstimator
                
                # Initialize Bayesian analyzer
                analyzer = BayesianFlareAnalyzer(
                    sequence_length=256,
                    n_features=2,
                    max_flares=5,
                    n_monte_carlo_samples=config['n_iterations']
                )
                
                # Check if we have input data
                input_data = config.get('data', [])
                if input_data and len(input_data) > 0:
                    # Convert input data to numpy array
                    data_array = np.array(input_data)
                    if data_array.ndim == 1:
                        data_array = data_array.reshape(-1, 1)
                    if data_array.shape[1] == 1:
                        # Duplicate column for 2-channel analysis
                        data_array = np.hstack([data_array, data_array * 0.1])
                    
                    # Run Bayesian inference
                    logger.info("Running real Bayesian inference...")
                    
                    # Build model
                    model = analyzer.build_bayesian_model()
                    
                    # Run Monte Carlo inference
                    uncertainty_results = analyzer.monte_carlo_inference(
                        data_array,
                        n_samples=config['n_iterations'],
                        chains=config['n_chains']
                    )
                    
                    # Energy estimation
                    energy_estimator = BayesianFlareEnergyEstimator(
                        n_monte_carlo_samples=config['n_iterations']
                    )
                    
                    energy_model = energy_estimator.build_energy_model()
                    energy_results = energy_estimator.estimate_energy_distribution(data_array)
                    
                    # Combine results
                    results = {
                        'uncertainty_analysis': uncertainty_results,
                        'energy_estimation': energy_results,
                        'model_diagnostics': {
                            'convergence_metrics': analyzer.compute_convergence_diagnostics(),
                            'posterior_summary': analyzer.summarize_posterior(),
                            'mcmc_diagnostics': analyzer.get_mcmc_diagnostics()
                        }
                    }
                    
                    return jsonify({
                        'success': True,
                        'source': 'ml_backend',
                        'results': results,
                        'config': config,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                else:
                    # No input data, run on synthetic data
                    logger.info("No input data provided, running Bayesian analysis on synthetic data")
                    synthetic_data = analyzer.generate_synthetic_flare_data(
                        n_samples=1000,
                        n_flares=3
                    )
                    
                    # Build and run model
                    model = analyzer.build_bayesian_model()
                    uncertainty_results = analyzer.monte_carlo_inference(
                        synthetic_data,
                        n_samples=config['n_iterations'],
                        chains=config['n_chains']
                    )
                    
                    results = {
                        'uncertainty_analysis': uncertainty_results,
                        'synthetic_data_used': True,
                        'data_characteristics': {
                            'n_samples': len(synthetic_data),
                            'n_features': synthetic_data.shape[1],
                            'data_range': [float(np.min(synthetic_data)), float(np.max(synthetic_data))]
                        }
                    }
                    
                    return jsonify({
                        'success': True,
                        'source': 'ml_backend',
                        'results': results,
                        'config': config,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"ML Bayesian analysis failed, using fallback: {e}")
                logger.warning(traceback.format_exc())
        
        # Generate mock Bayesian results
        mock_results = generate_mock_bayesian_results(config)
        
        return jsonify({
            'success': True,
            'source': 'mock_backend',
            'results': mock_results,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'message': 'Using mock Bayesian analysis results - full implementation pending'
        })
        
    except Exception as e:
        logger.error(f"Bayesian analysis endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Bayesian analysis failed',
            'details': str(e)
        }), 500

# Mock data generation functions
def generate_mock_monte_carlo_background(config):
    """Generate mock Monte Carlo background simulation results"""
    n_realizations = config['realizations']
    
    results = []
    for i in range(n_realizations):
        results.append({
            'realization_id': i + 1,
            'background_level': 0.05 + np.random.random() * 0.15,
            'flare_count': np.random.poisson(5),
            'total_energy': np.random.exponential(1e28),
            'confidence_interval': [
                np.random.uniform(0.02, 0.08),
                np.random.uniform(0.12, 0.18)
            ]
        })
    
    # Aggregate statistics
    background_levels = [r['background_level'] for r in results]
    flare_counts = [r['flare_count'] for r in results]
    total_energies = [r['total_energy'] for r in results]
    
    return {
        'realizations': results,
        'statistics': {
            'mean_background': np.mean(background_levels),
            'std_background': np.std(background_levels),
            'mean_flare_count': np.mean(flare_counts),
            'mean_total_energy': np.mean(total_energies),
            'confidence_level': config['confidence_level']
        }
    }

def generate_mock_monte_carlo_cv(config):
    """Generate mock Monte Carlo cross-validation results"""
    n_folds = config['cv_folds']
    
    fold_results = []
    for fold in range(n_folds):
        fold_results.append({
            'fold': fold + 1,
            'accuracy': 0.8 + np.random.random() * 0.15,
            'precision': 0.75 + np.random.random() * 0.2,
            'recall': 0.7 + np.random.random() * 0.25,
            'f1_score': 0.72 + np.random.random() * 0.2
        })
    
    # Aggregate metrics
    accuracies = [r['accuracy'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]
    
    return {
        'fold_results': fold_results,
        'aggregate_metrics': {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }
    }

def generate_mock_monte_carlo_augmentation(config):
    """Generate mock Monte Carlo data augmentation results"""
    augmentation_factor = config['augmentation_factor']
    
    original_accuracy = 0.82
    augmented_accuracies = []
    
    for i in range(augmentation_factor):
        # Simulate improved accuracy with augmentation
        accuracy = original_accuracy + np.random.normal(0.05, 0.02)
        augmented_accuracies.append(max(0, min(1, accuracy)))
    
    return {
        'original_accuracy': original_accuracy,
        'augmented_accuracies': augmented_accuracies,
        'improvement': {
            'mean_improvement': np.mean(augmented_accuracies) - original_accuracy,
            'best_improvement': max(augmented_accuracies) - original_accuracy,
            'consistency': np.std(augmented_accuracies)
        },
        'augmentation_factor': augmentation_factor
    }

def generate_mock_bayesian_results(config):
    """Generate mock Bayesian analysis results"""
    n_iterations = config['n_iterations']
    n_chains = config['n_chains']
    
    # Generate mock posterior samples
    posterior_samples = []
    for chain in range(n_chains):
        chain_samples = []
        for iteration in range(n_iterations // n_chains):
            sample = {
                'parameter_1': np.random.normal(0, 1),
                'parameter_2': np.random.normal(0, 1),
                'log_likelihood': np.random.normal(-100, 10)
            }
            chain_samples.append(sample)
        posterior_samples.append(chain_samples)
    
    # Mock convergence diagnostics
    return {
        'posterior_samples': posterior_samples,
        'convergence_diagnostics': {
            'r_hat': 1.01,
            'effective_sample_size': n_iterations * 0.8,
            'divergences': 0
        },
        'parameter_estimates': {
            'parameter_1': {
                'mean': 0.02,
                'std': 0.98,
                'credible_interval': [-1.96, 1.96]
            },
            'parameter_2': {
                'mean': -0.01,
                'std': 1.02,
                'credible_interval': [-2.0, 2.0]
            }
        }
    }

if __name__ == '__main__':
    logger.info("Starting Enhanced Python API server...")
    logger.info(f"ML modules available: {ML_AVAILABLE}")
    logger.info(f"Model initialized: {analyzer.initialized}")
    
    # Start the Flask development server
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
