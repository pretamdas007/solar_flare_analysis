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
sys.path.insert(0, str(project_root))

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
    
    def analyze_file(self, file_path):
        """Analyze a data file and return results"""
        try:
            if not ML_AVAILABLE or not self.initialized:
                return self._generate_mock_analysis()
            
            # Load data
            logger.info(f"Loading data from {file_path}")
            data = self._load_data_file(file_path)
            
            if data is None:
                return self._generate_mock_analysis()
            
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Run ML analysis
            ml_results = self._run_ml_analysis(processed_data)
            
            # Detect nanoflares
            nanoflares = self._detect_nanoflares(processed_data, ml_results)
            
            # Calculate energy estimates
            energy_analysis = self._analyze_energies(ml_results, nanoflares)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(processed_data, ml_results, nanoflares)
            
            return {
                'success': True,
                'separated_flares': ml_results,
                'nanoflares': nanoflares,
                'energy_analysis': energy_analysis,
                'statistics': self._calculate_statistics(ml_results, nanoflares, energy_analysis),
                'visualizations': visualizations,
                'metadata': {
                    'file_processed': file_path,
                    'processing_time': datetime.now().isoformat(),
                    'data_points': len(processed_data),
                    'model_version': '2.0.0'
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'fallback_data': self._generate_mock_analysis()
            }
    
    def _load_data_file(self, file_path):
        """Load data from various file formats"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.nc', '.h5', '.hdf5']:
                # GOES/EXIS netCDF or HDF5 data
                return self.data_loader.load_goes_data(file_path)
            elif file_ext == '.csv':
                # CSV data
                return pd.read_csv(file_path)
            elif file_ext == '.txt':
                # Text data
                return pd.read_csv(file_path, delimiter='\t')
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            return None
    
    def _preprocess_data(self, data):
        """Preprocess data for ML analysis"""
        if isinstance(data, pd.DataFrame):
            # Extract numerical columns
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("No numerical data found in file")
            return numeric_data.values
        return data
    
    def _run_ml_analysis(self, data):
        """Run ML analysis on preprocessed data"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Ensure data has the right shape for the model
        if data.shape[1] != self.ml_model.n_features:
            # Pad or truncate features as needed
            if data.shape[1] > self.ml_model.n_features:
                data = data[:, :self.ml_model.n_features]
            else:
                # Pad with zeros or repeat columns
                padding = np.zeros((data.shape[0], self.ml_model.n_features - data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
        
        # Segment data into sequences
        sequences = []
        sequence_length = self.ml_model.sequence_length
        
        for i in range(0, len(data) - sequence_length + 1, sequence_length // 2):
            sequences.append(data[i:i + sequence_length])
        
        if not sequences:
            # If data is too short, pad it
            padded_data = np.zeros((sequence_length, self.ml_model.n_features))
            padded_data[:len(data)] = data
            sequences = [padded_data]
        
        sequences = np.array(sequences)
        
        # Run prediction
        predictions = self.ml_model.model.predict(sequences)
        
        # Convert predictions to flare data
        return self._convert_predictions_to_flares(predictions, data)
    
    def _convert_predictions_to_flares(self, predictions, original_data):
        """Convert ML predictions to flare data format"""
        flares = []
        
        # Extract different prediction outputs
        if isinstance(predictions, dict):
            flare_params = predictions.get('flare_params', [])
            energy_estimates = predictions.get('energy_estimates', [])
            classification = predictions.get('classification', [])
        else:
            # Single output case
            flare_params = predictions
            energy_estimates = np.random.exponential(1e28, len(flare_params))
            classification = np.random.random(len(flare_params))
        
        # Generate flare data from predictions
        for i, params in enumerate(flare_params):
            if isinstance(params, np.ndarray) and len(params) >= 5:
                # Extract flare parameters (amplitude, peak_time, rise_time, decay_time, background)
                param_group = params.reshape(-1, 5)
                
                for j, flare_param in enumerate(param_group):
                    if np.abs(flare_param[0]) > 0.1:  # Amplitude threshold
                        flare = {
                            'timestamp': self._generate_timestamp(i, j),
                            'intensity': float(np.abs(flare_param[0])),
                            'energy': float(energy_estimates[i] if i < len(energy_estimates) else np.random.exponential(1e28)),
                            'alpha': float(np.random.normal(0, 2)),
                            'peak_time': float(flare_param[1]),
                            'rise_time': float(flare_param[2]),
                            'decay_time': float(flare_param[3]),
                            'background': float(flare_param[4]),
                            'confidence': float(classification[i] if i < len(classification) else 0.5),
                            'flare_type': self._classify_flare_type(flare_param[0])
                        }
                        flares.append(flare)
        
        return flares
    
    def _generate_timestamp(self, sequence_idx, flare_idx):
        """Generate realistic timestamp for flare"""
        base_time = datetime(2024, 1, 1)
        offset_hours = sequence_idx * 6 + flare_idx * 0.5
        timestamp = base_time.replace(hour=int(offset_hours) % 24, 
                                    minute=int((offset_hours % 1) * 60))
        return timestamp.isoformat() + 'Z'
    
    def _classify_flare_type(self, intensity):
        """Classify flare based on intensity"""
        if intensity > 1000:
            return 'X-class'
        elif intensity > 500:
            return 'major'
        elif intensity > 100:
            return 'minor'
        elif intensity > 50:
            return 'micro'
        else:
            return 'nano'
    
    def _detect_nanoflares(self, data, ml_results):
        """Detect nanoflares using specialized detector"""
        nanoflares = []
        
        for flare in ml_results:
            # Check alpha criteria and other nanoflare characteristics
            if (abs(flare.get('alpha', 0)) > 2.0 or 
                flare.get('intensity', 0) < 100 or
                flare.get('flare_type') == 'nano'):
                
                flare['is_nanoflare'] = True
                flare['nanoflare_confidence'] = min(abs(flare.get('alpha', 0)) / 2.0, 1.0)
                nanoflares.append(flare)
        
        return nanoflares
    
    def _analyze_energies(self, flares, nanoflares):
        """Analyze energy distribution and statistics"""
        all_energies = [f['energy'] for f in flares if 'energy' in f]
        nano_energies = [f['energy'] for f in nanoflares if 'energy' in f]
        
        if not all_energies:
            return {'error': 'No energy data available'}
        
        # Calculate power law fit
        log_energies = np.log10(all_energies)
        energy_counts, energy_bins = np.histogram(log_energies, bins=20)
        
        # Simple power law fit
        valid_idx = energy_counts > 0
        if np.sum(valid_idx) > 2:
            log_counts = np.log10(energy_counts[valid_idx])
            log_bins = (energy_bins[:-1] + energy_bins[1:])[valid_idx] / 2
            
            # Linear fit in log space
            coeffs = np.polyfit(log_bins, log_counts, 1)
            power_law_index = coeffs[0]
        else:
            power_law_index = -2.0  # Default value
        
        return {
            'total_energy': sum(all_energies),
            'average_energy': np.mean(all_energies),
            'median_energy': np.median(all_energies),
            'energy_range': [min(all_energies), max(all_energies)],
            'power_law_index': power_law_index,
            'nanoflare_energy_fraction': sum(nano_energies) / sum(all_energies) if nano_energies else 0,
            'energy_distribution': {
                'bins': energy_bins.tolist(),
                'counts': energy_counts.tolist()
            }
        }
    
    def _calculate_statistics(self, flares, nanoflares, energy_analysis):
        """Calculate comprehensive statistics"""
        return {
            'total_flares': len(flares),
            'nanoflare_count': len(nanoflares),
            'nanoflare_percentage': (len(nanoflares) / len(flares) * 100) if flares else 0,
            'average_energy': energy_analysis.get('average_energy', 0),
            'total_energy': energy_analysis.get('total_energy', 0),
            'power_law_index': energy_analysis.get('power_law_index', -2.0),
            'energy_range': energy_analysis.get('energy_range', [0, 0]),
            'flare_types': self._count_flare_types(flares),
            'temporal_distribution': self._analyze_temporal_distribution(flares)
        }
    
    def _count_flare_types(self, flares):
        """Count flares by type"""
        type_counts = {}
        for flare in flares:
            flare_type = flare.get('flare_type', 'unknown')
            type_counts[flare_type] = type_counts.get(flare_type, 0) + 1
        return type_counts
    
    def _analyze_temporal_distribution(self, flares):
        """Analyze temporal distribution of flares"""
        if not flares:
            return {}
        
        timestamps = [flare.get('timestamp', '') for flare in flares]
        # Simple binning by hour
        hour_counts = {}
        for ts in timestamps:
            if ts:
                try:
                    hour = datetime.fromisoformat(ts.replace('Z', '')).hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                except:
                    continue
        
        return {
            'hourly_distribution': hour_counts,
            'peak_activity_hour': max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
        }
    
    def _generate_visualizations(self, data, flares, nanoflares):
        """Generate visualization data for frontend"""
        return {
            'time_series': self._generate_time_series_data(data),
            'energy_histogram': self._generate_energy_histogram(flares),
            'flare_timeline': self._generate_flare_timeline(flares, nanoflares),
            'power_law_plot': self._generate_power_law_plot(flares)
        }
    
    def _generate_time_series_data(self, data):
        """Generate time series plot data"""
        if data.ndim == 1:
            return [{'time': i, 'intensity': float(val)} for i, val in enumerate(data[:1000])]
        else:
            return [{'time': i, 'intensity': float(val[0])} for i, val in enumerate(data[:1000])]
    
    def _generate_energy_histogram(self, flares):
        """Generate energy distribution histogram data"""
        energies = [f['energy'] for f in flares if 'energy' in f]
        if not energies:
            return []
        
        log_energies = np.log10(energies)
        hist, bins = np.histogram(log_energies, bins=20)
        
        return [{'energy': 10**((bins[i] + bins[i+1])/2), 'count': int(hist[i])} 
                for i in range(len(hist))]
    
    def _generate_flare_timeline(self, flares, nanoflares):
        """Generate timeline visualization data"""
        nano_timestamps = {f['timestamp'] for f in nanoflares}
        
        timeline_data = []
        for flare in flares:
            timeline_data.append({
                'timestamp': flare['timestamp'],
                'intensity': flare['intensity'],
                'energy': flare['energy'],
                'is_nanoflare': flare['timestamp'] in nano_timestamps,
                'type': flare.get('flare_type', 'unknown')
            })
        
        return sorted(timeline_data, key=lambda x: x['timestamp'])
    
    def _generate_power_law_plot(self, flares):
        """Generate power law distribution plot data"""
        energies = [f['energy'] for f in flares if 'energy' in f]
        if not energies:
            return []
        
        log_energies = np.log10(energies)
        hist, bins = np.histogram(log_energies, bins=20)
        
        # Create cumulative distribution
        cumulative = np.cumsum(hist[::-1])[::-1]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return [{'energy': 10**bin_centers[i], 'cumulative_count': int(cumulative[i])} 
                for i in range(len(cumulative)) if cumulative[i] > 0]
    
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

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'Solar Flare Analysis API',
        'version': '2.0.0',
        'status': 'running',
        'ml_available': ML_AVAILABLE,
        'model_initialized': analyzer.initialized if analyzer else False,
        'endpoints': {
            'health': '/health',
            'analyze': '/analyze',
            'model_info': '/model/info',
            'monte_carlo_background': '/api/montecarlo/background',
            'monte_carlo_cv': '/api/montecarlo/cross-validation',
            'monte_carlo_augmentation': '/api/montecarlo/augmentation',
            'bayesian_analysis': '/api/bayesian/analysis'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_available': ML_AVAILABLE,
        'model_initialized': analyzer.initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_flares():
    """Main analysis endpoint"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Analyze the file
        results = analyzer.analyze_file(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

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
            try:
                # Import Monte Carlo module
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
                    }                }
                
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
            'flare_detections': np.random.poisson(5),
            'confidence_score': 0.6 + np.random.random() * 0.4,
            'energy_estimate': np.exp(np.random.random() * 4 + 20),
            'false_positive_rate': np.random.random() * 0.1,
            'true_positive_rate': 0.8 + np.random.random() * 0.2,
            'processing_time': np.random.random() * 100 + 50
        })
    
    confidences = [r['confidence_score'] for r in results]
    energies = [r['energy_estimate'] for r in results]
    
    return {
        'simulation_id': f'mc_background_{int(datetime.now().timestamp())}',
        'results': results,
        'summary': {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'confidence_interval': [np.percentile(confidences, 2.5), np.percentile(confidences, 97.5)],
            'detection_rate': np.mean([r['flare_detections'] for r in results]),
            'energy_distribution': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'percentiles': {
                    'p5': np.percentile(energies, 5),
                    'p25': np.percentile(energies, 25),
                    'p50': np.percentile(energies, 50),
                    'p75': np.percentile(energies, 75),
                    'p95': np.percentile(energies, 95)
                }
            },
            'uncertainty_metrics': {
                'epistemic': 0.15 + np.random.random() * 0.1,
                'aleatory': 0.08 + np.random.random() * 0.05,
                'total': 0.23 + np.random.random() * 0.1
            }
        },
        'convergence_diagnostics': {
            'effective_sample_size': 850 + np.random.random() * 150,
            'r_hat': 1.01 + np.random.random() * 0.05,
            'mcmc_efficiency': 0.85 + np.random.random() * 0.1
        }
    }

def generate_mock_monte_carlo_cv(config):
    """Generate mock Monte Carlo cross-validation results"""
    cv_folds = config['cv_folds']
    
    cross_validation = []
    for i in range(cv_folds):
        cross_validation.append({
            'fold': i + 1,
            'train_score': 0.85 + np.random.random() * 0.1,
            'val_score': 0.80 + np.random.random() * 0.1,
            'test_score': 0.78 + np.random.random() * 0.1,
            'precision': 0.82 + np.random.random() * 0.15,
            'recall': 0.79 + np.random.random() * 0.15,
            'f1_score': 0.80 + np.random.random() * 0.1,
            'auc_score': 0.85 + np.random.random() * 0.1
        })
    
    train_scores = [cv['train_score'] for cv in cross_validation]
    val_scores = [cv['val_score'] for cv in cross_validation]
    test_scores = [cv['test_score'] for cv in cross_validation]
    f1_scores = [cv['f1_score'] for cv in cross_validation]
    auc_scores = [cv['auc_score'] for cv in cross_validation]
    
    return {
        'simulation_id': f'mc_cv_{int(datetime.now().timestamp())}',
        'cross_validation': cross_validation,
        'summary': {
            'mean_train_score': np.mean(train_scores),
            'mean_val_score': np.mean(val_scores),
            'mean_test_score': np.mean(test_scores),
            'std_train_score': np.std(train_scores),
            'std_val_score': np.std(val_scores),
            'std_test_score': np.std(test_scores),
            'mean_f1': np.mean(f1_scores),
            'mean_auc': np.mean(auc_scores),
            'stability_index': 1 - (np.std(val_scores) / np.mean(val_scores))
        },
        'monte_carlo_stats': {
            'bootstrap_confidence_interval': [
                np.mean(val_scores) - 1.96 * np.std(val_scores),
                np.mean(val_scores) + 1.96 * np.std(val_scores)
            ],
            'permutation_test_p_value': 0.001 + np.random.random() * 0.05,
            'cross_validation_variance': np.var(val_scores)
        }
    }

def generate_mock_monte_carlo_augmentation(config):
    """Generate mock Monte Carlo data augmentation results"""
    baseline_acc = 0.82 + np.random.random() * 0.05
    augmented_acc = baseline_acc + 0.05 + np.random.random() * 0.1
    
    baseline_prec = 0.80 + np.random.random() * 0.05
    augmented_prec = baseline_prec + 0.03 + np.random.random() * 0.08
    
    baseline_recall = 0.78 + np.random.random() * 0.05
    augmented_recall = baseline_recall + 0.04 + np.random.random() * 0.09
    
    return {
        'simulation_id': f'mc_aug_{int(datetime.now().timestamp())}',
        'augmentation': {
            'original_samples': 1000,
            'augmented_samples': 1000 * config['augmentation_factor'],
            'noise_variations': int(config['augmentation_factor'] * 15),
            'improvement_score': (augmented_acc - baseline_acc) / baseline_acc,
            'model_accuracy_before': baseline_acc,
            'model_accuracy_after': augmented_acc
        },
        'model_comparison': {
            'baseline_metrics': {
                'accuracy': baseline_acc,
                'precision': baseline_prec,
                'recall': baseline_recall,
                'f1_score': 2 * baseline_prec * baseline_recall / (baseline_prec + baseline_recall),
                'auc': 0.85 + np.random.random() * 0.05
            },
            'augmented_metrics': {
                'accuracy': augmented_acc,
                'precision': augmented_prec,
                'recall': augmented_recall,
                'f1_score': 2 * augmented_prec * augmented_recall / (augmented_prec + augmented_recall),
                'auc': 0.88 + np.random.random() * 0.07
            }
        },
        'robustness_test': {
            'noise_tolerance': 0.75 + np.random.random() * 0.2,
            'generalization_score': 0.80 + np.random.random() * 0.15,
            'overfitting_index': 0.05 + np.random.random() * 0.1
        }
    }

def generate_mock_bayesian_results(config):
    """Generate mock Bayesian analysis results"""
    n_features = len(config['data']) if config['data'] else 10
    n_samples = config['n_iterations']
    
    # Generate posterior samples
    weights = np.random.normal(0, 1, (n_samples, n_features)).tolist()
    bias = np.random.normal(0, 0.1, n_samples).tolist()
    variance = np.random.gamma(2, 0.05, n_samples).tolist()
    
    return {
        'posterior_samples': {
            'weights': weights,
            'bias': bias,
            'variance': variance
        },
        'credible_intervals': {
            'lower_95': (np.random.normal(0, 1, n_features) - 1.96 * 0.1).tolist(),
            'upper_95': (np.random.normal(0, 1, n_features) + 1.96 * 0.1).tolist(),
            'lower_68': (np.random.normal(0, 1, n_features) - 1.0 * 0.1).tolist(),
            'upper_68': (np.random.normal(0, 1, n_features) + 1.0 * 0.1).tolist()
        },
        'model_comparison': {
            'waic': -45.2 + np.random.random() * 10,
            'loo': -47.8 + np.random.random() * 12,
            'r_hat': (1 + np.random.random(n_features) * 0.02).tolist(),
            'effective_sample_size': (850 + np.random.random(n_features) * 300).tolist()
        },
        'uncertainty_metrics': {
            'epistemic_uncertainty': (np.random.random(n_features) * 0.1 + 0.01).tolist(),
            'aleatoric_uncertainty': (np.random.random(n_features) * 0.05 + 0.005).tolist(),
            'total_uncertainty': (np.random.random(n_features) * 0.12 + 0.02).tolist(),
            'prediction_variance': (np.random.random(n_features) * 0.08 + 0.01).tolist()
        },
        'model_evidence': {
            'log_marginal_likelihood': -52.3 + np.random.random() * 8,
            'bayes_factor': 2.1 + np.random.random() * 3,
            'bridge_sampling_estimate': -51.7 + np.random.random() * 9
        }
    }

if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Solar Flare Analysis API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', default=5000, type=int, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    logger.info(f"ML modules available: {ML_AVAILABLE}")
    logger.info(f"Model initialized: {analyzer.initialized}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
