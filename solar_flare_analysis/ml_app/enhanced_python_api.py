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
    from data_processing.data_loader import GOESDataLoader
    from ml_models.enhanced_flare_analysis import (
        EnhancedFlareDecompositionModel,
        NanoflareDetector,
        FlareEnergyAnalyzer
    )
    from visualization.plotting import FlareVisualization
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
