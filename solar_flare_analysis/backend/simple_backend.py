#!/usr/bin/env python3
"""
Simple Flask Backend for Solar Flare Analysis
Lightweight version with minimal dependencies
"""

import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simple in-memory storage
app_data = {
    'models_loaded': True,
    'last_prediction': None,
    'analysis_results': {},
    'system_status': 'healthy'
}

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_models_loaded': app_data['models_loaded'],
        'backend_available': True
    })

@app.route('/api/models/predict', methods=['POST'])
def predict():
    """Mock ML prediction endpoint"""
    try:
        data = request.get_json()
        
        # Simulate ML predictions
        predictions = {
            'binary_classification': {
                'flare_detected': True,
                'confidence': 0.85,
                'probability': 0.75
            },
            'multiclass_classification': {
                'predicted_class': 'M_class',
                'probabilities': {
                    'no_flare': 0.15,
                    'C_class': 0.25,
                    'M_class': 0.45,
                    'X_class': 0.15
                }
            },
            'energy_estimation': {
                'estimated_energy': 1.2e-5,
                'energy_class': 'M1.2'
            }
        }
        
        app_data['last_prediction'] = predictions
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/flares', methods=['GET'])
def get_flare_analysis():
    """Return mock flare analysis results"""
    mock_flares = [
        {
            'id': 1,
            'timestamp': '2024-06-01T10:30:00Z',
            'class': 'M',
            'peak_flux': 1.2e-5,
            'duration_minutes': 15,
            'confidence': 0.92
        },
        {
            'id': 2,
            'timestamp': '2024-06-01T14:45:00Z',
            'class': 'C',
            'peak_flux': 3.4e-6,
            'duration_minutes': 8,
            'confidence': 0.78
        },
        {
            'id': 3,
            'timestamp': '2024-06-01T18:20:00Z',
            'class': 'X',
            'peak_flux': 5.2e-4,
            'duration_minutes': 25,
            'confidence': 0.95
        }
    ]
    
    return jsonify({
        'status': 'success',
        'flares': mock_flares,
        'total_count': len(mock_flares),
        'analysis_time': datetime.now().isoformat()
    })

@app.route('/api/plots/time-series', methods=['GET'])
def get_time_series_data():
    """Return mock time series data for plotting"""
    # Generate mock GOES XRS data
    timestamps = []
    xrs_a_flux = []
    xrs_b_flux = []
    
    base_time = datetime.now()
    for i in range(100):
        timestamps.append((base_time.timestamp() + i * 60) * 1000)  # milliseconds
        
        # Base background level with some flare spikes
        base_a = 1e-8 + np.random.normal(0, 1e-9)
        base_b = 5e-9 + np.random.normal(0, 5e-10)
        
        # Add flare spikes at certain points
        if i in [30, 45, 70]:
            flare_factor = np.random.uniform(10, 100)
            base_a *= flare_factor
            base_b *= flare_factor * 0.5
        
        xrs_a_flux.append(max(base_a, 1e-10))
        xrs_b_flux.append(max(base_b, 1e-10))
    
    return jsonify({
        'status': 'success',
        'data': {
            'timestamps': timestamps,
            'xrs_a': xrs_a_flux,
            'xrs_b': xrs_b_flux
        },
        'metadata': {
            'start_time': timestamps[0],
            'end_time': timestamps[-1],
            'data_points': len(timestamps)
        }
    })

@app.route('/api/analysis/statistics', methods=['GET'])
def get_statistics():
    """Return analysis statistics"""
    stats = {
        'total_flares_detected': 3,
        'flare_distribution': {
            'X_class': 1,
            'M_class': 1,
            'C_class': 1,
            'B_class': 0
        },
        'average_confidence': 0.88,
        'data_coverage_hours': 24,
        'processing_time_ms': 1250,
        'model_accuracy': 0.942,
        'last_updated': datetime.now().isoformat()
    }
    
    return jsonify({
        'status': 'success',
        'statistics': stats
    })

@app.route('/api/data/load', methods=['POST'])
def load_data():
    """Mock data loading endpoint"""
    try:
        data = request.get_json()
        filename = data.get('filename', 'sample_data.nc')
        
        # Simulate data loading
        result = {
            'filename': filename,
            'status': 'loaded',
            'records_loaded': 10000,
            'time_range': {
                'start': '2024-06-01T00:00:00Z',
                'end': '2024-06-01T23:59:59Z'
            },
            'data_quality': 'good',
            'missing_data_percentage': 0.02
        }
        
        return jsonify({
            'status': 'success',
            'result': result,
            'message': f'Successfully loaded {filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Simple Solar Flare Analysis Backend")
    print("Backend will be available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
