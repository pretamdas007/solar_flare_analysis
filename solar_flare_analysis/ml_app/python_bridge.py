#!/usr/bin/env python3
"""
Bridge script to connect Next.js backend with Python ML model
This script provides REST API endpoints for the solar flare analysis
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Simple mock ML model for testing
class MockFlareModel:
    def __init__(self):
        self.model_trained = True
    
    def analyze_flares(self, data):
        """Mock analysis function"""
        num_flares = np.random.randint(20, 100)
        
        # Generate mock flare data
        flares = []
        for i in range(num_flares):
            flare = {
                'timestamp': f"2024-{1 + i//30:02d}-{1 + i%30:02d}T{i%24:02d}:00:00Z",
                'intensity': np.random.exponential(200) + 50,
                'energy': np.power(10, np.random.uniform(26, 30)),
                'alpha': np.random.normal(0, 2),
            }
            flares.append(flare)
        
        # Identify nanoflares
        nanoflares = [f for f in flares if abs(f['alpha']) > 2]
        
        return {
            'separated_flares': flares,
            'nanoflares': nanoflares,
            'average_energy': np.mean([f['energy'] for f in flares]),
            'power_law_index': np.random.uniform(-2.5, -1.5)
        }

app = Flask(__name__)
CORS(app)

# Global model instance
model = MockFlareModel()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'models': {
            'flare_decomposition': 'loaded' if model is not None else 'error'
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded GOES data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Mock data analysis (since we don't have actual GOES data processing)
        results = model.analyze_flares({})
        
        # Convert results to JSON-serializable format
        response_data = {
            'separatedFlares': format_flare_data(results['separated_flares']),
            'nanoflares': format_flare_data(results['nanoflares']),
            'statistics': {
                'totalFlares': len(results['separated_flares']),
                'nanoflareCount': len(results['nanoflares']),
                'averageEnergy': float(results['average_energy']),
                'powerLawIndex': float(results['power_law_index'])
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the ML model"""
    try:
        # Mock retraining
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

def format_flare_data(flares):
    """Format flare data for JSON response"""
    formatted = []
    for flare in flares:
        formatted.append({
            'timestamp': flare.get('timestamp', ''),
            'intensity': float(flare.get('intensity', 0)),
            'energy': float(flare.get('energy', 0)),
            'alpha': float(flare.get('alpha', 0)),
            'flareType': determine_flare_type(flare.get('energy', 0))
        })
    return formatted

def determine_flare_type(energy):
    """Determine flare type based on energy"""
    if energy < 1e27:
        return 'nano'
    elif energy < 1e28:
        return 'micro'
    elif energy < 1e29:
        return 'minor'
    elif energy < 1e30:
        return 'major'
    else:
        return 'X-class'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solar Flare Analysis API Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='localhost', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("Model initialized successfully (mock mode)")
    print(f"Starting Solar Flare Analysis API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
