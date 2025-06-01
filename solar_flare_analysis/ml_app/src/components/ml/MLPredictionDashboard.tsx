'use client';

import React, { useState, useEffect } from 'react';
import { Activity, Zap, TrendingUp, Brain, AlertTriangle, BarChart3, RefreshCw, Play, Upload, Sun, Flame, Star, Atom, Sparkles, Clock, Target } from 'lucide-react';
import RealTimeVisualization from './RealTimeVisualization';
import ModelPerformanceGrid from './ModelPerformanceGrid';
import FlareAnalysisGrid from './FlareAnalysisGrid';
import DataLoaderComponent from './DataLoaderComponent';

interface PredictionResult {
  model_name: string;
  prediction: number | string;
  confidence: number;
  processing_time: number;
}

interface MLPredictionResponse {
  predictions: PredictionResult[];
  timestamp: string;
  input_shape: number[];
  status: string;
}

interface FlareAnalysis {
  total_flares: number;
  flare_classes: {
    A: number;
    B: number;
    C: number;
    M: number;
    X: number;
  };
  energy_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  peak_times: string[];
  confidence_scores: number[];
}

interface PlotData {
  plot_type: string;
  title: string;
  image_data: string; // base64 encoded image
  description: string;
}

export default function MLPredictionDashboard() {
  const [predictions, setPredictions] = useState<MLPredictionResponse | null>(null);
  const [flareAnalysis, setFlareAnalysis] = useState<FlareAnalysis | null>(null);
  const [plots, setPlots] = useState<PlotData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Fetch real-time ML predictions
  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      // Generate sample data for prediction
      const sampleData = Array.from({ length: 100 }, () => Math.random());
      
      const response = await fetch('/api/ml/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: sampleData,
          model_types: ['binary', 'multiclass', 'energy', 'cnn', 'minimal']
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictions(result);
      } else {
        throw new Error('Failed to fetch predictions');
      }
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError('Failed to fetch ML predictions');
    } finally {
      setLoading(false);
    }
  };

  // Fetch flare analysis results
  const fetchFlareAnalysis = async () => {
    try {
      const response = await fetch('/api/analysis/flares');
      if (response.ok) {
        const result = await response.json();
        setFlareAnalysis(result);
      }
    } catch (error) {
      console.error('Error fetching flare analysis:', error);
    }
  };

  // Fetch visualization plots
  const fetchPlots = async () => {
    try {
      const response = await fetch('/api/analysis/plots');
      if (response.ok) {
        const result = await response.json();
        setPlots(result.plots || []);
      }
    } catch (error) {
      console.error('Error fetching plots:', error);
    }
  };

  // Handle file upload for data analysis
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/analysis/flares', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setFlareAnalysis(result);
        // Fetch updated plots after analysis
        fetchPlots();
      } else {
        throw new Error('Failed to analyze data');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setError('Failed to analyze uploaded data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
    fetchFlareAnalysis();
    fetchPlots();
  }, []);
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'status-online';
    if (confidence >= 0.6) return 'status-warning';
    return 'status-offline';
  };

  const getModelIcon = (modelName: string) => {
    switch (modelName.toLowerCase()) {
      case 'binary_flare_classifier':
        return <Brain className="h-5 w-5 stellar-icon" />;
      case 'multiclass_flare_classifier':
        return <BarChart3 className="h-5 w-5 stellar-icon" />;
      case 'energy_regression_model':
        return <Zap className="h-5 w-5 stellar-icon" />;
      case 'cnn_flare_detector':
        return <Activity className="h-5 w-5 stellar-icon" />;
      default:
        return <TrendingUp className="h-5 w-5 stellar-icon" />;
    }
  };

  return (
    <div className="solar-body cosmic-grid min-h-screen">
      <div className="p-8 space-y-8">
        {/* Solar Hero Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center items-center mb-6">
            <Sun className="h-12 w-12 text-yellow-400 stellar-icon mr-4" />
            <div>
              <h1 className="aurora-text text-4xl font-bold mb-2">ML Prediction Dashboard</h1>
              <p className="text-gray-300 text-lg">Neural Network Solar Intelligence</p>
            </div>
            <Brain className="h-12 w-12 text-purple-400 stellar-icon ml-4" />
          </div>
        </div>

        {/* Control Panel with Solar Theme */}
        <div className="solar-glassmorphism p-6 rounded-2xl">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Atom className="h-6 w-6 text-blue-400 stellar-icon" />
              <h2 className="aurora-text text-xl font-semibold">Control Center</h2>
            </div>
            <div className="flex space-x-4">
              <label className="solar-button cursor-pointer flex items-center space-x-2">
                <Upload className="h-4 w-4" />
                <span>Upload Solar Data</span>
                <input
                  type="file"
                  accept=".nc,.h5,.csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <button
                onClick={fetchPredictions}
                disabled={loading}
                className="plasma-button disabled:opacity-50 flex items-center space-x-2"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh Predictions</span>
              </button>
            </div>
          </div>
      </div>

      {/* Error Display */}
        {error && (
          <div className="solar-glassmorphism border-l-4 border-red-500 p-4 flex items-center space-x-2 mb-6">
            <AlertTriangle className="h-5 w-5 text-red-400 stellar-icon" />
            <span className="text-red-400">{error}</span>
          </div>
        )}

        {/* Real-time Data Visualization */}
        <RealTimeVisualization />        {/* ML Predictions Grid */}
        {predictions && Array.isArray(predictions.predictions) && (
          <div className="solar-glassmorphism p-6 mb-8">
            <div className="flex items-center space-x-3 mb-6">
              <Brain className="h-6 w-6 text-purple-400 stellar-icon" />
              <h2 className="aurora-text text-xl font-semibold">Neural Network Predictions</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {predictions.predictions.map((pred, index) => (
                <div key={index} className="solar-card group hover:scale-105 transition-all duration-300">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      {getModelIcon(pred.model_name)}
                      <span className="text-white font-medium">{pred.model_name.replace(/_/g, ' ')}</span>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-bold ${getConfidenceColor(pred.confidence)}`}>
                      {(pred.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Prediction:</span>
                      <span className="text-white font-semibold bg-white/10 px-2 py-1 rounded">
                        {typeof pred.prediction === 'number' 
                          ? pred.prediction.toFixed(4) 
                          : pred.prediction}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300">Processing Time:</span>
                      <span className="text-blue-400 font-mono text-sm">{pred.processing_time.toFixed(3)}s</span>
                    </div>
                  </div>
                  <div className="mt-4 h-1 bg-white/10 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-400 to-purple-500 transition-all duration-500"
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 text-sm text-gray-400 flex items-center space-x-2">
              <Clock className="h-4 w-4" />
              <span>Last updated: {new Date(predictions.timestamp).toLocaleString()}</span>
            </div>
          </div>
        )}        {/* Solar Flare Analysis Results */}
        {flareAnalysis && (
          <div className="solar-glassmorphism p-6 mb-8">
            <div className="flex items-center space-x-3 mb-6">
              <Flame className="h-6 w-6 text-orange-400 stellar-icon" />
              <h2 className="aurora-text text-xl font-semibold">Solar Flare Analysis</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Detection Summary */}
              <div className="solar-card">
                <div className="flex items-center space-x-2 mb-4">
                  <Target className="h-5 w-5 text-blue-400 stellar-icon" />
                  <h3 className="text-lg font-medium text-white">Detection Summary</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Total Flares:</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-white font-bold text-xl">{flareAnalysis.total_flares}</span>
                      <Sparkles className="h-4 w-4 text-yellow-400 stellar-icon" />
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Mean Confidence:</span>
                    <span className="status-online font-bold">
                      {Array.isArray(flareAnalysis.confidence_scores) && flareAnalysis.confidence_scores.length > 0
                        ? (flareAnalysis.confidence_scores.reduce((a, b) => a + b, 0) / flareAnalysis.confidence_scores.length * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="mt-4 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-green-400 to-emerald-500 stellar-glow"
                      style={{ width: `${Array.isArray(flareAnalysis.confidence_scores) && flareAnalysis.confidence_scores.length > 0
                        ? (flareAnalysis.confidence_scores.reduce((a, b) => a + b, 0) / flareAnalysis.confidence_scores.length * 100)
                        : 0}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Flare Classification */}
              <div className="solar-card">
                <div className="flex items-center space-x-2 mb-4">
                  <BarChart3 className="h-5 w-5 text-purple-400 stellar-icon" />
                  <h3 className="text-lg font-medium text-white">Flare Classification</h3>
                </div>
                <div className="space-y-3">
                  {flareAnalysis.flare_classes && Object.entries(flareAnalysis.flare_classes).map(([class_name, count]) => (
                    <div key={class_name} className="flex justify-between items-center">
                      <span className="text-gray-300 flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${
                          class_name === 'X' ? 'bg-red-500' :
                          class_name === 'M' ? 'bg-orange-500' :
                          class_name === 'C' ? 'bg-yellow-500' :
                          'bg-blue-500'
                        }`} />
                        <span>Class {class_name}:</span>
                      </span>
                      <span className="text-white font-bold bg-white/10 px-2 py-1 rounded">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Energy Statistics */}
              <div className="solar-card">
                <div className="flex items-center space-x-2 mb-4">
                  <Zap className="h-5 w-5 text-yellow-400 stellar-icon" />
                  <h3 className="text-lg font-medium text-white">Energy Statistics</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Mean Energy:</span>
                    <span className="text-white font-mono text-sm bg-white/10 px-2 py-1 rounded">{
                      flareAnalysis.energy_stats && typeof flareAnalysis.energy_stats.mean === 'number'
                        ? flareAnalysis.energy_stats.mean.toExponential(2)
                        : 'N/A'
                    }</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Max Energy:</span>
                    <span className="text-yellow-400 font-mono text-sm bg-yellow-400/10 px-2 py-1 rounded border border-yellow-400/20">{
                      flareAnalysis.energy_stats && typeof flareAnalysis.energy_stats.max === 'number'
                        ? flareAnalysis.energy_stats.max.toExponential(2)
                        : 'N/A'
                    }</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Std Deviation:</span>
                    <span className="text-white font-mono text-sm bg-white/10 px-2 py-1 rounded">{
                      flareAnalysis.energy_stats && typeof flareAnalysis.energy_stats.std === 'number'
                        ? flareAnalysis.energy_stats.std.toExponential(2)
                        : 'N/A'
                    }</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}        {/* Cosmic Data Visualization */}
        {plots.length > 0 && (
          <div className="solar-glassmorphism p-6 mb-8">
            <div className="flex items-center space-x-3 mb-6">
              <BarChart3 className="h-6 w-6 text-blue-400 stellar-icon" />
              <h2 className="aurora-text text-xl font-semibold">Cosmic Data Visualization</h2>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {plots.map((plot, index) => (
                <div key={index} className="solar-card group">
                  <div className="flex items-center space-x-2 mb-4">
                    <TrendingUp className="h-5 w-5 text-green-400 stellar-icon" />
                    <h3 className="text-lg font-medium text-white">{plot.title}</h3>
                  </div>
                  <div className="mb-4 overflow-hidden rounded-xl border border-white/20">
                    <img
                      src={`data:image/png;base64,${plot.image_data}`}
                      alt={plot.title}
                      className="w-full h-64 object-contain bg-gradient-to-br from-gray-900 to-gray-800 hover:scale-105 transition-transform duration-500"
                    />
                  </div>
                  <p className="text-gray-300 text-sm leading-relaxed">{plot.description}</p>
                  <div className="mt-3 flex items-center text-xs text-gray-400">
                    <Clock className="h-3 w-3 mr-1" />
                    Generated at {new Date().toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}        {/* Component Integration */}
        <DataLoaderComponent />
        <FlareAnalysisGrid />
        <ModelPerformanceGrid />

        {/* Processing State */}
        {loading && (
          <div className="solar-glassmorphism p-6">
            <div className="flex items-center justify-center space-x-3">
              <div className="relative">
                <RefreshCw className="h-8 w-8 text-blue-400 animate-spin" />
                <div className="absolute inset-0 h-8 w-8 border-2 border-blue-400/30 rounded-full animate-pulse" />
              </div>
              <div className="text-center">
                <span className="text-white font-medium block">Processing Neural Networks</span>
                <span className="text-gray-400 text-sm">Analyzing solar dynamics...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
