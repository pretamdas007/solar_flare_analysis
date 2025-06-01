'use client';

import React, { useState, useEffect } from 'react';
import { Brain, BarChart3, Zap, Activity, TrendingUp, Target, Award, Clock, Cpu, Sun, Flame, Star } from 'lucide-react';

interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  training_time: number;
  inference_time: number;
  model_size: number;
  parameters: number;
}

interface ModelComparison {
  models: ModelMetrics[];
  best_accuracy: string;
  best_f1: string;
  fastest_inference: string;
  timestamp: string;
}

export default function ModelPerformanceGrid() {
  const [metrics, setMetrics] = useState<ModelComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<'accuracy' | 'precision' | 'recall' | 'f1_score' | 'training_time' | 'inference_time'>('accuracy');

  // Mock data for model performance
  const mockMetrics: ModelComparison = {
    models: [
      {
        model_name: 'Binary Flare Classifier',
        accuracy: 0.943,
        precision: 0.925,
        recall: 0.891,
        f1_score: 0.908,
        training_time: 245.7,
        inference_time: 0.023,
        model_size: 2.1,
        parameters: 158420
      },
      {
        model_name: 'Multiclass Flare Classifier',
        accuracy: 0.887,
        precision: 0.872,
        recall: 0.864,
        f1_score: 0.868,
        training_time: 312.4,
        inference_time: 0.031,
        model_size: 3.7,
        parameters: 245680
      },
      {
        model_name: 'Energy Regression Model',
        accuracy: 0.901,
        precision: 0.889,
        recall: 0.876,
        f1_score: 0.882,
        training_time: 198.2,
        inference_time: 0.018,
        model_size: 1.8,
        parameters: 123456
      },
      {
        model_name: 'CNN Flare Detector',
        accuracy: 0.952,
        precision: 0.948,
        recall: 0.937,
        f1_score: 0.942,
        training_time: 456.8,
        inference_time: 0.042,
        model_size: 5.2,
        parameters: 387921
      },
      {
        model_name: 'Minimal Flare Model',
        accuracy: 0.821,
        precision: 0.798,
        recall: 0.834,
        f1_score: 0.816,
        training_time: 89.3,
        inference_time: 0.012,
        model_size: 0.8,
        parameters: 67890
      }
    ],
    best_accuracy: 'CNN Flare Detector',
    best_f1: 'CNN Flare Detector',
    fastest_inference: 'Minimal Flare Model',
    timestamp: new Date().toISOString()
  };

  useEffect(() => {
    // Simulate loading delay
    setTimeout(() => {
      setMetrics(mockMetrics);
      setLoading(false);
    }, 1000);
  }, []);
  const getModelIcon = (modelName: string) => {
    if (modelName.includes('Binary')) return <Brain className="h-5 w-5 stellar-icon" />;
    if (modelName.includes('Multiclass')) return <BarChart3 className="h-5 w-5 stellar-icon" />;
    if (modelName.includes('Energy')) return <Zap className="h-5 w-5 stellar-icon" />;
    if (modelName.includes('CNN')) return <Activity className="h-5 w-5 stellar-icon" />;
    return <TrendingUp className="h-5 w-5 stellar-icon" />;
  };

  const getPerformanceColor = (value: number, metric: string) => {
    if (metric === 'inference_time') {
      // Lower is better for inference time
      if (value <= 0.02) return 'status-online';
      if (value <= 0.03) return 'status-warning';
      return 'status-offline';
    } else {
      // Higher is better for other metrics
      if (value >= 0.9) return 'status-online';
      if (value >= 0.8) return 'status-warning';
      return 'status-offline';
    }
  };

  const getBarWidth = (value: number, max: number) => {
    return `${(value / max) * 100}%`;
  };
  if (loading) {
    return (
      <div className="solar-glassmorphism p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-white/20 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map(i => (
              <div key={i} className="h-16 bg-white/10 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!metrics) return null;

  const maxValue = Math.max(...metrics.models.map(m => m[selectedMetric] as number));

  return (
    <div className="solar-glassmorphism p-6 mb-8">
      {/* Cosmic Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-purple-400 stellar-icon" />
          <h2 className="aurora-text text-xl font-semibold">Neural Network Performance</h2>
          <Star className="h-5 w-5 text-blue-400 stellar-icon" />
        </div>
        <div className="flex space-x-2">
          {(['accuracy', 'precision', 'recall', 'f1_score'] as const).map(metric => (
            <button
              key={metric}
              onClick={() => setSelectedMetric(metric)}
              className={`px-3 py-1 rounded-lg text-sm transition-all duration-300 ${
                selectedMetric === metric
                  ? 'solar-button'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20 border border-white/20'
              }`}
            >
              {metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </button>
          ))}
        </div>
      </div>      {/* Solar Champions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="solar-card border-l-4 border-green-500">
          <div className="flex items-center space-x-2 mb-2">
            <Award className="h-5 w-5 text-green-400 stellar-icon" />
            <span className="text-green-400 font-medium">Best Accuracy</span>
          </div>
          <span className="text-white font-bold">{metrics.best_accuracy}</span>
        </div>
        <div className="solar-card border-l-4 border-blue-500">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="h-5 w-5 text-blue-400 stellar-icon" />
            <span className="text-blue-400 font-medium">Best F1-Score</span>
          </div>
          <span className="text-white font-bold">{metrics.best_f1}</span>
        </div>
        <div className="solar-card border-l-4 border-purple-500">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="h-5 w-5 text-purple-400 stellar-icon" />
            <span className="text-purple-400 font-medium">Fastest Inference</span>
          </div>
          <span className="text-white font-bold">{metrics.fastest_inference}</span>
        </div>
      </div>      {/* Neural Network Performance Grid */}
      <div className="space-y-4">
        {metrics.models.map((model, index) => (
          <div key={index} className="solar-card group hover:scale-[1.02] transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                {getModelIcon(model.model_name)}
                <span className="text-white font-medium">{model.model_name}</span>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-bold ${getPerformanceColor(model[selectedMetric] as number, selectedMetric)}`}>
                {selectedMetric === 'training_time' 
                  ? `${(model[selectedMetric] as number).toFixed(1)}s`
                  : selectedMetric === 'inference_time'
                  ? `${(model[selectedMetric] as number * 1000).toFixed(1)}ms`
                  : `${((model[selectedMetric] as number) * 100).toFixed(1)}%`}
              </div>
            </div>

            {/* Cosmic Performance Bar */}
            <div className="mb-4">
              <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-3 rounded-full transition-all duration-700 stellar-glow ${
                    getPerformanceColor(model[selectedMetric] as number, selectedMetric).includes('online')
                      ? 'bg-gradient-to-r from-green-400 to-emerald-500'
                      : getPerformanceColor(model[selectedMetric] as number, selectedMetric).includes('warning')
                      ? 'bg-gradient-to-r from-yellow-400 to-orange-500'
                      : 'bg-gradient-to-r from-red-400 to-red-600'
                  }`}
                  style={{ width: getBarWidth(model[selectedMetric] as number, maxValue) }}
                ></div>
              </div>
            </div>            {/* Core Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-4">
              <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                <span className="text-gray-300">Accuracy:</span>
                <span className="text-white font-semibold">{(model.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                <span className="text-gray-300">F1-Score:</span>
                <span className="text-white font-semibold">{(model.f1_score * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                <span className="text-gray-300">Size:</span>
                <span className="text-white font-semibold">{model.model_size.toFixed(1)}MB</span>
              </div>
              <div className="flex justify-between items-center bg-white/5 p-2 rounded">
                <span className="text-gray-300">Params:</span>
                <span className="text-white font-semibold">{(model.parameters / 1000).toFixed(0)}K</span>
              </div>
            </div>

            {/* Advanced Performance Metrics */}
            <div className="pt-3 border-t border-white/10">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-blue-400 stellar-icon" />
                  <span className="text-gray-300">Training:</span>
                  <span className="text-white font-mono text-xs bg-blue-400/10 px-2 py-1 rounded">{model.training_time.toFixed(1)}s</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Cpu className="h-4 w-4 text-green-400 stellar-icon" />
                  <span className="text-gray-300">Inference:</span>
                  <span className="text-white font-mono text-xs bg-green-400/10 px-2 py-1 rounded">{(model.inference_time * 1000).toFixed(1)}ms</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Target className="h-4 w-4 text-purple-400 stellar-icon" />
                  <span className="text-gray-300">Precision:</span>
                  <span className="text-white font-mono text-xs bg-purple-400/10 px-2 py-1 rounded">{(model.precision * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>      {/* Cosmic Footer */}
      <div className="mt-6 text-sm text-gray-400 text-center flex items-center justify-center space-x-2">
        <Clock className="h-4 w-4" />
        <span>Last updated: {new Date(metrics.timestamp).toLocaleString()}</span>
      </div>
    </div>
  );
}
