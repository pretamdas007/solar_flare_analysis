'use client';

import React, { useState, useEffect } from 'react';
import { Brain, BarChart3, TrendingUp, Target, Zap, Clock, RefreshCw, Play, Settings, AlertTriangle, CheckCircle, Activity, Atom, Sparkles } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter } from 'recharts';

interface BayesianPrediction {
  model_name: string;
  mean_prediction: number;
  std_prediction: number;
  confidence_interval: {
    lower: number;
    upper: number;
    confidence_level: number;
  };
  posterior_samples: number[];
  convergence_diagnostics: {
    r_hat: number;
    ess: number;
    converged: boolean;
  };
}

interface BayesianAnalysis {
  predictions: BayesianPrediction[];
  model_comparison: {
    models: string[];
    bayes_factors: number[];
    best_model: string;
  };
  uncertainty_quantification: {
    epistemic_uncertainty: number;
    aleatoric_uncertainty: number;
    total_uncertainty: number;
  };
  calibration_metrics: {
    calibration_error: number;
    sharpness: number;
    reliability_diagram: Array<{
      bin_center: number;
      accuracy: number;
      confidence: number;
      count: number;
    }>;
  };
  timestamp: string;
}

interface BayesianConfig {
  n_samples: number;
  n_chains: number;
  burn_in: number;
  model_type: 'hierarchical' | 'gaussian_process' | 'neural_bayesian';
  prior_type: 'normal' | 'laplace' | 'horseshoe';
}

export default function BayesianUncertaintyDashboard() {
  const [analysis, setAnalysis] = useState<BayesianAnalysis | null>(null);
  const [config, setConfig] = useState<BayesianConfig>({
    n_samples: 1000,
    n_chains: 4,
    burn_in: 500,
    model_type: 'neural_bayesian',
    prior_type: 'normal'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'predictions' | 'uncertainty' | 'calibration' | 'convergence'>('predictions');

  // Fetch Bayesian analysis results
  const fetchBayesianAnalysis = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/bayesian/analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        const result = await response.json();
        setAnalysis(result);
      } else {
        throw new Error('Failed to fetch Bayesian analysis');
      }
    } catch (error) {
      console.error('Error fetching Bayesian analysis:', error);
      setError('Failed to fetch Bayesian analysis');
      
      // Fallback to mock data for development
      const mockAnalysis: BayesianAnalysis = {
        predictions: [
          {
            model_name: 'Bayesian Neural Network',
            mean_prediction: 0.742,
            std_prediction: 0.123,
            confidence_interval: {
              lower: 0.541,
              upper: 0.943,
              confidence_level: 0.95
            },
            posterior_samples: Array.from({ length: 100 }, () => 0.742 + (Math.random() - 0.5) * 0.246),
            convergence_diagnostics: {
              r_hat: 1.002,
              ess: 850,
              converged: true
            }
          },
          {
            model_name: 'Gaussian Process',
            mean_prediction: 0.687,
            std_prediction: 0.156,
            confidence_interval: {
              lower: 0.423,
              upper: 0.951,
              confidence_level: 0.95
            },
            posterior_samples: Array.from({ length: 100 }, () => 0.687 + (Math.random() - 0.5) * 0.312),
            convergence_diagnostics: {
              r_hat: 1.015,
              ess: 723,
              converged: true
            }
          },
          {
            model_name: 'Hierarchical Bayes',
            mean_prediction: 0.823,
            std_prediction: 0.089,
            confidence_interval: {
              lower: 0.679,
              upper: 0.967,
              confidence_level: 0.95
            },
            posterior_samples: Array.from({ length: 100 }, () => 0.823 + (Math.random() - 0.5) * 0.178),
            convergence_diagnostics: {
              r_hat: 0.998,
              ess: 945,
              converged: true
            }
          }
        ],
        model_comparison: {
          models: ['Bayesian Neural Network', 'Gaussian Process', 'Hierarchical Bayes'],
          bayes_factors: [2.34, 1.67, 3.12],
          best_model: 'Hierarchical Bayes'
        },
        uncertainty_quantification: {
          epistemic_uncertainty: 0.134,
          aleatoric_uncertainty: 0.089,
          total_uncertainty: 0.162
        },
        calibration_metrics: {
          calibration_error: 0.023,
          sharpness: 0.156,
          reliability_diagram: Array.from({ length: 10 }, (_, i) => ({
            bin_center: (i + 0.5) / 10,
            accuracy: 0.1 + Math.random() * 0.8,
            confidence: (i + 0.5) / 10,
            count: Math.floor(Math.random() * 100) + 10
          }))
        },
        timestamp: new Date().toISOString()
      };
      setAnalysis(mockAnalysis);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBayesianAnalysis();
  }, []);

  const getConvergenceColor = (rHat: number) => {
    if (rHat <= 1.01) return 'text-green-400';
    if (rHat <= 1.05) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getUncertaintyColor = (uncertainty: number) => {
    if (uncertainty <= 0.1) return 'text-green-400';
    if (uncertainty <= 0.2) return 'text-yellow-400';
    return 'text-red-400';
  };

  const renderPredictions = () => {
    if (!analysis || !Array.isArray(analysis.predictions) || analysis.predictions.length === 0) return null;

    return (
      <div className="space-y-6">
        {/* Prediction Results Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.isArray(analysis.predictions) ? analysis.predictions.map((pred, index) => (
            <div key={index} className="solar-card group hover:scale-105 transition-all duration-300">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-purple-400 stellar-icon" />
                  <span className="text-white font-medium">{pred.model_name}</span>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-bold ${
                  pred.convergence_diagnostics.converged ? 'status-online' : 'status-offline'
                }`}>
                  {pred.convergence_diagnostics.converged ? 'Converged' : 'Not Converged'}
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Mean Prediction:</span>
                  <span className="text-white font-bold">{pred.mean_prediction.toFixed(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Std Deviation:</span>
                  <span className="text-blue-400 font-mono">{pred.std_prediction.toFixed(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">95% CI:</span>
                  <span className="text-green-400 font-mono text-sm">
                    [{pred.confidence_interval.lower.toFixed(3)}, {pred.confidence_interval.upper.toFixed(3)}]
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">R̂:</span>
                  <span className={`font-mono ${getConvergenceColor(pred.convergence_diagnostics.r_hat)}`}>
                    {pred.convergence_diagnostics.r_hat.toFixed(3)}
                  </span>
                </div>
              </div>

              {/* Confidence Interval Visualization */}
              <div className="mt-4 space-y-2">
                <span className="text-gray-400 text-sm">Confidence Interval</span>
                <div className="relative h-3 bg-white/10 rounded-full overflow-hidden">
                  <div 
                    className="absolute h-full bg-gradient-to-r from-red-400 to-green-400 rounded-full"
                    style={{
                      left: `${pred.confidence_interval.lower * 100}%`,
                      width: `${(pred.confidence_interval.upper - pred.confidence_interval.lower) * 100}%`
                    }}
                  />
                  <div 
                    className="absolute w-1 h-full bg-white"
                    style={{ left: `${pred.mean_prediction * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )) : null}
        </div>

        {/* Posterior Distributions */}
        <div className="solar-glassmorphism p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 text-blue-400 mr-2" />
            Posterior Distributions
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {Array.isArray(analysis.predictions) ? analysis.predictions.slice(0, 2).map((pred, index) => (
              <div key={index} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <h4 className="text-white font-medium mb-3">{pred.model_name}</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={pred.posterior_samples.map((value, i) => ({ index: i, value }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="index" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#8B5CF6" 
                      fill="#8B5CF6" 
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )) : null}
          </div>
        </div>
      </div>
    );
  };

  const renderUncertainty = () => {
    if (!analysis || !analysis.model_comparison || !Array.isArray(analysis.model_comparison.models) || !Array.isArray(analysis.model_comparison.bayes_factors)) return null;

    return (
      <div className="space-y-6">
        {/* Uncertainty Decomposition */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="solar-card border-l-4 border-blue-500">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="h-5 w-5 text-blue-400 stellar-icon" />
              <span className="text-blue-400 font-medium">Epistemic Uncertainty</span>
            </div>
            <span className={`text-2xl font-bold ${getUncertaintyColor(analysis.uncertainty_quantification.epistemic_uncertainty)}`}>
              {analysis.uncertainty_quantification.epistemic_uncertainty.toFixed(3)}
            </span>
            <p className="text-gray-400 text-sm mt-2">Model uncertainty (reducible with more data)</p>
          </div>
          
          <div className="solar-card border-l-4 border-yellow-500">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="h-5 w-5 text-yellow-400 stellar-icon" />
              <span className="text-yellow-400 font-medium">Aleatoric Uncertainty</span>
            </div>
            <span className={`text-2xl font-bold ${getUncertaintyColor(analysis.uncertainty_quantification.aleatoric_uncertainty)}`}>
              {analysis.uncertainty_quantification.aleatoric_uncertainty.toFixed(3)}
            </span>
            <p className="text-gray-400 text-sm mt-2">Data uncertainty (irreducible noise)</p>
          </div>
          
          <div className="solar-card border-l-4 border-red-500">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="h-5 w-5 text-red-400 stellar-icon" />
              <span className="text-red-400 font-medium">Total Uncertainty</span>
            </div>
            <span className={`text-2xl font-bold ${getUncertaintyColor(analysis.uncertainty_quantification.total_uncertainty)}`}>
              {analysis.uncertainty_quantification.total_uncertainty.toFixed(3)}
            </span>
            <p className="text-gray-400 text-sm mt-2">Combined uncertainty estimate</p>
          </div>
        </div>

        {/* Model Comparison */}
        <div className="solar-glassmorphism p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
            Bayesian Model Comparison
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              {Array.isArray(analysis.model_comparison.models) ? analysis.model_comparison.models.map((model, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                  <span className="text-white font-medium">{model}</span>
                  <div className="flex items-center space-x-3">
                    <span className="text-gray-300 text-sm">Bayes Factor:</span>
                    <span className="text-blue-400 font-bold">
                      {analysis.model_comparison.bayes_factors[index].toFixed(2)}
                    </span>
                    {model === analysis.model_comparison.best_model && (
                      <CheckCircle className="h-4 w-4 text-green-400" />
                    )}
                  </div>
                </div>
              )) : null}
            </div>
            
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
              <h4 className="text-white font-medium mb-3">Bayes Factors</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={Array.isArray(analysis.model_comparison.models) && Array.isArray(analysis.model_comparison.bayes_factors) ? analysis.model_comparison.models.map((model, i) => ({
                  model: model.split(' ')[0],
                  bayes_factor: analysis.model_comparison.bayes_factors[i]
                })) : []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="model" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  />
                  <Bar dataKey="bayes_factor" fill="#8B5CF6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCalibration = () => {
    if (!analysis || !analysis.calibration_metrics || !Array.isArray(analysis.calibration_metrics.reliability_diagram)) return null;

    return (
      <div className="space-y-6">
        {/* Calibration Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="solar-card">
            <div className="flex items-center space-x-2 mb-4">
              <Target className="h-5 w-5 text-green-400 stellar-icon" />
              <h3 className="text-lg font-medium text-white">Calibration Error</h3>
            </div>
            <div className="text-center">
              <span className={`text-4xl font-bold ${
                analysis.calibration_metrics.calibration_error <= 0.05 ? 'text-green-400' : 
                analysis.calibration_metrics.calibration_error <= 0.1 ? 'text-yellow-400' : 
                'text-red-400'
              }`}>
                {(analysis.calibration_metrics.calibration_error * 100).toFixed(1)}%
              </span>
              <p className="text-gray-400 text-sm mt-2">Lower is better</p>
            </div>
          </div>
          
          <div className="solar-card">
            <div className="flex items-center space-x-2 mb-4">
              <Zap className="h-5 w-5 text-yellow-400 stellar-icon" />
              <h3 className="text-lg font-medium text-white">Sharpness</h3>
            </div>
            <div className="text-center">
              <span className="text-4xl font-bold text-blue-400">
                {analysis.calibration_metrics.sharpness.toFixed(3)}
              </span>
              <p className="text-gray-400 text-sm mt-2">Prediction precision</p>
            </div>
          </div>
        </div>

        {/* Reliability Diagram */}
        <div className="solar-glassmorphism p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 text-blue-400 mr-2" />
            Reliability Diagram
          </h3>
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={Array.isArray(analysis.calibration_metrics.reliability_diagram) ? analysis.calibration_metrics.reliability_diagram : []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="confidence" 
                  stroke="#9CA3AF"
                  domain={[0, 1]}
                  tickFormatter={(value) => (value * 100).toFixed(0) + '%'}
                />
                <YAxis 
                  dataKey="accuracy"
                  stroke="#9CA3AF"
                  domain={[0, 1]}
                  tickFormatter={(value) => (value * 100).toFixed(0) + '%'}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  formatter={(value, name) => [
                    name === 'accuracy' ? `${(value as number * 100).toFixed(1)}%` : value,
                    name
                  ]}
                />
                <Scatter dataKey="accuracy" fill="#8B5CF6" />
                <Line 
                  type="linear" 
                  dataKey="confidence" 
                  stroke="#10B981" 
                  strokeDasharray="5 5"
                  name="Perfect Calibration"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  const renderConvergence = () => {
    if (!analysis || !Array.isArray(analysis.predictions) || analysis.predictions.length === 0) return null;

    return (
      <div className="space-y-6">
        {/* Convergence Diagnostics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {Array.isArray(analysis.predictions) ? analysis.predictions.map((pred, index) => (
            <div key={index} className="solar-card">
              <div className="flex items-center space-x-2 mb-4">
                <Activity className="h-5 w-5 text-blue-400 stellar-icon" />
                <h3 className="text-lg font-medium text-white">{pred.model_name}</h3>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">R̂ Statistic:</span>
                  <span className={`font-bold ${getConvergenceColor(pred.convergence_diagnostics.r_hat)}`}>
                    {pred.convergence_diagnostics.r_hat.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">ESS:</span>
                  <span className="text-white font-bold">{pred.convergence_diagnostics.ess}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Status:</span>
                  <div className={`flex items-center space-x-1 ${
                    pred.convergence_diagnostics.converged ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {pred.convergence_diagnostics.converged ? 
                      <CheckCircle className="h-4 w-4" /> : 
                      <AlertTriangle className="h-4 w-4" />
                    }
                    <span className="font-bold">
                      {pred.convergence_diagnostics.converged ? 'Converged' : 'Not Converged'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )) : null}
        </div>

        {/* Trace Plots */}
        <div className="solar-glassmorphism p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
            MCMC Trace Plots
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {Array.isArray(analysis.predictions) ? analysis.predictions.slice(0, 2).map((pred, index) => (
              <div key={index} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <h4 className="text-white font-medium mb-3">{pred.model_name}</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={pred.posterior_samples.map((value, i) => ({ iteration: i, value }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="iteration" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#8B5CF6" 
                      strokeWidth={1}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )) : null}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex justify-center items-center mb-4">
          <Brain className="h-10 w-10 text-purple-400 stellar-icon mr-3" />
          <div>
            <h2 className="aurora-text text-3xl font-bold">Bayesian Uncertainty Analysis</h2>
            <p className="text-gray-300">Probabilistic Model Inference & Uncertainty Quantification</p>
          </div>
          <Atom className="h-10 w-10 text-blue-400 stellar-icon ml-3" />
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="solar-glassmorphism p-6">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center space-x-3">
            <Settings className="h-6 w-6 text-blue-400 stellar-icon" />
            <h3 className="aurora-text text-xl font-semibold">MCMC Configuration</h3>
          </div>
          <button
            onClick={fetchBayesianAnalysis}
            disabled={loading}
            className="plasma-button disabled:opacity-50 flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Run Analysis</span>
          </button>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <label className="block text-gray-300 text-sm mb-2">Samples</label>
            <input
              type="number"
              value={config.n_samples}
              onChange={(e) => setConfig({ ...config, n_samples: parseInt(e.target.value) })}
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
            />
          </div>
          <div>
            <label className="block text-gray-300 text-sm mb-2">Chains</label>
            <input
              type="number"
              value={config.n_chains}
              onChange={(e) => setConfig({ ...config, n_chains: parseInt(e.target.value) })}
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
            />
          </div>
          <div>
            <label className="block text-gray-300 text-sm mb-2">Burn-in</label>
            <input
              type="number"
              value={config.burn_in}
              onChange={(e) => setConfig({ ...config, burn_in: parseInt(e.target.value) })}
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
            />
          </div>
          <div>
            <label className="block text-gray-300 text-sm mb-2">Model Type</label>
            <select
              value={config.model_type}
              onChange={(e) => setConfig({ ...config, model_type: e.target.value as any })}
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
            >
              <option value="neural_bayesian">Neural Bayesian</option>
              <option value="gaussian_process">Gaussian Process</option>
              <option value="hierarchical">Hierarchical</option>
            </select>
          </div>
          <div>
            <label className="block text-gray-300 text-sm mb-2">Prior</label>
            <select
              value={config.prior_type}
              onChange={(e) => setConfig({ ...config, prior_type: e.target.value as any })}
              className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-white"
            >
              <option value="normal">Normal</option>
              <option value="laplace">Laplace</option>
              <option value="horseshoe">Horseshoe</option>
            </select>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-2 border-b border-white/20">
        {[
          { key: 'predictions', label: 'Predictions', icon: Brain },
          { key: 'uncertainty', label: 'Uncertainty', icon: Target },
          { key: 'calibration', label: 'Calibration', icon: BarChart3 },
          { key: 'convergence', label: 'Convergence', icon: TrendingUp }
        ].map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveView(tab.key as any)}
              className={`px-4 py-2 rounded-t-lg flex items-center space-x-2 transition-all duration-300 ${
                activeView === tab.key
                  ? 'bg-white/10 text-white border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Error Display */}
      {error && (
        <div className="solar-glassmorphism border-l-4 border-red-500 p-4 flex items-center space-x-2">
          <AlertTriangle className="h-5 w-5 text-red-400 stellar-icon" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="solar-glassmorphism p-8">
          <div className="flex items-center justify-center space-x-3">
            <div className="relative">
              <RefreshCw className="h-8 w-8 text-purple-400 animate-spin" />
              <div className="absolute inset-0 h-8 w-8 border-2 border-purple-400/30 rounded-full animate-pulse" />
            </div>
            <div className="text-center">
              <span className="text-white font-medium block">Running Bayesian Analysis</span>
              <span className="text-gray-400 text-sm">Sampling posterior distributions...</span>
            </div>
          </div>
        </div>
      )}

      {/* Content Views */}
      {!loading && analysis && (
        <>
          {activeView === 'predictions' && renderPredictions()}
          {activeView === 'uncertainty' && renderUncertainty()}
          {activeView === 'calibration' && renderCalibration()}
          {activeView === 'convergence' && renderConvergence()}
        </>
      )}

      {/* Timestamp */}
      {analysis && (
        <div className="text-center text-sm text-gray-400 flex items-center justify-center space-x-2">
          <Clock className="h-4 w-4" />
          <span>Analysis completed: {new Date(analysis.timestamp).toLocaleString()}</span>
        </div>
      )}
    </div>
  );
}