'use client';

import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, Cell, PieChart, Pie
} from 'recharts';
import { 
  Zap, Brain, TrendingUp, Clock, Target, Award,
  Play, RefreshCw, Download, Settings, GitCompare,
  CheckCircle, XCircle, AlertTriangle, Info, BarChart3
} from 'lucide-react';

// Mock data for comparison metrics
const generateComparisonData = () => {
  return {
    accuracy: [
      { method: 'Traditional Threshold', value: 78, color: '#f59e0b' },
      { method: 'Peak Detection', value: 82, color: '#f59e0b' },
      { method: 'Enhanced CNN-LSTM', value: 94, color: '#8b5cf6' },
      { method: 'Transformer Model', value: 91, color: '#8b5cf6' },
      { method: 'Ensemble ML', value: 96, color: '#8b5cf6' }
    ],
    processingTime: [
      { method: 'Traditional', realTime: 0.5, batchTime: 2.1, color: '#f59e0b' },
      { method: 'ML Models', realTime: 1.2, batchTime: 0.8, color: '#8b5cf6' }
    ],
    detectionStats: [
      { metric: 'True Positives', traditional: 145, ml: 189 },
      { metric: 'False Positives', traditional: 42, ml: 18 },
      { metric: 'False Negatives', traditional: 38, ml: 12 },
      { metric: 'True Negatives', traditional: 2847, ml: 2853 }
    ],
    radarData: [
      { metric: 'Accuracy', traditional: 78, ml: 94, fullMark: 100 },
      { metric: 'Precision', traditional: 77, ml: 91, fullMark: 100 },
      { metric: 'Recall', traditional: 79, ml: 94, fullMark: 100 },
      { metric: 'F1-Score', traditional: 78, ml: 93, fullMark: 100 },
      { metric: 'Speed', traditional: 95, ml: 85, fullMark: 100 },
      { metric: 'Robustness', traditional: 65, ml: 88, fullMark: 100 }
    ]
  };
};

const generateAdvancedMetrics = () => {
  return {
    energyEstimation: {
      traditional: { mae: 0.45, rmse: 0.62, r2: 0.73 },
      ml: { mae: 0.18, rmse: 0.24, r2: 0.91 }
    },
    flareClassification: {
      traditional: { accuracy: 68, precision: 65, recall: 71 },
      ml: { accuracy: 89, precision: 87, recall: 91 }
    },
    backgroundSubtraction: {
      traditional: { snrImprovement: 2.3, artifacts: 12 },
      ml: { snrImprovement: 4.7, artifacts: 3 }
    },
    overlappingFlares: {
      traditional: { separated: 23, missed: 15 },
      ml: { separated: 34, missed: 4 }
    }
  };
};

const generateCostBenefit = () => {
  return {
    computational: [
      { category: 'Training Cost', traditional: 0, ml: 85 },
      { category: 'Inference Cost', traditional: 45, ml: 65 },
      { category: 'Maintenance', traditional: 30, ml: 40 },
      { category: 'Hardware Requirements', traditional: 20, ml: 75 }
    ],
    benefits: [
      { category: 'Accuracy Gain', value: 18, description: 'ML shows 18% higher accuracy' },
      { category: 'Processing Speed', value: -15, description: 'ML is 15% slower in real-time' },
      { category: 'False Positive Reduction', value: 57, description: '57% fewer false positives' },
      { category: 'Automation Level', value: 45, description: '45% less manual intervention' }
    ]
  };
};

export default function ComparisonPage() {
  const [comparisonData, setComparisonData] = useState(generateComparisonData());
  const [advancedMetrics, setAdvancedMetrics] = useState(generateAdvancedMetrics());
  const [costBenefit, setCostBenefit] = useState(generateCostBenefit());
  const [runningComparison, setRunningComparison] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState('full-dataset');
  const [timeRange, setTimeRange] = useState('2023-2024');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const runComparison = () => {
    setRunningComparison(true);
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setRunningComparison(false);
          // Regenerate data to simulate new results
          setComparisonData(generateComparisonData());
          setAdvancedMetrics(generateAdvancedMetrics());
          return 100;
        }
        return prev + 8;
      });
    }, 400);
  };

  const exportComparison = () => {
    const results = {
      comparisonData,
      advancedMetrics,
      costBenefit,
      parameters: {
        dataset: selectedDataset,
        timeRange,
        timestamp: new Date().toISOString()
      }
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], 
      { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ml_vs_traditional_comparison.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <GitCompare className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">ML vs Traditional Comparison</h1>
              <p className="text-gray-600">Compare machine learning and traditional methods for solar flare analysis</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={exportComparison}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center space-x-2"
            >
              <Download className="h-4 w-4" />
              <span>Export</span>
            </button>
            <button
              onClick={runComparison}
              disabled={runningComparison}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center space-x-2"
            >
              {runningComparison ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>{runningComparison ? 'Comparing...' : 'Run Comparison'}</span>
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        {runningComparison && (
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Running comprehensive comparison analysis...</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Comparison Parameters */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="h-5 w-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">Comparison Parameters</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="full-dataset">Full Dataset (2020-2024)</option>
              <option value="validation-set">Validation Set</option>
              <option value="test-set">Test Set</option>
              <option value="cross-validation">Cross-Validation</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Time Range
            </label>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="2023-2024">2023-2024 (Recent)</option>
              <option value="2020-2024">2020-2024 (Full Range)</option>
              <option value="solar-max">Solar Maximum Periods</option>
              <option value="solar-min">Solar Minimum Periods</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Display Options
            </label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showAdvanced}
                  onChange={(e) => setShowAdvanced(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">Show advanced metrics</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">ML Accuracy Advantage</p>
              <p className="text-2xl font-bold text-green-600">+18%</p>
            </div>
            <Target className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">False Positive Reduction</p>
              <p className="text-2xl font-bold text-blue-600">-57%</p>
            </div>
            <CheckCircle className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Processing Speed</p>
              <p className="text-2xl font-bold text-orange-600">-15%</p>
            </div>
            <Clock className="h-8 w-8 text-orange-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Overall Score</p>
              <p className="text-2xl font-bold text-purple-600">ML Wins</p>
            </div>
            <Award className="h-8 w-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Main Comparison Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Accuracy Comparison */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Accuracy Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData.accuracy}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="method" angle={-45} textAnchor="end" height={80} />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(value) => [`${value}%`, 'Accuracy']} />
              <Bar dataKey="value" fill="#8b5cf6">
                {comparisonData.accuracy.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart Comparison */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Radar</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={comparisonData.radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar
                name="Traditional"
                dataKey="traditional"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.3}
              />
              <Radar
                name="ML Models"
                dataKey="ml"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.3}
              />
              <Legend />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Performance Metrics</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Detection Statistics */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Detection Statistics</h4>
            <div className="space-y-2 text-sm">
              {comparisonData.detectionStats.map((stat, index) => (
                <div key={index} className="flex justify-between">
                  <span className="text-gray-600">{stat.metric}:</span>
                  <div className="flex space-x-2">
                    <span className="text-orange-600 font-mono">{stat.traditional}</span>
                    <span className="text-purple-600 font-mono">{stat.ml}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Energy Estimation */}
          <div className="p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Energy Estimation</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">MAE:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.energyEstimation.traditional.mae}</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.energyEstimation.ml.mae}</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">RMSE:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.energyEstimation.traditional.rmse}</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.energyEstimation.ml.rmse}</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">R²:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.energyEstimation.traditional.r2}</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.energyEstimation.ml.r2}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Classification Accuracy */}
          <div className="p-4 bg-green-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Flare Classification</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Accuracy:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.flareClassification.traditional.accuracy}%</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.flareClassification.ml.accuracy}%</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Precision:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.flareClassification.traditional.precision}%</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.flareClassification.ml.precision}%</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Recall:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.flareClassification.traditional.recall}%</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.flareClassification.ml.recall}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Overlapping Flares */}
          <div className="p-4 bg-purple-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3">Overlapping Flares</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Separated:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.overlappingFlares.traditional.separated}</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.overlappingFlares.ml.separated}</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Missed:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">{advancedMetrics.overlappingFlares.traditional.missed}</span>
                  <span className="text-purple-600 font-mono">{advancedMetrics.overlappingFlares.ml.missed}</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Success Rate:</span>
                <div className="flex space-x-2">
                  <span className="text-orange-600 font-mono">
                    {Math.round(advancedMetrics.overlappingFlares.traditional.separated / 
                    (advancedMetrics.overlappingFlares.traditional.separated + advancedMetrics.overlappingFlares.traditional.missed) * 100)}%
                  </span>
                  <span className="text-purple-600 font-mono">
                    {Math.round(advancedMetrics.overlappingFlares.ml.separated / 
                    (advancedMetrics.overlappingFlares.ml.separated + advancedMetrics.overlappingFlares.ml.missed) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Cost-Benefit Analysis */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost-Benefit Analysis</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Computational Costs */}
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Computational Costs (Relative)</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={costBenefit.computational}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="traditional" fill="#f59e0b" name="Traditional" />
                <Bar dataKey="ml" fill="#8b5cf6" name="ML Models" />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Benefits Summary */}
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Key Benefits</h4>
            <div className="space-y-4">
              {costBenefit.benefits.map((benefit, index) => (
                <div key={index} className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-medium text-gray-900">{benefit.category}</span>
                    <span className={`font-bold ${benefit.value > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {benefit.value > 0 ? '+' : ''}{benefit.value}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">{benefit.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommendations</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-green-50 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <h4 className="font-medium text-green-900">Recommended: ML Approach</h4>
            </div>
            <ul className="text-sm text-green-800 space-y-1">
              <li>• 18% higher accuracy</li>
              <li>• 57% fewer false positives</li>
              <li>• Better overlapping flare separation</li>
              <li>• Superior energy estimation</li>
            </ul>
          </div>

          <div className="p-4 bg-yellow-50 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <AlertTriangle className="h-5 w-5 text-yellow-600" />
              <h4 className="font-medium text-yellow-900">Consider Traditional When</h4>
            </div>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• Real-time processing critical</li>
              <li>• Limited computational resources</li>
              <li>• Interpretability required</li>
              <li>• Small dataset available</li>
            </ul>
          </div>

          <div className="p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Info className="h-5 w-5 text-blue-600" />
              <h4 className="font-medium text-blue-900">Hybrid Approach</h4>
            </div>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Traditional for real-time alerts</li>
              <li>• ML for detailed post-analysis</li>
              <li>• Ensemble for critical decisions</li>
              <li>• Best of both worlds</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
