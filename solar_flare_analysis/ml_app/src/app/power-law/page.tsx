'use client';

import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, ReferenceLine
} from 'recharts';
import { 
  TrendingUp, Activity, Calculator, Database, 
  BarChart3, Settings, Play, Download, RefreshCw,
  Zap, AlertCircle, CheckCircle, Info
} from 'lucide-react';

// Mock data for power-law analysis
const generatePowerLawData = () => {
  return Array.from({ length: 50 }, (_, i) => {
    const energy = Math.pow(10, 28 + i * 0.1); // Energy from 10^28 to 10^33 ergs
    const frequency = Math.pow(energy / 1e30, -1.6) * 1000; // Power law with index -1.6
    const noise = 1 + (Math.random() - 0.5) * 0.2;
    return {
      energy: energy,
      logEnergy: Math.log10(energy),
      frequency: frequency * noise,
      logFrequency: Math.log10(frequency * noise),
      cumulativeFreq: frequency * noise * 2
    };
  }).sort((a, b) => a.energy - b.energy);
};

const generateFlareDistribution = () => {
  const classes = ['A', 'B', 'C', 'M', 'X'];
  return classes.map(className => ({
    class: className,
    count: Math.floor(Math.random() * 1000) + 100,
    energy: Math.pow(10, 27 + classes.indexOf(className) * 1.5),
    percentage: Math.random() * 100
  }));
};

const generateFittingResults = () => {
  return {
    powerLawIndex: -1.62,
    powerLawError: 0.08,
    rSquared: 0.94,
    chiSquared: 15.7,
    ksStatistic: 0.12,
    pValue: 0.15,
    energyRange: { min: 1e28, max: 1e32 },
    breakPoints: [
      { energy: 1e29, index: -1.2 },
      { energy: 5e30, index: -1.8 }
    ]
  };
};

export default function PowerLawAnalysisPage() {
  const [powerLawData, setPowerLawData] = useState(generatePowerLawData());
  const [flareDistribution, setFlareDistribution] = useState(generateFlareDistribution());
  const [fittingResults, setFittingResults] = useState(generateFittingResults());
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedMethod, setSelectedMethod] = useState('maximum-likelihood');
  const [energyRange, setEnergyRange] = useState({ min: 28, max: 32 });
  const [showCumulative, setShowCumulative] = useState(false);

  const runPowerLawAnalysis = () => {
    setAnalysisRunning(true);
    setAnalysisProgress(0);
    
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setAnalysisRunning(false);
          setPowerLawData(generatePowerLawData());
          setFittingResults(generateFittingResults());
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  const exportResults = () => {
    const results = {
      powerLawData,
      fittingResults,
      analysisParameters: {
        method: selectedMethod,
        energyRange,
        timestamp: new Date().toISOString()
      }
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], 
      { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'power_law_analysis_results.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <TrendingUp className="h-6 w-6 text-purple-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Power-Law Analysis</h1>
              <p className="text-gray-600">Analyze energy distribution and scaling laws of solar flares</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={exportResults}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center space-x-2"
            >
              <Download className="h-4 w-4" />
              <span>Export</span>
            </button>
            <button
              onClick={runPowerLawAnalysis}
              disabled={analysisRunning}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center space-x-2"
            >
              {analysisRunning ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span>{analysisRunning ? 'Analyzing...' : 'Run Analysis'}</span>
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        {analysisRunning && (
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Power-law fitting in progress...</span>
              <span>{analysisProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${analysisProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Analysis Parameters */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="h-5 w-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">Analysis Parameters</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Fitting Method
            </label>
            <select
              value={selectedMethod}
              onChange={(e) => setSelectedMethod(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="maximum-likelihood">Maximum Likelihood</option>
              <option value="least-squares">Least Squares</option>
              <option value="kolmogorov-smirnov">Kolmogorov-Smirnov</option>
              <option value="anderson-darling">Anderson-Darling</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Energy Range (log₁₀ erg)
            </label>
            <div className="flex space-x-2">
              <input
                type="number"
                value={energyRange.min}
                onChange={(e) => setEnergyRange(prev => ({ ...prev, min: Number(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Min"
              />
              <input
                type="number"
                value={energyRange.max}
                onChange={(e) => setEnergyRange(prev => ({ ...prev, max: Number(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Max"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Display Options
            </label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showCumulative}
                  onChange={(e) => setShowCumulative(e.target.checked)}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                <span className="ml-2 text-sm text-gray-700">Show cumulative distribution</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Results Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Power-Law Index</p>
              <p className="text-2xl font-bold text-gray-900">
                {fittingResults.powerLawIndex.toFixed(2)} ± {fittingResults.powerLawError.toFixed(2)}
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-purple-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">R² Value</p>
              <p className="text-2xl font-bold text-gray-900">{fittingResults.rSquared.toFixed(3)}</p>
            </div>
            <BarChart3 className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">P-Value</p>
              <p className="text-2xl font-bold text-gray-900">{fittingResults.pValue.toFixed(3)}</p>
            </div>
            <Calculator className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">KS Statistic</p>
              <p className="text-2xl font-bold text-gray-900">{fittingResults.ksStatistic.toFixed(3)}</p>
            </div>
            <Activity className="h-8 w-8 text-orange-600" />
          </div>
        </div>
      </div>

      {/* Main Analysis Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Power-Law Distribution */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Energy Distribution</h3>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">Log-Log Scale</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={powerLawData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="logEnergy" 
                type="number"
                domain={['dataMin', 'dataMax']}
                label={{ value: 'log₁₀(Energy) [erg]', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                dataKey="logFrequency"
                type="number"
                label={{ value: 'log₁₀(Frequency)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip formatter={(value, name) => [typeof value === 'number' ? value.toFixed(2) : value, name]} />
              <Scatter dataKey="logFrequency" fill="#8b5cf6" />
              <ReferenceLine 
                slope={fittingResults.powerLawIndex} 
                stroke="#ef4444" 
                strokeDasharray="5 5"
                strokeWidth={2}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Flare Class Distribution */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Flare Class Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={flareDistribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6">
                {flareDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={
                    ['#fef3c7', '#fed7aa', '#fca5a5', '#f87171', '#dc2626'][index]
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Results */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Fitting Results</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Statistical Tests</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Chi-squared:</span>
                  <span className="font-mono">{fittingResults.chiSquared.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">KS Statistic:</span>
                  <span className="font-mono">{fittingResults.ksStatistic.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">P-value:</span>
                  <span className="font-mono">{fittingResults.pValue.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">R² value:</span>
                  <span className="font-mono">{fittingResults.rSquared.toFixed(3)}</span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-blue-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Energy Range</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Minimum Energy:</span>
                  <span className="font-mono">{fittingResults.energyRange.min.toExponential(1)} erg</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Maximum Energy:</span>
                  <span className="font-mono">{fittingResults.energyRange.max.toExponential(1)} erg</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Span (decades):</span>
                  <span className="font-mono">{Math.log10(fittingResults.energyRange.max / fittingResults.energyRange.min).toFixed(1)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-green-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Interpretation</h4>
              <div className="space-y-3 text-sm">
                <div className="flex items-start space-x-2">
                  {fittingResults.rSquared > 0.9 ? (
                    <CheckCircle className="h-4 w-4 text-green-600 mt-0.5" />
                  ) : (
                    <AlertCircle className="h-4 w-4 text-yellow-600 mt-0.5" />
                  )}
                  <span className="text-gray-700">
                    {fittingResults.rSquared > 0.9 
                      ? "Excellent power-law fit (R² > 0.9)"
                      : "Moderate power-law fit"
                    }
                  </span>
                </div>
                <div className="flex items-start space-x-2">
                  {fittingResults.pValue > 0.05 ? (
                    <CheckCircle className="h-4 w-4 text-green-600 mt-0.5" />
                  ) : (
                    <AlertCircle className="h-4 w-4 text-red-600 mt-0.5" />
                  )}
                  <span className="text-gray-700">
                    {fittingResults.pValue > 0.05 
                      ? "Cannot reject power-law hypothesis (p > 0.05)"
                      : "Power-law hypothesis may be rejected"
                    }
                  </span>
                </div>
                <div className="flex items-start space-x-2">
                  <Info className="h-4 w-4 text-blue-600 mt-0.5" />
                  <span className="text-gray-700">
                    Index ≈ -1.6 suggests self-organized criticality
                  </span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-purple-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Physical Significance</h4>
              <p className="text-sm text-gray-700">
                The power-law index of {fittingResults.powerLawIndex.toFixed(2)} is consistent with 
                avalanche models of magnetic reconnection in the solar corona, supporting theories 
                of self-organized criticality in solar flare production.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
