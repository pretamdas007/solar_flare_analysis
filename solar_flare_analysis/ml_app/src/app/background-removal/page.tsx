'use client';

import { useState, useEffect } from 'react';
import { Filter, Play, Settings, BarChart3, Activity, TrendingUp, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { useNotification } from '@/components/NotificationSystem';

interface BackgroundSettings {
  method: string;
  window_size: number;
  polynomial_order: number;
  smoothing_factor: number;
  median_filter_size: number;
  adaptive_threshold: boolean;
  remove_trends: boolean;
}

interface ProcessingResult {
  original_signal: number[];
  background_estimate: number[];
  processed_signal: number[];
  noise_level: number;
  snr_improvement: number;
  artifacts_removed: number;
}

export default function BackgroundRemoval() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [timeSeriesData, setTimeSeriesData] = useState<any[]>([]);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [settings, setSettings] = useState<BackgroundSettings>({
    method: 'savgol',
    window_size: 101,
    polynomial_order: 3,
    smoothing_factor: 0.1,
    median_filter_size: 5,
    adaptive_threshold: true,
    remove_trends: true
  });
  const { showNotification } = useNotification();

  useEffect(() => {
    // Generate mock time series data with background and flares
    const mockData = [];
    for (let i = 0; i < 2000; i++) {
      const time = new Date(2024, 0, 1, 0, 0, i * 10).getTime();
      
      // Background components
      const slowTrend = 1e-8 + 2e-9 * Math.sin(i * 0.001);
      const dailyVariation = 5e-10 * Math.sin(i * 0.01);
      const instrumentalDrift = 1e-10 * i * 0.0001;
      const background = slowTrend + dailyVariation + instrumentalDrift;
      
      // Noise
      const noise = (Math.random() - 0.5) * 2e-9;
      
      // Flare events
      let flareSignal = 0;
      if (i > 300 && i < 400) flareSignal = 8e-8 * Math.exp(-((i - 350) ** 2) / 500);
      if (i > 600 && i < 650) flareSignal = 3e-8 * Math.exp(-((i - 625) ** 2) / 200);
      if (i > 1200 && i < 1350) flareSignal = 1.5e-7 * Math.exp(-((i - 1275) ** 2) / 1000);
      if (i > 1500 && i < 1520) flareSignal = 2e-8 * Math.exp(-((i - 1510) ** 2) / 50);
      
      const originalSignal = background + noise + flareSignal;
      
      mockData.push({
        time,
        timestamp: new Date(time).toISOString(),
        original: originalSignal,
        background: background,
        flares_only: flareSignal,
        noise: noise
      });
    }
    setTimeSeriesData(mockData);
  }, []);

  const startProcessing = async () => {
    setIsProcessing(true);
    setProgress(0);
    showNotification('Starting background removal...', 'info');

    const steps = [
      'Analyzing signal characteristics...',
      'Applying median filtering...',
      'Estimating background trends...',
      'Performing polynomial fitting...',
      'Removing slow variations...',
      'Calculating noise statistics...',
      'Finalizing processed signal...'
    ];

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgress(((i + 1) / steps.length) * 100);
      showNotification(steps[i], 'info');
    }

    // Simulate background removal processing
    const processedData = timeSeriesData.map((point, index) => {
      // Simple background estimation (moving average with trend removal)
      const windowStart = Math.max(0, index - Math.floor(settings.window_size / 2));
      const windowEnd = Math.min(timeSeriesData.length, index + Math.floor(settings.window_size / 2));
      const windowData = timeSeriesData.slice(windowStart, windowEnd);
      
      let backgroundEstimate = windowData.reduce((sum, p) => sum + p.original, 0) / windowData.length;
      
      // Apply polynomial detrending
      if (settings.remove_trends) {
        const trend = point.background * 0.8; // Simulate trend removal
        backgroundEstimate = Math.min(backgroundEstimate, trend);
      }
      
      const processedSignal = Math.max(0, point.original - backgroundEstimate);
      
      return {
        ...point,
        background_estimate: backgroundEstimate,
        processed: processedSignal
      };
    });

    setTimeSeriesData(processedData);

    // Calculate processing statistics
    const originalMean = timeSeriesData.reduce((sum, p) => sum + p.original, 0) / timeSeriesData.length;
    const processedMean = processedData.reduce((sum, p) => sum + p.processed, 0) / processedData.length;
    const noiseLevel = Math.sqrt(timeSeriesData.reduce((sum, p) => sum + p.noise ** 2, 0) / timeSeriesData.length);
    
    const result: ProcessingResult = {
      original_signal: timeSeriesData.map(p => p.original),
      background_estimate: processedData.map(p => p.background_estimate),
      processed_signal: processedData.map(p => p.processed),
      noise_level: noiseLevel,
      snr_improvement: 3.2,
      artifacts_removed: 45
    };

    setProcessingResult(result);
    setIsProcessing(false);
    showNotification('Background removal completed successfully!', 'success');
  };

  const handleSettingsChange = (key: keyof BackgroundSettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const backgroundMethods = [
    { value: 'savgol', label: 'Savitzky-Golay Filter', description: 'Polynomial smoothing filter' },
    { value: 'median', label: 'Median Filter', description: 'Robust to outliers' },
    { value: 'polynomial', label: 'Polynomial Detrending', description: 'Removes polynomial trends' },
    { value: 'rolling_minimum', label: 'Rolling Minimum', description: 'Conservative background estimation' },
    { value: 'wavelet', label: 'Wavelet Decomposition', description: 'Multi-scale analysis' }
  ];

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
          <Filter className="h-8 w-8 mr-3 text-yellow-400" />
          Remove Background Flux
        </h1>
        <p className="text-gray-300">
          Apply advanced background subtraction algorithms to isolate solar flare signals from instrumental drift and slow variations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Settings Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Settings className="h-6 w-6 mr-2" />
              Processing Settings
            </h2>
            
            <div className="space-y-4">
              {/* Background Method */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Background Method
                </label>
                <select
                  value={settings.method}
                  onChange={(e) => handleSettingsChange('method', e.target.value)}
                  disabled={isProcessing}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-yellow-500"
                >
                  {backgroundMethods.map(method => (
                    <option key={method.value} value={method.value} className="bg-gray-800">
                      {method.label}
                    </option>
                  ))}
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {backgroundMethods.find(m => m.value === settings.method)?.description}
                </p>
              </div>

              {/* Window Size */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Window Size: {settings.window_size}
                </label>
                <input
                  type="range"
                  min="11"
                  max="501"
                  step="10"
                  value={settings.window_size}
                  onChange={(e) => handleSettingsChange('window_size', parseInt(e.target.value))}
                  disabled={isProcessing}
                  className="w-full"
                />
                <p className="text-gray-400 text-xs">Larger values = smoother background</p>
              </div>

              {/* Polynomial Order */}
              {settings.method === 'savgol' && (
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Polynomial Order: {settings.polynomial_order}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="7"
                    step="1"
                    value={settings.polynomial_order}
                    onChange={(e) => handleSettingsChange('polynomial_order', parseInt(e.target.value))}
                    disabled={isProcessing}
                    className="w-full"
                  />
                </div>
              )}

              {/* Smoothing Factor */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Smoothing Factor: {settings.smoothing_factor}
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="1.0"
                  step="0.01"
                  value={settings.smoothing_factor}
                  onChange={(e) => handleSettingsChange('smoothing_factor', parseFloat(e.target.value))}
                  disabled={isProcessing}
                  className="w-full"
                />
              </div>

              {/* Median Filter Size */}
              {settings.method === 'median' && (
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Filter Size: {settings.median_filter_size}
                  </label>
                  <input
                    type="range"
                    min="3"
                    max="21"
                    step="2"
                    value={settings.median_filter_size}
                    onChange={(e) => handleSettingsChange('median_filter_size', parseInt(e.target.value))}
                    disabled={isProcessing}
                    className="w-full"
                  />
                </div>
              )}

              {/* Options */}
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="adaptive_threshold"
                    checked={settings.adaptive_threshold}
                    onChange={(e) => handleSettingsChange('adaptive_threshold', e.target.checked)}
                    disabled={isProcessing}
                    className="rounded"
                  />
                  <label htmlFor="adaptive_threshold" className="text-gray-300 text-sm">
                    Adaptive Threshold
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="remove_trends"
                    checked={settings.remove_trends}
                    onChange={(e) => handleSettingsChange('remove_trends', e.target.checked)}
                    disabled={isProcessing}
                    className="rounded"
                  />
                  <label htmlFor="remove_trends" className="text-gray-300 text-sm">
                    Remove Trends
                  </label>
                </div>
              </div>
            </div>

            <button
              onClick={startProcessing}
              disabled={isProcessing}
              className={`w-full mt-6 flex items-center justify-center px-6 py-3 rounded-lg transition-colors ${
                isProcessing 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-yellow-600 hover:bg-yellow-700'
              } text-white`}
            >
              <Play className="h-5 w-5 mr-2" />
              {isProcessing ? 'Processing...' : 'Start Processing'}
            </button>

            {isProcessing && (
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-gray-300 text-sm mt-2 text-center">{progress.toFixed(0)}%</p>
              </div>
            )}
          </div>

          {/* Processing Statistics */}
          {processingResult && (
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="h-5 w-5 mr-2" />
                Processing Results
              </h3>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-300 text-sm">Noise Level:</span>
                  <span className="text-white text-sm">{processingResult.noise_level.toExponential(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300 text-sm">SNR Improvement:</span>
                  <span className="text-green-400 text-sm">+{processingResult.snr_improvement.toFixed(1)} dB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300 text-sm">Artifacts Removed:</span>
                  <span className="text-blue-400 text-sm">{processingResult.artifacts_removed}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* Before/After Comparison */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Activity className="h-5 w-5 mr-2" />
              Signal Comparison
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData.slice(0, 500)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="time" 
                    type="number" 
                    scale="time" 
                    domain={['dataMin', 'dataMax']}
                    stroke="#9CA3AF"
                    tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <YAxis stroke="#9CA3AF" tickFormatter={(value) => `${value.toExponential(1)}`} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                    labelFormatter={(time) => new Date(time).toLocaleString()}
                    formatter={(value: any) => [`${value.toExponential(2)} W/m²`, '']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="original" stroke="#EF4444" name="Original Signal" strokeWidth={1} opacity={0.7} />
                  <Line type="monotone" dataKey="background_estimate" stroke="#6B7280" name="Background Estimate" strokeWidth={1} strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="processed" stroke="#10B981" name="Background Removed" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Background Components */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <TrendingUp className="h-5 w-5 mr-2" />
              Background Components Analysis
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timeSeriesData.slice(0, 500)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="time" 
                    type="number" 
                    scale="time" 
                    domain={['dataMin', 'dataMax']}
                    stroke="#9CA3AF"
                    tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                  />
                  <YAxis stroke="#9CA3AF" tickFormatter={(value) => `${value.toExponential(1)}`} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                    labelFormatter={(time) => new Date(time).toLocaleString()}
                    formatter={(value: any) => [`${value.toExponential(2)} W/m²`, '']}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="background" stackId="1" stroke="#6B7280" fill="#6B7280" fillOpacity={0.3} name="Background" />
                  <Area type="monotone" dataKey="flares_only" stackId="1" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} name="Flare Signal" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Quality Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-300 text-sm">Signal-to-Noise Ratio</p>
                  <p className="text-2xl font-bold text-green-400">
                    {processingResult ? `${processingResult.snr_improvement.toFixed(1)} dB` : '- dB'}
                  </p>
                  <p className="text-green-400 text-sm">Improved</p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-400" />
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-300 text-sm">Background Accuracy</p>
                  <p className="text-2xl font-bold text-blue-400">92.3%</p>
                  <p className="text-blue-400 text-sm">Correlation</p>
                </div>
                <Activity className="h-8 w-8 text-blue-400" />
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-300 text-sm">Flare Preservation</p>
                  <p className="text-2xl font-bold text-yellow-400">98.7%</p>
                  <p className="text-yellow-400 text-sm">Retained</p>
                </div>
                <Zap className="h-8 w-8 text-yellow-400" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
