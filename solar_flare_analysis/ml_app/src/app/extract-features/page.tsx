'use client';

import { useState, useEffect } from 'react';
import { Target, Play, Settings, BarChart3, Zap, Activity, Filter, Eye } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { useNotification } from '@/components/NotificationSystem';

interface FlareEvent {
  id: string;
  start_time: string;
  peak_time: string;
  end_time: string;
  intensity: number;
  duration: number;
  energy: number;
  alpha_parameter: number;
  confidence: number;
  classification: string;
  is_nanoflare: boolean;
  overlapping: boolean;
}

interface ExtractionSettings {
  sensitivity_threshold: number;
  min_duration: number;
  max_duration: number;
  overlap_detection: boolean;
  nanoflare_alpha_threshold: number;
  background_subtraction: boolean;
  smoothing_window: number;
}

export default function ExtractFeatures() {
  const [isExtracting, setIsExtracting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [detectedFlares, setDetectedFlares] = useState<FlareEvent[]>([]);
  const [selectedFlare, setSelectedFlare] = useState<FlareEvent | null>(null);
  const [settings, setSettings] = useState<ExtractionSettings>({
    sensitivity_threshold: 0.75,
    min_duration: 60,
    max_duration: 3600,
    overlap_detection: true,
    nanoflare_alpha_threshold: 2.0,
    background_subtraction: true,
    smoothing_window: 5
  });
  const { showNotification } = useNotification();

  // Mock time series data for visualization
  const [timeSeriesData, setTimeSeriesData] = useState<any[]>([]);

  useEffect(() => {
    // Generate mock time series data
    const mockData = [];
    for (let i = 0; i < 1000; i++) {
      const time = new Date(2024, 0, 1, 0, 0, i * 30).getTime();
      const baseSignal = 1e-8 + Math.sin(i * 0.01) * 2e-9;
      const noise = (Math.random() - 0.5) * 1e-9;
      
      // Add some flare events
      let flareSignal = 0;      if (i > 200 && i < 250) flareSignal = 5e-8 * Math.exp(-((i - 225) ** 2) / 100);
      if (i > 400 && i < 420) flareSignal = 2e-8 * Math.exp(-((i - 410) ** 2) / 50);
      if (i > 700 && i < 780) flareSignal = 8e-8 * Math.exp(-((i - 740) ** 2) / 200);
      
      mockData.push({
        time,
        timestamp: new Date(time).toISOString(),
        intensity: baseSignal + noise + flareSignal,
        background: baseSignal,
        processed: baseSignal + flareSignal // After noise reduction
      });
    }
    setTimeSeriesData(mockData);
  }, []);

  const startExtraction = async () => {
    setIsExtracting(true);
    setProgress(0);
    setDetectedFlares([]);
    showNotification('Starting flare extraction...', 'info');

    // Simulate extraction process
    const steps = [
      'Loading time series data...',
      'Applying background subtraction...',
      'Detecting flare candidates...',
      'Separating overlapping events...',
      'Calculating flare parameters...',
      'Classifying nanoflares...',
      'Finalizing results...'
    ];

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      setProgress(((i + 1) / steps.length) * 100);
      showNotification(steps[i], 'info');
    }

    // Generate mock detected flares
    const mockFlares: FlareEvent[] = [
      {
        id: 'flare_001',
        start_time: '2024-01-01T01:40:00Z',
        peak_time: '2024-01-01T01:52:30Z',
        end_time: '2024-01-01T02:05:00Z',
        intensity: 5.2e-7,
        duration: 1500,
        energy: 2.3e24,
        alpha_parameter: 2.4,
        confidence: 0.94,
        classification: 'C-class',
        is_nanoflare: true,
        overlapping: false
      },
      {
        id: 'flare_002',
        start_time: '2024-01-01T03:20:00Z',
        peak_time: '2024-01-01T03:25:15Z',
        end_time: '2024-01-01T03:30:00Z',
        intensity: 1.8e-7,
        duration: 600,
        energy: 8.7e23,
        alpha_parameter: 1.8,
        confidence: 0.87,
        classification: 'B-class',
        is_nanoflare: false,
        overlapping: true
      },
      {
        id: 'flare_003',
        start_time: '2024-01-01T05:45:00Z',
        peak_time: '2024-01-01T06:02:45Z',
        end_time: '2024-01-01T06:25:00Z',
        intensity: 8.9e-7,
        duration: 2400,
        energy: 5.1e24,
        alpha_parameter: 2.7,
        confidence: 0.98,
        classification: 'M-class',
        is_nanoflare: true,
        overlapping: false
      }
    ];

    setDetectedFlares(mockFlares);
    setIsExtracting(false);
    showNotification(`Extraction complete! Found ${mockFlares.length} flare events.`, 'success');
  };

  const handleSettingsChange = (key: keyof ExtractionSettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const formatEnergy = (energy: number): string => {
    const exponent = Math.floor(Math.log10(energy));
    const mantissa = energy / Math.pow(10, exponent);
    return `${mantissa.toFixed(1)} × 10^${exponent}`;
  };

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'X-class': return 'text-red-400';
      case 'M-class': return 'text-orange-400';
      case 'C-class': return 'text-yellow-400';
      case 'B-class': return 'text-green-400';
      case 'A-class': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
          <Target className="h-8 w-8 mr-3 text-green-400" />
          Extract Flare Characteristics
        </h1>
        <p className="text-gray-300">
          Identify and separate overlapping solar flares using advanced ML techniques. Extract key parameters for each detected event.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Settings Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Settings className="h-6 w-6 mr-2" />
              Extraction Settings
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Sensitivity Threshold: {settings.sensitivity_threshold}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={settings.sensitivity_threshold}
                  onChange={(e) => handleSettingsChange('sensitivity_threshold', parseFloat(e.target.value))}
                  disabled={isExtracting}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-1 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Min Duration (s)
                  </label>
                  <input
                    type="number"
                    value={settings.min_duration}
                    onChange={(e) => handleSettingsChange('min_duration', parseInt(e.target.value))}
                    disabled={isExtracting}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-green-500"
                    min="1"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Max Duration (s)
                  </label>
                  <input
                    type="number"
                    value={settings.max_duration}
                    onChange={(e) => handleSettingsChange('max_duration', parseInt(e.target.value))}
                    disabled={isExtracting}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-green-500"
                    min="1"
                  />
                </div>
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Nanoflare α Threshold: {settings.nanoflare_alpha_threshold}
                </label>
                <input
                  type="range"
                  min="1.5"
                  max="3.0"
                  step="0.1"
                  value={settings.nanoflare_alpha_threshold}
                  onChange={(e) => handleSettingsChange('nanoflare_alpha_threshold', parseFloat(e.target.value))}
                  disabled={isExtracting}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Smoothing Window: {settings.smoothing_window}
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={settings.smoothing_window}
                  onChange={(e) => handleSettingsChange('smoothing_window', parseInt(e.target.value))}
                  disabled={isExtracting}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="overlap_detection"
                    checked={settings.overlap_detection}
                    onChange={(e) => handleSettingsChange('overlap_detection', e.target.checked)}
                    disabled={isExtracting}
                    className="rounded"
                  />
                  <label htmlFor="overlap_detection" className="text-gray-300 text-sm">
                    Overlap Detection
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="background_subtraction"
                    checked={settings.background_subtraction}
                    onChange={(e) => handleSettingsChange('background_subtraction', e.target.checked)}
                    disabled={isExtracting}
                    className="rounded"
                  />
                  <label htmlFor="background_subtraction" className="text-gray-300 text-sm">
                    Background Subtraction
                  </label>
                </div>
              </div>
            </div>

            <button
              onClick={startExtraction}
              disabled={isExtracting}
              className={`w-full mt-6 flex items-center justify-center px-6 py-3 rounded-lg transition-colors ${
                isExtracting 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-green-600 hover:bg-green-700'
              } text-white`}
            >
              <Play className="h-5 w-5 mr-2" />
              {isExtracting ? 'Extracting...' : 'Start Extraction'}
            </button>

            {isExtracting && (
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-gray-300 text-sm mt-2 text-center">{progress.toFixed(0)}%</p>
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* Time Series Visualization */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2" />
              X-ray Flux Time Series
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData.slice(0, 200)}>
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
                  <Line type="monotone" dataKey="intensity" stroke="#EF4444" name="Raw Signal" strokeWidth={1} />
                  <Line type="monotone" dataKey="background" stroke="#6B7280" name="Background" strokeWidth={1} />
                  <Line type="monotone" dataKey="processed" stroke="#10B981" name="Processed" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detected Flares */}
          {detectedFlares.length > 0 && (
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2" />
                Detected Flare Events ({detectedFlares.length})
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-6">
                {detectedFlares.map((flare) => (
                  <div 
                    key={flare.id}
                    className={`bg-white/5 rounded-xl p-4 border cursor-pointer transition-all duration-200 ${
                      selectedFlare?.id === flare.id 
                        ? 'border-green-400 bg-green-400/10' 
                        : 'border-white/10 hover:border-white/30'
                    }`}
                    onClick={() => setSelectedFlare(flare)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">{flare.id}</span>
                      <span className={`text-sm px-2 py-1 rounded ${getClassificationColor(flare.classification)} bg-white/10`}>
                        {flare.classification}
                      </span>
                    </div>
                    
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Peak Time:</span>
                        <span className="text-white">{new Date(flare.peak_time).toLocaleTimeString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Duration:</span>
                        <span className="text-white">{flare.duration}s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Energy:</span>
                        <span className="text-white">{formatEnergy(flare.energy)} erg</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">α Parameter:</span>
                        <span className={flare.is_nanoflare ? 'text-yellow-400' : 'text-white'}>
                          {flare.alpha_parameter.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="text-white">{(flare.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div className="flex items-center mt-3 space-x-2">
                      {flare.is_nanoflare && (
                        <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded">
                          Nanoflare
                        </span>
                      )}
                      {flare.overlapping && (
                        <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded">
                          Overlapping
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Detailed View */}
              {selectedFlare && (
                <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                  <h4 className="text-white font-semibold mb-4 flex items-center">
                    <Eye className="h-5 w-5 mr-2" />
                    Detailed Analysis: {selectedFlare.id}
                  </h4>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div>
                      <h5 className="text-gray-300 font-medium mb-2">Temporal Properties</h5>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Start:</span>
                          <span className="text-white">{new Date(selectedFlare.start_time).toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Peak:</span>
                          <span className="text-white">{new Date(selectedFlare.peak_time).toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">End:</span>
                          <span className="text-white">{new Date(selectedFlare.end_time).toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Duration:</span>
                          <span className="text-white">{selectedFlare.duration} seconds</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h5 className="text-gray-300 font-medium mb-2">Physical Properties</h5>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Peak Intensity:</span>
                          <span className="text-white">{selectedFlare.intensity.toExponential(2)} W/m²</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Total Energy:</span>
                          <span className="text-white">{formatEnergy(selectedFlare.energy)} erg</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">α Parameter:</span>
                          <span className={selectedFlare.is_nanoflare ? 'text-yellow-400' : 'text-white'}>
                            {selectedFlare.alpha_parameter.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Classification:</span>
                          <span className={getClassificationColor(selectedFlare.classification)}>
                            {selectedFlare.classification}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h5 className="text-gray-300 font-medium mb-2">Detection Quality</h5>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Confidence:</span>
                          <span className="text-white">{(selectedFlare.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Nanoflare:</span>
                          <span className={selectedFlare.is_nanoflare ? 'text-green-400' : 'text-gray-400'}>
                            {selectedFlare.is_nanoflare ? 'Yes' : 'No'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Overlapping:</span>
                          <span className={selectedFlare.overlapping ? 'text-yellow-400' : 'text-gray-400'}>
                            {selectedFlare.overlapping ? 'Yes' : 'No'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
