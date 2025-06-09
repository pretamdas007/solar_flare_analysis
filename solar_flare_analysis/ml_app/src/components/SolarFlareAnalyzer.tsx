'use client';

import { useState, useEffect } from 'react';
import { Upload, Play, Download, Settings, Zap, TrendingUp, Activity, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import NotificationSystem, { useNotifications } from './NotificationSystem';
import ResultsVisualization from './ResultsVisualization';

interface AnalysisResults {
  success: boolean;
  separated_flares: FlareData[];
  nanoflares: FlareData[];
  energy_analysis: EnergyAnalysis;
  statistics: FlareStatistics;
  visualizations: Visualizations;
  metadata: {
    file_processed: string;
    processing_time: string;
    data_points: number;
    model_version: string;
  };
  error?: string;
  fallback_data?: AnalysisResults;
}

interface FlareData {
  timestamp: string;
  intensity: number;
  energy: number;
  alpha: number;
  flare_type: 'nano' | 'micro' | 'minor' | 'major' | 'X-class';
  confidence?: number;
  is_nanoflare?: boolean;
  peak_time?: number;
  rise_time?: number;
  decay_time?: number;
  background?: number;
}

interface EnergyAnalysis {
  total_energy: number;
  average_energy: number;
  median_energy: number;
  power_law_index: number;
  nanoflare_energy_fraction: number;
  energy_range: [number, number];
}

interface FlareStatistics {
  total_flares: number;
  nanoflare_count: number;
  nanoflare_percentage: number;
  average_energy: number;
  power_law_index: number;
  flare_types: Record<string, number>;
  temporal_distribution?: {
    hourly_distribution: Record<number, number>;
    peak_activity_hour: number;
  };
}

interface Visualizations {
  time_series?: Array<{time: number; intensity: number}>;
  energy_histogram: Array<{energy: number; count: number}>;
  flare_timeline: FlareData[];
  power_law_plot?: Array<{energy: number; cumulative_count: number}>;
}

export default function SolarFlareAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [apiStatus, setApiStatus] = useState<'checking' | 'available' | 'unavailable'>('checking');
  const { notifications, addNotification, removeNotification } = useNotifications();

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const health = await response.json();
        setApiStatus('available');
        if (!health.ml_available) {
          addNotification({
            type: 'warning',
            title: 'Mock Mode',
            message: 'ML models unavailable, using demonstration data',
            duration: 5000
          });
        }
      } else {
        setApiStatus('unavailable');
      }
    } catch (error) {
      setApiStatus('unavailable');
      addNotification({
        type: 'error',
        title: 'API Unavailable',
        message: 'Backend API is not responding',
        duration: 5000
      });
    }
  };

  // Handle file selection
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      addNotification({
        type: 'info',
        title: 'File Selected',
        message: `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`,
        duration: 3000
      });
    }
  };

  // Run ML analysis using the API
  const runAnalysis = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setAnalysisProgress(0);

    try {
      addNotification({
        type: 'info',
        title: 'Analysis Started',
        message: `Processing ${selectedFile.name} with ML model...`,
        duration: 3000
      });

      const formData = new FormData();
      formData.append('file', selectedFile);

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setAnalysisProgress((prev: number) => Math.min(prev + 10, 90));
      }, 500);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setAnalysisProgress(100);

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }

      const analysisResults = await response.json();
      setResults(analysisResults);

      addNotification({
        type: 'success',
        title: 'Analysis Complete',
        message: `Found ${analysisResults.statistics?.total_flares || 0} flares, ${analysisResults.statistics?.nanoflare_count || 0} nanoflares`,
        duration: 5000
      });

    } catch (error) {
      console.error('Analysis failed:', error);
      addNotification({
        type: 'error',
        title: 'Analysis Failed',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        duration: 5000
      });
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => setAnalysisProgress(0), 2000);
    }
  };

  // Export results as JSON
  const exportResults = () => {
    if (!results) return;

    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `solar_flare_analysis_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    addNotification({
      type: 'success',
      title: 'Export Complete',
      message: 'Analysis results exported successfully',
      duration: 3000
    });
  };

  const StatusIndicator = ({ status, children }: { status: 'checking' | 'available' | 'unavailable', children: React.ReactNode }) => {
    const colors = {
      checking: 'text-yellow-400',
      available: 'text-green-400',
      unavailable: 'text-red-400'
    };
    
    const icons = {
      checking: Activity,
      available: CheckCircle,
      unavailable: AlertCircle
    };
    
    const Icon = icons[status];
    
    return (
      <div className={`flex items-center gap-2 ${colors[status]}`}>
        <Icon size={16} />
        {children}
      </div>
    );
  };

  const formatFileSize = (bytes: number) => {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} bytes`;
  };

  return (
    <div className="space-y-8">
      <NotificationSystem
        notifications={notifications}
        removeNotification={removeNotification}
      />

      {/* Control Panel */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          
          {/* File Upload Section */}
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Upload GOES/EXIS Data File
            </label>
            <div className="flex items-center gap-4">
              <div className="relative">
                <input
                  type="file"
                  accept=".nc,.h5,.hdf5,.fits,.csv,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer transition-colors"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Choose File
                </label>
              </div>
              
              {selectedFile && (
                <div className="flex items-center gap-2">
                  <FileText className="text-green-400" size={20} />
                  <div>
                    <p className="text-white font-medium">{selectedFile.name}</p>
                    <p className="text-gray-400 text-sm">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>
              )}
            </div>
            
            <p className="text-gray-400 text-sm mt-2">
              Supported formats: NetCDF (.nc), HDF5 (.h5, .hdf5), FITS (.fits), CSV (.csv), Text (.txt)
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col gap-3">
            <button
              onClick={runAnalysis}
              disabled={!selectedFile || isAnalyzing || apiStatus === 'unavailable'}
              className="flex items-center justify-center px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors font-medium"
            >
              <Play className="w-5 h-5 mr-2" />
              {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
            </button>
            
            {results && (
              <button
                onClick={exportResults}
                className="flex items-center justify-center px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors font-medium"
              >
                <Download className="w-5 h-5 mr-2" />
                Export Results
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {isAnalyzing && (
          <div className="mt-6">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Processing...</span>
              <span className="text-gray-300">{analysisProgress}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${analysisProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* API Status */}
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">API Status:</span>
            <StatusIndicator status={apiStatus}>
              {apiStatus === 'checking' && 'Checking...'}
              {apiStatus === 'available' && 'Connected'}
              {apiStatus === 'unavailable' && 'Disconnected'}
            </StatusIndicator>
          </div>
        </div>
      </div>

      {/* Analysis Results */}
      {results && (
        <ResultsVisualization 
          results={results} 
          onClose={() => setResults(null)}
        />
      )}

      {/* Instructions */}
      {!results && (
        <div className="bg-gray-800/30 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">How to Use</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold">
                1
              </div>
              <div>
                <p className="font-medium text-white mb-1">Upload Data</p>
                <p className="text-gray-400">Select a GOES/EXIS data file in supported format</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                2
              </div>
              <div>
                <p className="font-medium text-white mb-1">Run Analysis</p>
                <p className="text-gray-400">Execute ML model for flare detection and separation</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 bg-green-600 rounded-full flex items-center justify-center text-white font-bold">
                3
              </div>
              <div>
                <p className="font-medium text-white mb-1">View Results</p>
                <p className="text-gray-400">Explore visualizations and export findings</p>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-900/20 border border-blue-700 rounded-lg">
            <div className="flex items-start gap-2">
              <Zap className="text-blue-400 mt-0.5" size={16} />
              <div>
                <p className="text-blue-300 font-medium">Nanoflare Detection</p>
                <p className="text-blue-200 text-sm mt-1">
                  The system automatically identifies nanoflares using |α| &gt; 2 criteria, 
                  which indicates sufficient energy density for coronal heating.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
