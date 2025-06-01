'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Activity, Zap, TrendingUp, Play, Pause, RefreshCw, Sun, Flame, Star } from 'lucide-react';

interface DataPoint {
  timestamp: string;
  short_channel: number;
  long_channel: number;
  flare_detected: boolean;
  confidence: number;
}

interface RealTimeData {
  data_points: DataPoint[];
  current_flare_class: string;
  peak_flux: number;
  background_level: number;
  analysis_status: string;
}

export default function RealTimeVisualization() {
  const [data, setData] = useState<RealTimeData | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Simulate real-time data streaming
  const startStreaming = () => {
    setIsStreaming(true);
    setError(null);

    intervalRef.current = setInterval(async () => {
      try {
        // Generate mock real-time data
        const currentTime = new Date();
        const newDataPoint: DataPoint = {
          timestamp: currentTime.toISOString(),
          short_channel: Math.random() * 1e-6 + 1e-8,
          long_channel: Math.random() * 1e-7 + 1e-9,
          flare_detected: Math.random() > 0.85,
          confidence: Math.random() * 0.3 + 0.7
        };

        setData(prevData => {
          const currentData = prevData || {
            data_points: [],
            current_flare_class: 'A',
            peak_flux: 0,
            background_level: 1e-9,
            analysis_status: 'monitoring'
          };

          // Keep only last 100 data points for performance
          const updatedPoints = [...currentData.data_points, newDataPoint].slice(-100);
          
          return {
            ...currentData,
            data_points: updatedPoints,
            current_flare_class: newDataPoint.flare_detected ? 
              ['A', 'B', 'C', 'M', 'X'][Math.floor(Math.random() * 5)] : 'A',
            peak_flux: Math.max(currentData.peak_flux, newDataPoint.short_channel),
            analysis_status: newDataPoint.flare_detected ? 'flare_detected' : 'monitoring'
          };
        });
      } catch (error) {
        console.error('Error in streaming:', error);
        setError('Failed to fetch real-time data');
      }
    }, 1000); // Update every second
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Draw real-time chart
  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const points = data.data_points;
    if (points.length < 2) return;

    // Set up canvas dimensions
    const padding = 40;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;

    // Find data ranges
    const maxShort = Math.max(...points.map(p => p.short_channel));
    const minShort = Math.min(...points.map(p => p.short_channel));
    const maxLong = Math.max(...points.map(p => p.long_channel));
    const minLong = Math.min(...points.map(p => p.long_channel));

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = padding + (i * chartHeight) / 10;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }

    // Draw short channel data
    ctx.strokeStyle = '#3B82F6'; // Blue
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = padding + (index * chartWidth) / (points.length - 1);
      const y = padding + chartHeight - ((point.short_channel - minShort) / (maxShort - minShort)) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw long channel data
    ctx.strokeStyle = '#EF4444'; // Red
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = padding + (index * chartWidth) / (points.length - 1);
      const y = padding + chartHeight - ((point.long_channel - minLong) / (maxLong - minLong)) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Highlight flare detections
    ctx.fillStyle = '#F59E0B'; // Yellow
    points.forEach((point, index) => {
      if (point.flare_detected) {
        const x = padding + (index * chartWidth) / (points.length - 1);
        ctx.beginPath();
        ctx.arc(x, padding + chartHeight / 2, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw labels
    ctx.fillStyle = '#D1D5DB';
    ctx.font = '12px Arial';
    ctx.fillText('Short Channel (0.5-4.0 Å)', padding, 20);
    ctx.fillStyle = '#EF4444';
    ctx.fillText('Long Channel (1.0-8.0 Å)', padding + 200, 20);
    ctx.fillStyle = '#F59E0B';
    ctx.fillText('● Flare Detection', padding + 400, 20);

  }, [data]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
  const getFlareClassColor = (flareClass: string) => {
    switch (flareClass) {
      case 'X': return 'status-offline';
      case 'M': return 'text-orange-500';
      case 'C': return 'status-warning';
      case 'B': return 'status-online';
      default: return 'text-blue-500';
    }
  };
  return (
    <div className="solar-glassmorphism p-6 mb-8">
      {/* Cosmic Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-3">
          <Sun className="h-6 w-6 text-yellow-400 stellar-icon" />
          <h2 className="aurora-text text-xl font-semibold">Real-time GOES XRS Solar Monitoring</h2>
          <Star className="h-5 w-5 text-blue-400 stellar-icon" />
        </div>
        <div className="flex space-x-3">
          {!isStreaming ? (
            <button
              onClick={startStreaming}
              className="solar-button flex items-center space-x-2"
            >
              <Play className="h-4 w-4" />
              <span>Start Stream</span>
            </button>
          ) : (
            <button
              onClick={stopStreaming}
              className="plasma-button flex items-center space-x-2"
            >
              <Pause className="h-4 w-4" />
              <span>Stop Stream</span>
            </button>
          )}
        </div>
      </div>      {/* Solar Status Indicators */}
      {data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="solar-card">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="h-5 w-5 text-blue-400 stellar-icon" />
              <span className="text-white font-medium">Status</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${data.analysis_status === 'flare_detected' ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
              <span className={`text-sm font-semibold ${data.analysis_status === 'flare_detected' ? 'status-offline' : 'status-online'}`}>
                {data.analysis_status.replace('_', ' ').toUpperCase()}
              </span>
            </div>
          </div>

          <div className="solar-card">
            <div className="flex items-center space-x-2 mb-2">
              <Flame className="h-5 w-5 text-orange-400 stellar-icon" />
              <span className="text-white font-medium">Flare Class</span>
            </div>
            <span className={`text-lg font-bold ${getFlareClassColor(data.current_flare_class)}`}>
              {data.current_flare_class}
            </span>
          </div>

          <div className="solar-card">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="h-5 w-5 text-green-400 stellar-icon" />
              <span className="text-white font-medium">Peak Flux</span>
            </div>
            <span className="text-white text-sm font-mono bg-white/10 px-2 py-1 rounded">{data.peak_flux.toExponential(2)}</span>
          </div>

          <div className="solar-card">
            <div className="flex items-center space-x-2 mb-2">
              <RefreshCw className={`h-5 w-5 ${isStreaming ? 'animate-spin status-online' : 'text-gray-400'}`} />
              <span className="text-white font-medium">Stream</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
              <span className={`text-sm font-semibold ${isStreaming ? 'status-online' : 'text-gray-400'}`}>
                {isStreaming ? 'LIVE' : 'STOPPED'}
              </span>
            </div>
          </div>
        </div>
      )}      {/* Cosmic Real-time Chart */}
      <div className="solar-card bg-gradient-to-br from-gray-900 to-black border-2 border-white/20">
        <div className="mb-3">
          <h3 className="aurora-text text-lg font-semibold">Solar X-ray Flux Monitor</h3>
        </div>
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          className="w-full h-64 rounded-xl border border-white/30 bg-gradient-to-br from-gray-900 via-black to-gray-800"
        />
      </div>

      {/* Cosmic Data Summary */}
      {data && (
        <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="solar-card">
            <span className="text-gray-300 text-sm block mb-1">Data Points</span>
            <div className="flex items-center space-x-2">
              <div className="text-white font-bold text-lg">{data.data_points.length}</div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-400 to-purple-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(data.data_points.length, 100)}%` }}
                />
              </div>
            </div>
          </div>
          <div className="solar-card">
            <span className="text-gray-300 text-sm block mb-1">Background Level</span>
            <div className="text-white font-mono text-sm bg-white/10 px-2 py-1 rounded">{data.background_level.toExponential(2)}</div>
          </div>
          <div className="solar-card">
            <span className="text-gray-300 text-sm block mb-1">Flares Detected</span>
            <div className="flex items-center space-x-2">
              <Flame className="h-4 w-4 text-orange-400 stellar-icon" />
              <div className="text-white font-bold text-lg">
                {data.data_points.filter(p => p.flare_detected).length}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 solar-glassmorphism border-l-4 border-red-500 p-4">
          <span className="text-red-400">{error}</span>
        </div>
      )}
    </div>
  );
}
