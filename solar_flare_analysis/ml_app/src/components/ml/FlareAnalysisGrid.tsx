'use client';

import React, { useState, useEffect } from 'react';
import { Zap, TrendingUp, BarChart3, PieChart, Calendar, Clock, ArrowUp, ArrowDown, Sun, Flame, Star, Sparkles } from 'lucide-react';

interface FlareStatistics {
  total_events: number;
  class_distribution: {
    A: number;
    B: number;
    C: number;
    M: number;
    X: number;
  };
  energy_analysis: {
    total_energy: number;
    average_energy: number;
    peak_energy: number;
    energy_range: [number, number];
  };
  temporal_analysis: {
    most_active_hour: number;
    most_active_day: string;
    events_per_day: number[];
    peak_frequency: number;
  };
  trends: {
    energy_trend: 'increasing' | 'decreasing' | 'stable';
    frequency_trend: 'increasing' | 'decreasing' | 'stable';
    severity_trend: 'increasing' | 'decreasing' | 'stable';
  };
}

export default function FlareAnalysisGrid() {
  const [statistics, setStatistics] = useState<FlareStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | 'all'>('24h');

  // Mock comprehensive flare analysis data
  const mockStatistics: FlareStatistics = {
    total_events: 1247,
    class_distribution: {
      A: 892,
      B: 234,
      C: 89,
      M: 28,
      X: 4
    },
    energy_analysis: {
      total_energy: 2.47e29,
      average_energy: 1.98e26,
      peak_energy: 3.45e28,
      energy_range: [1.2e24, 3.45e28]
    },
    temporal_analysis: {
      most_active_hour: 14,
      most_active_day: 'Tuesday',
      events_per_day: [12, 18, 23, 15, 8, 19, 14],
      peak_frequency: 0.34
    },
    trends: {
      energy_trend: 'increasing',
      frequency_trend: 'stable',
      severity_trend: 'increasing'
    }
  };

  useEffect(() => {
    // Simulate data loading with time range dependency
    setLoading(true);
    setTimeout(() => {
      setStatistics(mockStatistics);
      setLoading(false);
    }, 800);
  }, [timeRange]);
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return <ArrowUp className="h-4 w-4 text-red-400 stellar-icon" />;
      case 'decreasing': return <ArrowDown className="h-4 w-4 text-green-400 stellar-icon" />;
      default: return <TrendingUp className="h-4 w-4 text-yellow-400 stellar-icon" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'increasing': return 'status-offline';
      case 'decreasing': return 'status-online';
      default: return 'status-warning';
    }
  };

  const getClassColor = (flareClass: string) => {
    switch (flareClass) {
      case 'X': return 'bg-red-500';
      case 'M': return 'bg-orange-500';
      case 'C': return 'bg-yellow-500';
      case 'B': return 'bg-green-500';
      default: return 'bg-blue-500';
    }
  };

  const getPercentage = (value: number, total: number) => {
    return ((value / total) * 100).toFixed(1);
  };
  if (loading) {
    return (
      <div className="solar-glassmorphism p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-white/20 rounded w-1/2"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-white/10 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!statistics) return null;

  return (
    <div className="solar-glassmorphism p-6 mb-8">
      {/* Cosmic Header with Time Range Selector */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-3">
          <Flame className="h-6 w-6 text-orange-400 stellar-icon" />
          <h2 className="aurora-text text-xl font-semibold">Solar Flare Analysis Dashboard</h2>
          <Star className="h-5 w-5 text-blue-400 stellar-icon" />
        </div>
        <div className="flex space-x-2">
          {(['24h', '7d', '30d', 'all'] as const).map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 rounded-lg text-sm transition-all duration-300 ${
                timeRange === range
                  ? 'solar-button'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20 border border-white/20'
              }`}
            >
              {range === 'all' ? 'All Time' : range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>      {/* Solar Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="solar-card border-aurora-blue bg-gradient-to-br from-aurora-blue/20 to-aurora-blue/5 hover:from-aurora-blue/30 hover:to-aurora-blue/10 transition-all duration-300">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="h-5 w-5 text-aurora-blue stellar-icon animate-solarPulse" />
            <span className="text-aurora-blue font-medium">Total Events</span>
          </div>
          <span className="aurora-text text-2xl font-bold">{statistics.total_events.toLocaleString()}</span>
        </div>

        <div className="solar-card border-solar-gold bg-gradient-to-br from-solar-gold/20 to-solar-gold/5 hover:from-solar-gold/30 hover:to-solar-gold/10 transition-all duration-300">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-5 w-5 text-solar-gold stellar-icon animate-stellarRotation" />
            <span className="text-solar-gold font-medium">Total Energy</span>
          </div>
          <span className="aurora-text text-lg font-bold">{statistics.energy_analysis.total_energy.toExponential(2)} J</span>
        </div>

        <div className="solar-card border-plasma-violet bg-gradient-to-br from-plasma-violet/20 to-plasma-violet/5 hover:from-plasma-violet/30 hover:to-plasma-violet/10 transition-all duration-300">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart3 className="h-5 w-5 text-plasma-violet stellar-icon animate-plasmaFlow" />
            <span className="text-plasma-violet font-medium">Peak Energy</span>
          </div>
          <span className="aurora-text text-lg font-bold">{statistics.energy_analysis.peak_energy.toExponential(2)} J</span>
        </div>

        <div className="solar-card border-coronal-orange bg-gradient-to-br from-coronal-orange/20 to-coronal-orange/5 hover:from-coronal-orange/30 hover:to-coronal-orange/10 transition-all duration-300">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="h-5 w-5 text-coronal-orange stellar-icon animate-coronalWave" />
            <span className="text-coronal-orange font-medium">Peak Hour</span>
          </div>
          <span className="aurora-text text-xl font-bold">{statistics.temporal_analysis.most_active_hour}:00 UTC</span>
        </div>
      </div>      {/* Cosmic Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Flare Class Distribution */}
        <div className="solar-card bg-gradient-to-br from-deep-space-black to-plasma-violet/10 border-plasma-violet/50">
          <h3 className="aurora-text text-lg font-medium mb-4 flex items-center space-x-2">
            <PieChart className="h-5 w-5 stellar-icon animate-stellarRotation" />
            <span>Flare Class Distribution</span>
            <Sparkles className="h-4 w-4 text-plasma-violet stellar-icon animate-auroraShimmer" />
          </h3>
          <div className="space-y-3">
            {Object.entries(statistics.class_distribution).map(([flareClass, count]) => (
              <div key={flareClass} className="flex items-center justify-between p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center space-x-3">
                  <div className={`w-4 h-4 rounded ${getClassColor(flareClass)} animate-solarPulse`}></div>
                  <span className="aurora-text font-medium">Class {flareClass}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="aurora-text font-bold">{count}</span>
                  <span className="text-gray-400 text-sm">
                    ({getPercentage(count, statistics.total_events)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
          
          {/* Cosmic Bar Chart */}
          <div className="mt-4 space-y-2">
            {Object.entries(statistics.class_distribution).map(([flareClass, count]) => (
              <div key={flareClass} className="flex items-center space-x-2">
                <span className="aurora-text text-sm w-8">Class {flareClass}</span>
                <div className="flex-1 bg-deep-space-black/60 rounded-full h-3 border border-white/20">
                  <div
                    className={`h-3 rounded-full ${getClassColor(flareClass)} transition-all duration-1000 ease-out animate-plasmaFlow`}
                    style={{ width: `${getPercentage(count, statistics.total_events)}%` }}
                  ></div>
                </div>
                <span className="text-gray-400 text-xs w-12">{getPercentage(count, statistics.total_events)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Energy Analysis */}
        <div className="solar-card bg-gradient-to-br from-deep-space-black to-solar-gold/10 border-solar-gold/50">
          <h3 className="aurora-text text-lg font-medium mb-4 flex items-center space-x-2">
            <Zap className="h-5 w-5 stellar-icon animate-flareEruption" />
            <span>Energy Analysis</span>
            <Sun className="h-4 w-4 text-solar-gold stellar-icon animate-stellarRotation" />
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="solar-glassmorphism p-3 bg-gradient-to-br from-solar-gold/20 to-transparent border-solar-gold/30">
                <span className="text-solar-gold text-sm">Average Energy</span>
                <div className="aurora-text font-bold">{statistics.energy_analysis.average_energy.toExponential(2)} J</div>
              </div>
              <div className="solar-glassmorphism p-3 bg-gradient-to-br from-coronal-orange/20 to-transparent border-coronal-orange/30">
                <span className="text-coronal-orange text-sm">Energy Range</span>
                <div className="aurora-text font-bold text-xs">
                  {statistics.energy_analysis.energy_range[0].toExponential(1)} - {statistics.energy_analysis.energy_range[1].toExponential(1)} J
                </div>
              </div>
            </div>

            {/* Cosmic Energy Distribution Visualization */}
            <div className="space-y-2">
              <span className="text-solar-gold text-sm flex items-center space-x-2">
                <Star className="h-4 w-4 stellar-icon animate-auroraShimmer" />
                <span>Energy Distribution by Class</span>
              </span>
              {Object.entries(statistics.class_distribution).reverse().map(([flareClass, count]) => {
                const energyContribution = count * Math.pow(10, ['X', 'M', 'C', 'B', 'A'].indexOf(flareClass) * 2 + 24);
                const percentage = (energyContribution / statistics.energy_analysis.total_energy) * 100;
                return (
                  <div key={flareClass} className="flex items-center space-x-2">
                    <span className="aurora-text text-sm w-8">Class {flareClass}</span>
                    <div className="flex-1 bg-deep-space-black/60 rounded-full h-3 border border-white/20">
                      <div
                        className={`h-3 rounded-full ${getClassColor(flareClass)} transition-all duration-1000 ease-out animate-coronalWave`}
                        style={{ width: `${Math.min(percentage, 100)}%` }}
                      ></div>
                    </div>
                    <span className="text-gray-400 text-xs w-12">{percentage.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Temporal Analysis */}
        <div className="solar-card bg-gradient-to-br from-deep-space-black to-aurora-blue/10 border-aurora-blue/50">
          <h3 className="aurora-text text-lg font-medium mb-4 flex items-center space-x-2">
            <Calendar className="h-5 w-5 stellar-icon animate-coronalWave" />
            <span>Temporal Patterns</span>
            <Clock className="h-4 w-4 text-aurora-blue stellar-icon animate-stellarRotation" />
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="solar-glassmorphism p-3 bg-gradient-to-br from-aurora-blue/20 to-transparent border-aurora-blue/30">
                <span className="text-aurora-blue text-sm">Most Active Day</span>
                <div className="aurora-text font-bold">{statistics.temporal_analysis.most_active_day}</div>
              </div>
              <div className="solar-glassmorphism p-3 bg-gradient-to-br from-plasma-violet/20 to-transparent border-plasma-violet/30">
                <span className="text-plasma-violet text-sm">Peak Frequency</span>
                <div className="aurora-text font-bold">{statistics.temporal_analysis.peak_frequency.toFixed(2)} events/hour</div>
              </div>
            </div>

            {/* Cosmic Weekly Activity Chart */}
            <div className="space-y-2">
              <span className="text-aurora-blue text-sm flex items-center space-x-2">
                <BarChart3 className="h-4 w-4 stellar-icon animate-solarPulse" />
                <span>Weekly Activity Pattern</span>
              </span>
              <div className="flex items-end space-x-1 h-20 p-3 bg-gradient-to-t from-deep-space-black/50 to-transparent rounded-lg border border-white/10">
                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, index) => {
                  const events = statistics.temporal_analysis.events_per_day[index];
                  const height = (events / Math.max(...statistics.temporal_analysis.events_per_day)) * 100;
                  return (
                    <div key={day} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full bg-gradient-to-t from-aurora-blue to-plasma-violet rounded-t animate-plasmaFlow transition-all duration-1000"
                        style={{ height: `${height}%` }}
                      ></div>
                      <span className="text-gray-400 text-xs mt-1">{day}</span>
                      <span className="aurora-text text-xs font-semibold">{events}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Trend Analysis */}
        <div className="solar-card bg-gradient-to-br from-deep-space-black to-coronal-orange/10 border-coronal-orange/50">
          <h3 className="aurora-text text-lg font-medium mb-4 flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 stellar-icon animate-plasmaFlow" />
            <span>Trend Analysis</span>
            <Flame className="h-4 w-4 text-coronal-orange stellar-icon animate-flareEruption" />
          </h3>
          <div className="space-y-4">
            {Object.entries(statistics.trends).map(([category, trend]) => (
              <div key={category} className="flex items-center justify-between p-3 solar-glassmorphism bg-gradient-to-r from-white/10 to-white/5 border border-white/20 hover:from-white/20 hover:to-white/10 transition-all duration-300">
                <span className="aurora-text capitalize font-medium">{category.replace('_', ' ')}</span>
                <div className="flex items-center space-x-2">
                  {getTrendIcon(trend)}
                  <span className={`font-medium capitalize ${getTrendColor(trend)} animate-solarPulse`}>
                    {trend}
                  </span>
                </div>
              </div>
            ))}
          </div>
          
          {/* Cosmic Trend Insights */}
          <div className="mt-4 p-4 solar-glassmorphism bg-gradient-to-br from-solar-gold/20 to-coronal-orange/20 border border-solar-gold/30">
            <h4 className="text-solar-gold font-medium mb-2 flex items-center space-x-2">
              <Sparkles className="h-4 w-4 stellar-icon animate-auroraShimmer" />
              <span>Key Insights</span>
            </h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li className="flex items-center space-x-2">
                <Star className="h-3 w-3 text-solar-gold stellar-icon" />
                <span>Solar activity showing increased energy output</span>
              </li>
              <li className="flex items-center space-x-2">
                <Star className="h-3 w-3 text-aurora-blue stellar-icon" />
                <span>Flare frequency remains stable over time period</span>
              </li>
              <li className="flex items-center space-x-2">
                <Star className="h-3 w-3 text-plasma-violet stellar-icon" />
                <span>Higher severity events becoming more common</span>
              </li>
              <li className="flex items-center space-x-2">
                <Star className="h-3 w-3 text-coronal-orange stellar-icon" />
                <span>Tuesday shows peak activity consistently</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
