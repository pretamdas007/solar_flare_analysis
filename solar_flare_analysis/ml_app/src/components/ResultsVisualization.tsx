'use client';

import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Area, AreaChart } from 'recharts';
import { Activity, Zap, TrendingUp, BarChart3, Target, AlertTriangle, CheckCircle } from 'lucide-react';

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
  fallback_data?: AnalysisResults;
}

interface FlareData {
  timestamp: string;
  intensity: number;
  energy: number;
  alpha: number;
  flare_type: string;
  confidence?: number;
  is_nanoflare?: boolean;
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
}

interface Visualizations {
  time_series?: Array<{time: number; intensity: number}>;
  energy_histogram: Array<{energy: number; count: number}>;
  flare_timeline: FlareData[];
  power_law_plot?: Array<{energy: number; cumulative_count: number}>;
}

interface ResultsVisualizationProps {
  results: AnalysisResults;
  onClose: () => void;
}

export default function ResultsVisualization({ results, onClose }: ResultsVisualizationProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'flares' | 'nanoflares' | 'energy' | 'statistics'>('overview');

  if (!results.success) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="text-red-400" size={24} />
          <h3 className="text-xl font-bold text-red-400">Analysis Failed</h3>
        </div>
        <p className="text-red-300 mb-4">
          The analysis could not be completed. Using fallback data for demonstration.
        </p>
        {results.fallback_data && (
          <ResultsVisualization results={results.fallback_data as AnalysisResults} onClose={onClose} />
        )}
      </div>
    );
  }

  const formatEnergy = (energy: number) => {
    if (energy >= 1e30) return `${(energy / 1e30).toFixed(1)}×10³⁰`;
    if (energy >= 1e27) return `${(energy / 1e27).toFixed(1)}×10²⁷`;
    if (energy >= 1e24) return `${(energy / 1e24).toFixed(1)}×10²⁴`;
    return energy.toExponential(2);
  };

  const formatNumber = (num: number) => {
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toFixed(0);
  };

  const StatCard = ({ icon: Icon, title, value, subtitle, color = "blue" }: {
    icon: any;
    title: string;
    value: string | number;
    subtitle?: string;
    color?: string;
  }) => (
    <div className={`bg-gray-800/50 border border-gray-700 rounded-lg p-6`}>
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`text-${color}-400`} size={24} />
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      <p className="text-2xl font-bold text-white mb-1">{value}</p>
      {subtitle && <p className="text-sm text-gray-400">{subtitle}</p>}
    </div>
  );

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Activity}
          title="Total Flares"
          value={results.statistics.total_flares}
          subtitle="Detected events"
          color="blue"
        />
        <StatCard
          icon={Target}
          title="Nanoflares"
          value={results.statistics.nanoflare_count}
          subtitle={`${results.statistics.nanoflare_percentage.toFixed(1)}% of total`}
          color="purple"
        />
        <StatCard
          icon={Zap}
          title="Total Energy"
          value={formatEnergy(results.energy_analysis.total_energy)}
          subtitle="Joules"
          color="yellow"
        />
        <StatCard
          icon={TrendingUp}
          title="Power Law Index"
          value={results.energy_analysis.power_law_index.toFixed(2)}
          subtitle="α parameter"
          color="green"
        />
      </div>

      {/* Quick Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Energy Distribution */}
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Energy Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={results.visualizations.energy_histogram}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="energy" 
                stroke="#9CA3AF"
                tickFormatter={(value) => formatEnergy(value)}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelFormatter={(value) => `Energy: ${formatEnergy(Number(value))} J`}
              />
              <Bar dataKey="count" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Flare Types */}
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Flare Classification</h3>
          <div className="space-y-3">
            {Object.entries(results.statistics.flare_types).map(([type, count]) => (
              <div key={type} className="flex justify-between items-center">
                <span className="text-gray-300 capitalize">{type} Flares</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(count / results.statistics.total_flares) * 100}%` }}
                    />
                  </div>
                  <span className="text-white font-semibold w-8">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Metadata */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Analysis Metadata</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-400">File Processed:</span>
            <p className="text-white font-medium">{results.metadata.file_processed}</p>
          </div>
          <div>
            <span className="text-gray-400">Data Points:</span>
            <p className="text-white font-medium">{formatNumber(results.metadata.data_points)}</p>
          </div>
          <div>
            <span className="text-gray-400">Model Version:</span>
            <p className="text-white font-medium">{results.metadata.model_version}</p>
          </div>
          <div>
            <span className="text-gray-400">Processing Time:</span>
            <p className="text-white font-medium">
              {new Date(results.metadata.processing_time).toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderFlareTimeline = () => (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Flare Timeline</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart data={results.visualizations.flare_timeline}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="timestamp" 
            stroke="#9CA3AF"
            tickFormatter={(value) => new Date(value).toLocaleTimeString()}
          />
          <YAxis 
            dataKey="intensity"
            stroke="#9CA3AF"
            tickFormatter={formatNumber}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
            formatter={(value, name) => [
              name === 'intensity' ? formatNumber(Number(value)) : value,
              name
            ]}
          />
          <Scatter 
            name="Regular Flares" 
            dataKey="intensity" 
            fill="#3B82F6"
            shape="circle"
          />
          <Scatter 
            name="Nanoflares" 
            data={results.nanoflares}
            dataKey="intensity" 
            fill="#8B5CF6"
            shape="diamond"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );

  const renderNanoflareAnalysis = () => (
    <div className="space-y-6">
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Nanoflare Detection Results</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <StatCard
            icon={Target}
            title="Detected Nanoflares"
            value={results.nanoflares.length}
            subtitle={`${((results.nanoflares.length / results.separated_flares.length) * 100).toFixed(1)}%`}
            color="purple"
          />
          <StatCard
            icon={Zap}
            title="Energy Contribution"
            value={`${(results.energy_analysis.nanoflare_energy_fraction * 100).toFixed(1)}%`}
            subtitle="of total energy"
            color="yellow"
          />
          <StatCard
            icon={Activity}
            title="Average α Parameter"
            value={results.nanoflares.length > 0 ? 
              (results.nanoflares.reduce((sum, nf) => sum + Math.abs(nf.alpha), 0) / results.nanoflares.length).toFixed(2) : 
              'N/A'
            }
            subtitle="|α| > 2 for nanoflares"
            color="green"
          />
        </div>
        
        {results.nanoflares.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 text-gray-400">Time</th>
                  <th className="text-left py-2 text-gray-400">Intensity</th>
                  <th className="text-left py-2 text-gray-400">Energy</th>
                  <th className="text-left py-2 text-gray-400">α Parameter</th>
                  <th className="text-left py-2 text-gray-400">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.nanoflares.slice(0, 10).map((nf, idx) => (
                  <tr key={idx} className="border-b border-gray-800">
                    <td className="py-2 text-white">
                      {new Date(nf.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2 text-white">{formatNumber(nf.intensity)}</td>
                    <td className="py-2 text-white">{formatEnergy(nf.energy)}</td>
                    <td className="py-2 text-white">{nf.alpha.toFixed(2)}</td>
                    <td className="py-2 text-white">
                      {nf.confidence ? `${(nf.confidence * 100).toFixed(0)}%` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {results.nanoflares.length > 10 && (
              <p className="text-gray-400 text-sm mt-2">
                Showing first 10 of {results.nanoflares.length} nanoflares
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );

  const renderEnergyAnalysis = () => (
    <div className="space-y-6">
      {/* Power Law Analysis */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Power Law Distribution</h3>
        {results.visualizations.power_law_plot && (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={results.visualizations.power_law_plot}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="energy" 
                stroke="#9CA3AF"
                scale="log"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => formatEnergy(value)}
              />
              <YAxis 
                stroke="#9CA3AF"
                scale="log"
                domain={['dataMin', 'dataMax']}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelFormatter={(value) => `Energy: ${formatEnergy(Number(value))} J`}
              />
              <Line 
                type="monotone" 
                dataKey="cumulative_count" 
                stroke="#8B5CF6" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Energy Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Energy Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Energy:</span>
              <span className="text-white font-semibold">{formatEnergy(results.energy_analysis.total_energy)} J</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Average Energy:</span>
              <span className="text-white font-semibold">{formatEnergy(results.energy_analysis.average_energy)} J</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Median Energy:</span>
              <span className="text-white font-semibold">{formatEnergy(results.energy_analysis.median_energy)} J</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Energy Range:</span>
              <span className="text-white font-semibold">
                {formatEnergy(results.energy_analysis.energy_range[0])} - {formatEnergy(results.energy_analysis.energy_range[1])} J
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Physical Parameters</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Power Law Index (α):</span>
              <span className="text-white font-semibold">{results.energy_analysis.power_law_index.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Nanoflare Energy Fraction:</span>
              <span className="text-white font-semibold">{(results.energy_analysis.nanoflare_energy_fraction * 100).toFixed(1)}%</span>
            </div>
            <div className="pt-2">
              <p className="text-sm text-gray-400">
                <strong>Note:</strong> |α| &gt; 2 indicates sufficient nanoflare density for coronal heating
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-gray-900/95 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Analysis Results</h2>
            <div className="flex items-center gap-2">
              <CheckCircle className="text-green-400" size={20} />
              <span className="text-green-100">Analysis completed successfully</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="flex">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'flares', label: 'Flare Timeline', icon: Activity },
            { id: 'nanoflares', label: 'Nanoflares', icon: Target },
            { id: 'energy', label: 'Energy Analysis', icon: Zap },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-colors ${
                activeTab === id
                  ? 'border-purple-500 text-purple-400 bg-purple-500/10'
                  : 'border-transparent text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'flares' && renderFlareTimeline()}
        {activeTab === 'nanoflares' && renderNanoflareAnalysis()}
        {activeTab === 'energy' && renderEnergyAnalysis()}
      </div>
    </div>
  );
}
