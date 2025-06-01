'use client';

import { useState, useEffect } from 'react';
import { Zap, Play, Settings, Calculator, BarChart3, Activity, TrendingUp, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { useNotification } from '@/components/NotificationSystem';

interface EnergySettings {
  integration_method: string;
  wavelength_band: string;
  energy_conversion_factor: number;
  temperature_model: string;
  emission_measure_correction: boolean;
  plasma_physics_model: string;
  geometric_correction: number;
}

interface FlareEnergyResult {
  flare_id: string;
  peak_time: string;
  duration: number;
  peak_flux: number;
  integrated_flux: number;
  thermal_energy: number;
  radiated_energy: number;
  total_energy: number;
  energy_class: string;
  temperature: number;
  emission_measure: number;
  confidence: number;
}

export default function EnergyCalculation() {
  const [isCalculating, setIsCalculating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [energyResults, setEnergyResults] = useState<FlareEnergyResult[]>([]);
  const [selectedFlare, setSelectedFlare] = useState<FlareEnergyResult | null>(null);
  const [settings, setSettings] = useState<EnergySettings>({
    integration_method: 'trapezoidal',
    wavelength_band: 'goes_1_8A',
    energy_conversion_factor: 1.0,
    temperature_model: 'isothermal',
    emission_measure_correction: true,
    plasma_physics_model: 'chianti',
    geometric_correction: 1.5
  });
  const { showNotification } = useNotification();

  const [energyDistribution, setEnergyDistribution] = useState<any[]>([]);
  const [temperatureEvolution, setTemperatureEvolution] = useState<any[]>([]);

  useEffect(() => {
    // Generate mock energy distribution data
    const distribution = [];
    for (let i = 20; i <= 30; i += 0.2) {
      const energy = Math.pow(10, i);
      const count = Math.max(1, 1000 * Math.pow(energy, -1.8) * Math.exp(Math.random() * 2 - 1));
      distribution.push({
        log_energy: i,
        energy: energy,
        count: count,
        cumulative: count * 1.5
      });
    }
    setEnergyDistribution(distribution);
  }, []);

  const startCalculation = async () => {
    setIsCalculating(true);
    setProgress(0);
    setEnergyResults([]);
    showNotification('Starting energy calculation...', 'info');

    const steps = [
      'Loading flare event data...',
      'Applying wavelength corrections...',
      'Integrating X-ray flux...',
      'Calculating plasma parameters...',
      'Estimating thermal energy...',
      'Computing radiated energy losses...',
      'Applying geometric corrections...',
      'Finalizing energy estimates...'
    ];

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1200));
      setProgress(((i + 1) / steps.length) * 100);
      showNotification(steps[i], 'info');
    }

    // Generate mock energy calculation results
    const mockResults: FlareEnergyResult[] = [
      {
        flare_id: 'flare_001',
        peak_time: '2024-01-01T01:52:30Z',
        duration: 1500,
        peak_flux: 5.2e-6,
        integrated_flux: 2.1e-3,
        thermal_energy: 2.3e25,
        radiated_energy: 4.7e24,
        total_energy: 2.8e25,
        energy_class: 'C-class',
        temperature: 1.2e7,
        emission_measure: 3.4e48,
        confidence: 0.92
      },
      {
        flare_id: 'flare_002',
        peak_time: '2024-01-01T03:25:15Z',
        duration: 600,
        peak_flux: 1.8e-6,
        integrated_flux: 5.4e-4,
        thermal_energy: 8.7e24,
        radiated_energy: 1.9e24,
        total_energy: 1.1e25,
        energy_class: 'B-class',
        temperature: 8.5e6,
        emission_measure: 1.2e48,
        confidence: 0.85
      },
      {
        flare_id: 'flare_003',
        peak_time: '2024-01-01T06:02:45Z',
        duration: 2400,
        peak_flux: 8.9e-6,
        integrated_flux: 8.7e-3,
        thermal_energy: 5.1e25,
        radiated_energy: 1.2e25,
        total_energy: 6.3e25,
        energy_class: 'M-class',
        temperature: 1.8e7,
        emission_measure: 7.8e48,
        confidence: 0.96
      },
      {
        flare_id: 'flare_004',
        peak_time: '2024-01-01T09:15:22Z',
        duration: 180,
        peak_flux: 3.2e-7,
        integrated_flux: 1.1e-4,
        thermal_energy: 1.5e24,
        radiated_energy: 3.8e23,
        total_energy: 1.9e24,
        energy_class: 'A-class',
        temperature: 6.2e6,
        emission_measure: 4.5e47,
        confidence: 0.78
      }
    ];

    setEnergyResults(mockResults);
    setSelectedFlare(mockResults[0]);

    // Generate temperature evolution for selected flare
    const tempEvolution = [];
    for (let t = 0; t <= mockResults[0].duration; t += 60) {
      const phase = t / mockResults[0].duration;
      let temp;
      if (phase < 0.3) {
        // Rise phase
        temp = 5e6 + (1.2e7 - 5e6) * (phase / 0.3);
      } else if (phase < 0.7) {
        // Peak phase
        temp = 1.2e7 * (1 + 0.1 * Math.sin(phase * 10));
      } else {
        // Decay phase
        temp = 1.2e7 * Math.exp(-(phase - 0.7) * 3);
      }
      
      tempEvolution.push({
        time: t,
        temperature: temp,
        energy_rate: temp * 1e-13 // Simplified energy rate
      });
    }
    setTemperatureEvolution(tempEvolution);

    setIsCalculating(false);
    showNotification(`Energy calculation completed! Analyzed ${mockResults.length} flare events.`, 'success');
  };

  const handleSettingsChange = (key: keyof EnergySettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const formatEnergy = (energy: number): string => {
    const exponent = Math.floor(Math.log10(energy));
    const mantissa = energy / Math.pow(10, exponent);
    return `${mantissa.toFixed(1)} × 10^${exponent}`;
  };

  const formatTemperature = (temp: number): string => {
    return `${(temp / 1e6).toFixed(1)} MK`;
  };

  const getEnergyClassColor = (energyClass: string) => {
    switch (energyClass) {
      case 'X-class': return 'text-red-400';
      case 'M-class': return 'text-orange-400';
      case 'C-class': return 'text-yellow-400';
      case 'B-class': return 'text-green-400';
      case 'A-class': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  const integrationMethods = [
    { value: 'trapezoidal', label: 'Trapezoidal Rule', description: 'Standard numerical integration' },
    { value: 'simpson', label: 'Simpson\'s Rule', description: 'Higher accuracy integration' },
    { value: 'gaussian', label: 'Gaussian Quadrature', description: 'Adaptive integration' },
    { value: 'monte_carlo', label: 'Monte Carlo', description: 'Statistical integration' }
  ];

  const wavelengthBands = [
    { value: 'goes_1_8A', label: 'GOES 1-8 Å', description: 'Standard GOES band' },
    { value: 'goes_0_5_4A', label: 'GOES 0.5-4 Å', description: 'Higher energy band' },
    { value: 'fermi_12_25keV', label: 'Fermi 12-25 keV', description: 'Hard X-ray band' },
    { value: 'rhessi_full', label: 'RHESSI Full Range', description: 'Full spectral range' }
  ];

  const temperatureModels = [
    { value: 'isothermal', label: 'Isothermal', description: 'Single temperature' },
    { value: 'multithermal', label: 'Multi-thermal', description: 'Temperature distribution' },
    { value: 'dem', label: 'DEM Analysis', description: 'Differential emission measure' }
  ];

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
          <Zap className="h-8 w-8 mr-3 text-red-400" />
          Calculate Flare Energy
        </h1>
        <p className="text-gray-300">
          Estimate thermal and radiated energy of solar flare events using plasma physics models and X-ray observations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Settings Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Settings className="h-6 w-6 mr-2" />
              Calculation Settings
            </h2>
            
            <div className="space-y-4">
              {/* Integration Method */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Integration Method
                </label>
                <select
                  value={settings.integration_method}
                  onChange={(e) => handleSettingsChange('integration_method', e.target.value)}
                  disabled={isCalculating}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  {integrationMethods.map(method => (
                    <option key={method.value} value={method.value} className="bg-gray-800">
                      {method.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Wavelength Band */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Wavelength Band
                </label>
                <select
                  value={settings.wavelength_band}
                  onChange={(e) => handleSettingsChange('wavelength_band', e.target.value)}
                  disabled={isCalculating}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  {wavelengthBands.map(band => (
                    <option key={band.value} value={band.value} className="bg-gray-800">
                      {band.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Temperature Model */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Temperature Model
                </label>
                <select
                  value={settings.temperature_model}
                  onChange={(e) => handleSettingsChange('temperature_model', e.target.value)}
                  disabled={isCalculating}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  {temperatureModels.map(model => (
                    <option key={model.value} value={model.value} className="bg-gray-800">
                      {model.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Energy Conversion Factor */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Conversion Factor: {settings.energy_conversion_factor}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={settings.energy_conversion_factor}
                  onChange={(e) => handleSettingsChange('energy_conversion_factor', parseFloat(e.target.value))}
                  disabled={isCalculating}
                  className="w-full"
                />
              </div>

              {/* Geometric Correction */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Geometric Factor: {settings.geometric_correction}
                </label>
                <input
                  type="range"
                  min="1.0"
                  max="3.0"
                  step="0.1"
                  value={settings.geometric_correction}
                  onChange={(e) => handleSettingsChange('geometric_correction', parseFloat(e.target.value))}
                  disabled={isCalculating}
                  className="w-full"
                />
              </div>

              {/* Options */}
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="emission_measure_correction"
                    checked={settings.emission_measure_correction}
                    onChange={(e) => handleSettingsChange('emission_measure_correction', e.target.checked)}
                    disabled={isCalculating}
                    className="rounded"
                  />
                  <label htmlFor="emission_measure_correction" className="text-gray-300 text-sm">
                    EM Correction
                  </label>
                </div>
              </div>
            </div>

            <button
              onClick={startCalculation}
              disabled={isCalculating}
              className={`w-full mt-6 flex items-center justify-center px-6 py-3 rounded-lg transition-colors ${
                isCalculating 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-red-600 hover:bg-red-700'
              } text-white`}
            >
              <Calculator className="h-5 w-5 mr-2" />
              {isCalculating ? 'Calculating...' : 'Calculate Energy'}
            </button>

            {isCalculating && (
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-red-500 h-2 rounded-full transition-all duration-300"
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
          {/* Energy Results Table */}
          {energyResults.length > 0 && (
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="h-5 w-5 mr-2" />
                Energy Calculation Results
              </h3>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-300 border-b border-gray-600">
                      <th className="text-left py-3">Flare ID</th>
                      <th className="text-left py-3">Class</th>
                      <th className="text-left py-3">Duration</th>
                      <th className="text-left py-3">Peak Flux</th>
                      <th className="text-left py-3">Total Energy</th>
                      <th className="text-left py-3">Temperature</th>
                      <th className="text-left py-3">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-300">
                    {energyResults.map((result) => (
                      <tr 
                        key={result.flare_id}
                        className={`border-b border-gray-700 cursor-pointer hover:bg-white/5 ${
                          selectedFlare?.flare_id === result.flare_id ? 'bg-red-400/10' : ''
                        }`}
                        onClick={() => setSelectedFlare(result)}
                      >
                        <td className="py-3">{result.flare_id}</td>
                        <td className={`py-3 ${getEnergyClassColor(result.energy_class)}`}>
                          {result.energy_class}
                        </td>
                        <td className="py-3">{result.duration}s</td>
                        <td className="py-3">{result.peak_flux.toExponential(1)} W/m²</td>
                        <td className="py-3">{formatEnergy(result.total_energy)} erg</td>
                        <td className="py-3">{formatTemperature(result.temperature)}</td>
                        <td className="py-3">{(result.confidence * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Detailed Analysis */}
          {selectedFlare && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Energy Breakdown */}
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Energy Breakdown: {selectedFlare.flare_id}
                </h3>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-gray-300">Thermal Energy</span>
                    <span className="text-white font-mono">
                      {formatEnergy(selectedFlare.thermal_energy)} erg
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                    <span className="text-gray-300">Radiated Energy</span>
                    <span className="text-white font-mono">
                      {formatEnergy(selectedFlare.radiated_energy)} erg
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-red-400/20 rounded-lg border border-red-400/30">
                    <span className="text-gray-300">Total Energy</span>
                    <span className="text-red-400 font-mono font-bold">
                      {formatEnergy(selectedFlare.total_energy)} erg
                    </span>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <p className="text-gray-300 text-sm">Peak Temperature</p>
                    <p className="text-white text-xl font-bold">
                      {formatTemperature(selectedFlare.temperature)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-300 text-sm">Emission Measure</p>
                    <p className="text-white text-xl font-bold">
                      {selectedFlare.emission_measure.toExponential(1)} cm⁻³
                    </p>
                  </div>
                </div>
              </div>

              {/* Temperature Evolution */}
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Temperature Evolution
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={temperatureEvolution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="time" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" tickFormatter={(value) => `${(value / 1e6).toFixed(0)} MK`} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1F2937', 
                          border: '1px solid #374151',
                          borderRadius: '8px'
                        }}
                        formatter={(value: any) => [`${(value / 1e6).toFixed(1)} MK`, 'Temperature']}
                        labelFormatter={(time) => `Time: ${time}s`}
                      />
                      <Line type="monotone" dataKey="temperature" stroke="#EF4444" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Energy Distribution */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <TrendingUp className="h-5 w-5 mr-2" />
              Energy Distribution Analysis
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart data={energyDistribution.slice(0, 30)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="log_energy" 
                    stroke="#9CA3AF"
                    label={{ value: 'Log₁₀(Energy [erg])', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    stroke="#9CA3AF" 
                    scale="log"
                    domain={['dataMin', 'dataMax']}
                    label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                    formatter={(value: any, name: string) => [
                      name === 'count' ? (typeof value === 'number' ? value.toFixed(0) : value) : value,
                      name === 'count' ? 'Count' : 'Energy'
                    ]}
                  />
                  <Scatter dataKey="count" fill="#EF4444" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Summary Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { 
                label: 'Total Events', 
                value: energyResults.length.toString(), 
                icon: Activity, 
                color: 'text-blue-400' 
              },
              { 
                label: 'Total Energy', 
                value: energyResults.length > 0 ? formatEnergy(energyResults.reduce((sum, r) => sum + r.total_energy, 0)) + ' erg' : '0 erg', 
                icon: Zap, 
                color: 'text-red-400' 
              },
              { 
                label: 'Avg Temperature', 
                value: energyResults.length > 0 ? formatTemperature(energyResults.reduce((sum, r) => sum + r.temperature, 0) / energyResults.length) : '0 MK', 
                icon: TrendingUp, 
                color: 'text-yellow-400' 
              },
              { 
                label: 'Avg Confidence', 
                value: energyResults.length > 0 ? `${(energyResults.reduce((sum, r) => sum + r.confidence, 0) / energyResults.length * 100).toFixed(1)}%` : '0%', 
                icon: AlertTriangle, 
                color: 'text-green-400' 
              }
            ].map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-300 text-sm">{stat.label}</p>
                      <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
                    </div>
                    <Icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
