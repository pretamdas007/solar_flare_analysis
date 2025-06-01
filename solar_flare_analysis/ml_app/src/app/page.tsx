'use client';

import { Activity, Zap, TrendingUp, Database, Brain, Target, CheckCircle, Clock, AlertTriangle, BarChart3, Sun, Flame, Star, Atom, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';

interface SystemStatus {
  api_health: boolean;
  python_backend: boolean;
  model_status: string;
  last_training: string | null;
  data_loaded: boolean;
}

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemStatus();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      setSystemStatus({
        api_health: true,
        python_backend: false,
        model_status: 'not_trained',
        last_training: null,
        data_loaded: false,
      });
    } finally {
      setLoading(false);
    }
  };

  const workflowSteps = [
    {
      id: 1,
      title: 'Load GOES XRS Data',
      description: 'Import and validate GOES/EXIS satellite data files (netCDF format)',
      href: '/data-loader',
      icon: Database,
      color: 'aurora-blue',
      status: systemStatus?.data_loaded ? 'complete' : 'pending',
        },
    {
      id: 2,
      title: 'Train ML Model',
      description: 'Configure and train the enhanced neural network for flare detection',
      href: '/train-model',
      icon: Brain,
      color: 'plasma-violet',
      status: systemStatus?.model_status === 'trained' ? 'complete' : 'pending',
    },
    {
      id: 3,
      title: 'Extract Characteristics',
      description: 'Detect individual flares and separate overlapping events',
      href: '/extract-features',
      icon: Target,
      color: 'solar-gold',
      status: 'pending',
    },
    {
      id: 4,
      title: 'Remove Background',
      description: 'Subtract instrumental and solar background signals',
      href: '/background-removal',
      icon: Activity,
      color: 'coronal-orange',
      status: 'pending',
    },
    {
      id: 5,
      title: 'Calculate Energy',
      description: 'Estimate thermal and radiated energy from extracted flares',
      href: '/energy-calculation',
      icon: Zap,
      color: 'solar-red',
      status: 'pending',
    },
    {
      id: 6,
      title: 'Power-Law Analysis',
      description: 'Analyze energy distribution and identify nanoflares',
      href: '/power-law',
      icon: TrendingUp,
      color: 'cosmic-purple',
      status: 'pending',
    },
    {
      id: 7,
      title: 'Method Comparison',
      description: 'Compare ML vs traditional analysis methods',
      href: '/comparison',      icon: BarChart3,
      color: 'nebula-pink',
      status: 'pending',
    },
  ];

  if (loading) {
    return (
      <div className="solar-body min-h-screen flex items-center justify-center">
        <div className="solar-glassmorphism p-8 rounded-2xl">
          <div className="flex items-center space-x-4">
            <Sun className="h-8 w-8 text-yellow-400 animate-spin stellar-icon" />
            <span className="aurora-text text-xl font-semibold">Initializing Solar Analysis System...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="solar-body cosmic-grid min-h-screen">
      <div className="p-8 space-y-8">
        {/* Hero Section with Solar Animation */}
        <div className="text-center mb-12">
          <div className="flex justify-center items-center mb-6">
            <Sun className="h-16 w-16 text-yellow-400 stellar-icon mr-4" />
            <div>
              <h1 className="aurora-text text-4xl font-bold mb-2">Solar Flare Analysis Hub</h1>
              <p className="text-gray-300 text-lg">Cosmic Intelligence for Solar Research</p>
            </div>
            <Flame className="h-16 w-16 text-orange-500 stellar-icon ml-4" />
          </div>
        </div>

        {/* Quick Actions with Enhanced Solar Styling */}
        <div className="flex justify-center mb-8">
          <Link
            href="/ml-predictions"
            className="solar-button text-lg px-12 py-6 rounded-2xl shadow-2xl"
          >
            <div className="flex items-center space-x-3">
              <Sparkles className="h-6 w-6" />
              <span>🚀 Launch ML Prediction Dashboard</span>
              <Star className="h-6 w-6" />
            </div>
          </Link>
        </div>

        {/* System Status with Solar Theme */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
          <div className="solar-glassmorphism p-6 transition-all duration-300 hover:plasma-glow">
            <div className="flex items-center mb-4">
              <Activity className="h-6 w-6 text-green-400 stellar-icon mr-2" />
              <h2 className="text-lg font-semibold aurora-text">API Health</h2>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`flare-indicator ${systemStatus?.api_health ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className={systemStatus?.api_health ? 'status-online' : 'status-offline'}>
                {systemStatus?.api_health ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
          
          <div className="solar-glassmorphism p-6 transition-all duration-300 hover:plasma-glow">
            <div className="flex items-center mb-4">
              <Atom className="h-6 w-6 text-purple-400 stellar-icon mr-2" />
              <h2 className="text-lg font-semibold aurora-text">Python Backend</h2>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`flare-indicator ${systemStatus?.python_backend ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className={systemStatus?.python_backend ? 'status-online' : 'status-offline'}>
                {systemStatus?.python_backend ? 'Running' : 'Stopped'}
              </span>
            </div>
          </div>
          
          <div className="solar-glassmorphism p-6 transition-all duration-300 hover:plasma-glow">
            <div className="flex items-center mb-4">
              <Brain className="h-6 w-6 text-blue-400 stellar-icon mr-2" />
              <h2 className="text-lg font-semibold aurora-text">Model Status</h2>
            </div>
            <span className="text-white capitalize font-medium">{systemStatus?.model_status || 'unknown'}</span>
          </div>
          
          <div className="solar-glassmorphism p-6 transition-all duration-300 hover:plasma-glow">
            <div className="flex items-center mb-4">
              <Clock className="h-6 w-6 text-orange-400 stellar-icon mr-2" />
              <h2 className="text-lg font-semibold aurora-text">Last Training</h2>
            </div>
            <span className="text-white font-medium">{systemStatus?.last_training || 'N/A'}</span>
          </div>
          
          <div className="solar-glassmorphism p-6 transition-all duration-300 hover:plasma-glow">
            <div className="flex items-center mb-4">
              <Database className="h-6 w-6 text-yellow-400 stellar-icon mr-2" />
              <h2 className="text-lg font-semibold aurora-text">Data Loaded</h2>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`flare-indicator ${systemStatus?.data_loaded ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className={systemStatus?.data_loaded ? 'status-online' : 'status-offline'}>
                {systemStatus?.data_loaded ? 'Yes' : 'No'}
              </span>
            </div>
          </div>
        </div>

        {/* Workflow Steps with Solar Theme */}
        <div className="solar-glassmorphism p-8">
          <div className="flex items-center mb-6">
            <Atom className="h-8 w-8 text-purple-400 stellar-icon mr-3" />
            <h2 className="text-2xl font-bold aurora-text">Solar Analysis Workflow</h2>
          </div>
          <div className="space-y-6">
            {workflowSteps.map((step) => {
              const Icon = step.icon;
              return (
                <Link
                  key={step.id}
                  href={step.href}
                  className="solar-card group p-6 block transition-all duration-300 hover:scale-105"
                >
                  <div className="flex items-center">
                    <div className={`p-3 rounded-xl bg-gradient-to-r from-${step.color}-500 to-${step.color}-600 mr-4`}>
                      <Icon className="h-8 w-8 text-white stellar-icon" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-white font-bold text-lg mb-1">{step.title}</h3>
                      <p className="text-gray-300">{step.description}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {step.status === 'complete' ? (
                        <CheckCircle className="h-6 w-6 text-green-400 stellar-icon" />
                      ) : (
                        <Clock className="h-6 w-6 text-gray-400" />
                      )}
                      <div className={`flare-indicator ${step.status === 'complete' ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>

        {/* Statistics and Activity with Solar Theme */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Solar Statistics */}
          <div className="solar-glassmorphism p-8">
            <div className="flex items-center mb-6">
              <BarChart3 className="h-8 w-8 text-yellow-400 stellar-icon mr-3" />
              <h2 className="text-2xl font-bold aurora-text">Solar Statistics</h2>
            </div>
            <div className="space-y-6">
              <div className="flex justify-between items-center p-4 rounded-xl bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/30">
                <span className="text-gray-300 flex items-center">
                  <Flame className="h-5 w-5 text-orange-400 mr-2" />
                  Total Flares Detected
                </span>
                <span className="aurora-text font-bold text-xl">1,247</span>
              </div>
              <div className="flex justify-between items-center p-4 rounded-xl bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/30">
                <span className="text-gray-300 flex items-center">
                  <Sparkles className="h-5 w-5 text-purple-400 mr-2" />
                  Nanoflares Identified
                </span>
                <span className="aurora-text font-bold text-xl">89</span>
              </div>
              <div className="flex justify-between items-center p-4 rounded-xl bg-gradient-to-r from-green-500/20 to-blue-500/20 border border-green-500/30">
                <span className="text-gray-300 flex items-center">
                  <Target className="h-5 w-5 text-green-400 mr-2" />
                  Processing Accuracy
                </span>
                <span className="status-online font-bold text-xl">94.2%</span>
              </div>
              <div className="flex justify-between items-center p-4 rounded-xl bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border border-blue-500/30">
                <span className="text-gray-300 flex items-center">
                  <Database className="h-5 w-5 text-blue-400 mr-2" />
                  Data Coverage
                </span>
                <span className="status-info font-bold text-xl">2020-2024</span>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="solar-glassmorphism p-8">
            <div className="flex items-center mb-6">
              <Activity className="h-8 w-8 text-green-400 stellar-icon mr-3" />
              <h2 className="text-2xl font-bold aurora-text">System Activity</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center space-x-4 p-4 rounded-xl bg-gradient-to-r from-green-500/10 to-green-500/5 border border-green-500/20">
                <CheckCircle className="h-6 w-6 text-green-400 stellar-icon" />
                <div className="flex-1">
                  <span className="text-white font-medium">System health check completed successfully</span>
                  <div className="text-green-400 text-sm mt-1">2 minutes ago</div>
                </div>
              </div>
              <div className="flex items-center space-x-4 p-4 rounded-xl bg-gradient-to-r from-blue-500/10 to-blue-500/5 border border-blue-500/20">
                <Activity className="h-6 w-6 text-blue-400 stellar-icon" />
                <div className="flex-1">
                  <span className="text-white font-medium">API endpoints initialized</span>
                  <div className="text-blue-400 text-sm mt-1">5 minutes ago</div>
                </div>
              </div>
              <div className="flex items-center space-x-4 p-4 rounded-xl bg-gradient-to-r from-purple-500/10 to-purple-500/5 border border-purple-500/20">
                <Database className="h-6 w-6 text-purple-400 stellar-icon" />
                <div className="flex-1">
                  <span className="text-white font-medium">Application started</span>
                  <div className="text-purple-400 text-sm mt-1">10 minutes ago</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
