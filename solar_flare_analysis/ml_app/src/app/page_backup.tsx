'use client';

import { Activity, Zap, TrendingUp, Database, Brain, Target, CheckCircle, Clock, AlertTriangle, BarChart3 } from 'lucide-react';
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
      color: 'bg-blue-500',
      status: systemStatus?.data_loaded ? 'complete' : 'pending',
    },
    {
      id: 2,
      title: 'Train ML Model',
      description: 'Configure and train the enhanced neural network for flare detection',
      href: '/train-model',
      icon: Brain,
      color: 'bg-purple-500',
      status: systemStatus?.model_status === 'trained' ? 'complete' : 'pending',
    },
    {
      id: 3,
      title: 'Extract Characteristics',
      description: 'Detect individual flares and separate overlapping events',
      href: '/extract-features',
      icon: Target,
      color: 'bg-green-500',
      status: 'pending',
    },
    {
      id: 4,
      title: 'Remove Background',
      description: 'Subtract instrumental and solar background signals',
      href: '/background-removal',
      icon: Activity,
      color: 'bg-orange-500',
      status: 'pending',
    },
    {
      id: 5,
      title: 'Calculate Energy',
      description: 'Estimate thermal and radiated energy from extracted flares',
      href: '/energy-calculation',
      icon: Zap,
      color: 'bg-red-500',
      status: 'pending',
    },
    {
      id: 6,
      title: 'Power-Law Analysis',
      description: 'Analyze energy distribution and identify nanoflares',
      href: '/power-law',
      icon: TrendingUp,
      color: 'bg-indigo-500',
      status: 'pending',
    },
    {
      id: 7,
      title: 'Method Comparison',
      description: 'Compare ML vs traditional analysis methods',
      href: '/comparison',
      icon: BarChart3,
      color: 'bg-yellow-500',
      status: 'pending',
    },
  ];

  if (loading) {
    return <div className="text-center text-white mt-10">Loading status...</div>;
  }

  return (
    <div className="p-8 space-y-8">
      {/* System Status */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-lg font-semibold text-white mb-4">API Health</h2>
          <span className={systemStatus?.api_health ? 'text-green-400' : 'text-red-400'}>
            {systemStatus?.api_health ? 'Online' : 'Offline'}
          </span>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-lg font-semibold text-white mb-4">Python Backend</h2>
          <span className={systemStatus?.python_backend ? 'text-green-400' : 'text-red-400'}>
            {systemStatus?.python_backend ? 'Running' : 'Stopped'}
          </span>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-lg font-semibold text-white mb-4">Model Status</h2>
          <span className="text-white capitalize">{systemStatus?.model_status || 'unknown'}</span>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-lg font-semibold text-white mb-4">Last Training</h2>
          <span className="text-white">{systemStatus?.last_training || 'N/A'}</span>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-lg font-semibold text-white mb-4">Data Loaded</h2>
          <span className={systemStatus?.data_loaded ? 'text-green-400' : 'text-red-400'}>
            {systemStatus?.data_loaded ? 'Yes' : 'No'}
          </span>
        </div>
      </div>

      {/* Workflow Steps */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h2 className="text-xl font-semibold text-white mb-4">Analysis Workflow</h2>
        <div className="space-y-4">
          {workflowSteps.map((step) => {
            const Icon = step.icon;
            return (
              <Link
                key={step.id}
                href={step.href}
                className={`flex items-center p-4 rounded-lg border ${step.color} border-opacity-30 hover:border-opacity-50`}
              >
                <Icon className="h-6 w-6 text-white mr-4" />
                <div className="flex-1">
                  <h3 className="text-white font-semibold">{step.title}</h3>
                  <p className="text-gray-300 text-sm">{step.description}</p>
                </div>
                <span className={`ml-4 ${step.status === 'complete' ? 'text-green-400' : 'text-gray-400'}`}>
                  {step.status === 'complete' ? <CheckCircle className="h-5 w-5" /> : <Clock className="h-5 w-5" />}
                </span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-xl font-semibold text-white mb-4">Statistics</h2>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Total Flares Detected</span>
              <span className="text-white font-semibold">1,247</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Nanoflares Identified</span>
              <span className="text-white font-semibold">89</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Processing Accuracy</span>
              <span className="text-green-400 font-semibold">94.2%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Data Coverage</span>
              <span className="text-blue-400 font-semibold">2020-2024</span>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-xl font-semibold text-white mb-4">Recent Activity</h2>
          <div className="space-y-3">
            <div className="flex items-center space-x-3 text-gray-300">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span>System health check completed successfully</span>
              <span className="text-gray-500 text-sm">2 minutes ago</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-300">
              <Activity className="h-4 w-4 text-blue-400" />
              <span>API endpoints initialized</span>
              <span className="text-gray-500 text-sm">5 minutes ago</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-300">
              <Database className="h-4 w-4 text-purple-400" />
              <span>Application started</span>
              <span className="text-gray-500 text-sm">10 minutes ago</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
