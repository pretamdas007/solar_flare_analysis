'use client';

import { useState, useEffect } from 'react';
import { Brain, Play, Square, Settings, BarChart3, Activity, Zap, Target, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useNotification } from '@/components/NotificationSystem';

interface TrainingConfig {
  model_type: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  validation_split: number;
  early_stopping: boolean;
  patience: number;
  dropout_rate: number;
  l2_regularization: number;
}

interface TrainingMetrics {
  epoch: number;
  loss: number;
  val_loss: number;
  accuracy: number;
  val_accuracy: number;
  flare_detection_f1: number;
  nanoflare_precision: number;
}

export default function TrainModel() {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  const [config, setConfig] = useState<TrainingConfig>({
    model_type: 'enhanced_cnn_lstm',
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2,
    early_stopping: true,
    patience: 10,
    dropout_rate: 0.3,
    l2_regularization: 0.01
  });
  const { showNotification } = useNotification();

  const startTraining = async () => {
    setIsTraining(true);
    setCurrentEpoch(0);
    setTrainingMetrics([]);
    showNotification('Training started!', 'info');

    // Simulate training process
    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate epoch duration
      
      // Generate realistic training metrics
      const loss = Math.max(0.1, 2.0 * Math.exp(-epoch * 0.05) + Math.random() * 0.1);
      const val_loss = Math.max(0.1, loss + Math.random() * 0.3 - 0.15);
      const accuracy = Math.min(0.99, 0.5 + (1 - Math.exp(-epoch * 0.08)) * 0.45 + Math.random() * 0.05);
      const val_accuracy = Math.max(0.1, accuracy - Math.random() * 0.1);

      const newMetric: TrainingMetrics = {
        epoch,
        loss,
        val_loss,
        accuracy,
        val_accuracy,
        flare_detection_f1: Math.min(0.98, accuracy + Math.random() * 0.05),
        nanoflare_precision: Math.min(0.95, accuracy - 0.05 + Math.random() * 0.1)
      };

      setCurrentEpoch(epoch);
      setTrainingMetrics(prev => [...prev, newMetric]);

      // Early stopping simulation
      if (config.early_stopping && epoch > config.patience) {
        const recentLosses = trainingMetrics.slice(-config.patience).map(m => m.val_loss);
        const isImproving = recentLosses.some((loss, i) => i === 0 || loss < recentLosses[i - 1]);
        
        if (!isImproving && Math.random() > 0.7) {
          showNotification(`Early stopping at epoch ${epoch}`, 'info');
          break;
        }
      }
    }

    setIsTraining(false);
    showNotification('Training completed successfully!', 'success');
  };

  const stopTraining = () => {
    setIsTraining(false);
    showNotification('Training stopped by user', 'warning');
  };

  const handleConfigChange = (key: keyof TrainingConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const modelArchitectures = [
    { value: 'enhanced_cnn_lstm', label: 'Enhanced CNN-LSTM', description: 'Advanced architecture with attention mechanism' },
    { value: 'residual_net', label: 'Residual Network', description: 'Deep residual network for complex patterns' },
    { value: 'transformer', label: 'Transformer', description: 'Self-attention based model for sequence analysis' },
    { value: 'ensemble', label: 'Ensemble Model', description: 'Combination of multiple architectures' }
  ];

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
          <Brain className="h-8 w-8 mr-3 text-purple-400" />
          Train ML Model
        </h1>
        <p className="text-gray-300">
          Configure and train the enhanced neural network for solar flare detection and nanoflare identification.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Settings className="h-6 w-6 mr-2" />
              Model Configuration
            </h2>
            
            <div className="space-y-4">
              {/* Model Architecture */}
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Model Architecture
                </label>
                <select
                  value={config.model_type}
                  onChange={(e) => handleConfigChange('model_type', e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {modelArchitectures.map(arch => (
                    <option key={arch.value} value={arch.value} className="bg-gray-800">
                      {arch.label}
                    </option>
                  ))}
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {modelArchitectures.find(a => a.value === config.model_type)?.description}
                </p>
              </div>

              {/* Training Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
                    disabled={isTraining}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="1"
                    max="1000"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => handleConfigChange('batch_size', parseInt(e.target.value))}
                    disabled={isTraining}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="1"
                    max="512"
                  />
                </div>
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Learning Rate: {config.learning_rate}
                </label>
                <input
                  type="range"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={config.learning_rate}
                  onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                  disabled={isTraining}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Validation Split: {config.validation_split}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.05"
                  value={config.validation_split}
                  onChange={(e) => handleConfigChange('validation_split', parseFloat(e.target.value))}
                  disabled={isTraining}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Dropout Rate: {config.dropout_rate}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.8"
                  step="0.1"
                  value={config.dropout_rate}
                  onChange={(e) => handleConfigChange('dropout_rate', parseFloat(e.target.value))}
                  disabled={isTraining}
                  className="w-full"
                />
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="early_stopping"
                  checked={config.early_stopping}
                  onChange={(e) => handleConfigChange('early_stopping', e.target.checked)}
                  disabled={isTraining}
                  className="rounded"
                />
                <label htmlFor="early_stopping" className="text-gray-300 text-sm">
                  Enable Early Stopping
                </label>
              </div>

              {config.early_stopping && (
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Patience
                  </label>
                  <input
                    type="number"
                    value={config.patience}
                    onChange={(e) => handleConfigChange('patience', parseInt(e.target.value))}
                    disabled={isTraining}
                    className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="1"
                    max="50"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Training Controls */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4">Training Controls</h3>
            <div className="space-y-4">
              {!isTraining ? (
                <button
                  onClick={startTraining}
                  className="w-full flex items-center justify-center px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  <Play className="h-5 w-5 mr-2" />
                  Start Training
                </button>
              ) : (
                <button
                  onClick={stopTraining}
                  className="w-full flex items-center justify-center px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                >
                  <Square className="h-5 w-5 mr-2" />
                  Stop Training
                </button>
              )}

              {isTraining && (
                <div className="text-center">
                  <p className="text-white text-lg">Epoch {currentEpoch} / {config.epochs}</p>
                  <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                    <div 
                      className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentEpoch / config.epochs) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Training Metrics */}
        <div className="lg:col-span-2 space-y-6">
          {/* Loss Chart */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2" />
              Training Loss
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="#8B5CF6" name="Training Loss" />
                  <Line type="monotone" dataKey="val_loss" stroke="#EF4444" name="Validation Loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Accuracy Chart */}
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Target className="h-5 w-5 mr-2" />
              Model Accuracy
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[0, 1]} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                    formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, '']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#10B981" name="Training Accuracy" />
                  <Line type="monotone" dataKey="val_accuracy" stroke="#F59E0B" name="Validation Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Current Metrics */}
          {trainingMetrics.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { 
                  label: 'Current Loss', 
                  value: trainingMetrics[trainingMetrics.length - 1]?.loss.toFixed(4), 
                  icon: Activity, 
                  color: 'text-purple-400' 
                },
                { 
                  label: 'Accuracy', 
                  value: `${(trainingMetrics[trainingMetrics.length - 1]?.accuracy * 100).toFixed(1)}%`, 
                  icon: Target, 
                  color: 'text-green-400' 
                },
                { 
                  label: 'Flare F1 Score', 
                  value: `${(trainingMetrics[trainingMetrics.length - 1]?.flare_detection_f1 * 100).toFixed(1)}%`, 
                  icon: Zap, 
                  color: 'text-yellow-400' 
                },
                { 
                  label: 'Nanoflare Precision', 
                  value: `${(trainingMetrics[trainingMetrics.length - 1]?.nanoflare_precision * 100).toFixed(1)}%`, 
                  icon: TrendingUp, 
                  color: 'text-blue-400' 
                }
              ].map((metric, index) => {
                const Icon = metric.icon;
                return (
                  <div key={index} className="bg-white/5 rounded-xl p-4 border border-white/10">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-gray-300 text-sm">{metric.label}</p>
                        <p className={`text-xl font-bold ${metric.color}`}>{metric.value}</p>
                      </div>
                      <Icon className={`h-6 w-6 ${metric.color}`} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
