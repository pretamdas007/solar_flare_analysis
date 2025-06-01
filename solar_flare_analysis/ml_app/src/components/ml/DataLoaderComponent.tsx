'use client';

import React, { useState, useCallback } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader, Database, Calendar, BarChart3, Sun, Flame, Star, Sparkles, Zap } from 'lucide-react';

interface FileInfo {
  name: string;
  size: number;
  type: string;
  lastModified: number;
}

interface ProcessingResult {
  success: boolean;
  message: string;
  data_points: number;
  time_range: [string, string];
  flares_detected: number;
  file_format: string;
  processing_time: number;
}

export default function DataLoaderComponent() {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<ProcessingResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Handle drag events
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Handle drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  // Handle file selection
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files ? Array.from(e.target.files) : [];
    handleFiles(selectedFiles);
  };

  // Process selected files
  const handleFiles = (fileList: File[]) => {
    setError(null);
    
    // Filter for supported file types
    const supportedFiles = fileList.filter(file => {
      const ext = file.name.toLowerCase().split('.').pop();
      return ['nc', 'h5', 'hdf5', 'csv', 'txt'].includes(ext || '');
    });

    if (supportedFiles.length === 0) {
      setError('No supported files found. Please upload .nc, .h5, .csv, or .txt files.');
      return;
    }

    const fileInfos: FileInfo[] = supportedFiles.map(file => ({
      name: file.name,
      size: file.size,
      type: file.type || 'unknown',
      lastModified: file.lastModified
    }));

    setFiles(fileInfos);
    processFiles(supportedFiles);
  };

  // Simulate file processing
  const processFiles = async (fileList: File[]) => {
    setProcessing(true);
    setResults([]);

    for (const file of fileList) {
      try {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        // Mock processing result
        const mockResult: ProcessingResult = {
          success: Math.random() > 0.1, // 90% success rate
          message: Math.random() > 0.1 ? 'File processed successfully' : 'Error: Invalid data format',
          data_points: Math.floor(Math.random() * 100000) + 10000,
          time_range: [
            new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            new Date().toISOString().split('T')[0]
          ],
          flares_detected: Math.floor(Math.random() * 50) + 5,
          file_format: file.name.split('.').pop()?.toUpperCase() || 'UNKNOWN',
          processing_time: Math.random() * 3 + 0.5
        };

        setResults(prev => [...prev, mockResult]);
      } catch (error) {
        const errorResult: ProcessingResult = {
          success: false,
          message: 'Processing failed: Unexpected error',
          data_points: 0,
          time_range: ['', ''],
          flares_detected: 0,
          file_format: 'ERROR',
          processing_time: 0
        };
        setResults(prev => [...prev, errorResult]);
      }
    }

    setProcessing(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  const getFileIcon = (fileName: string) => {
    const ext = fileName.toLowerCase().split('.').pop();
    switch (ext) {
      case 'nc': return <Database className="h-6 w-6 text-aurora-blue stellar-icon animate-solarPulse" />;
      case 'h5':
      case 'hdf5': return <BarChart3 className="h-6 w-6 text-solar-gold stellar-icon animate-stellarRotation" />;
      case 'csv': return <FileText className="h-6 w-6 text-coronal-orange stellar-icon animate-coronalWave" />;
      default: return <FileText className="h-6 w-6 text-plasma-violet stellar-icon animate-plasmaFlow" />;
    }
  };
  return (
    <div className="space-y-6">
      {/* Cosmic Upload Area */}
      <div className="solar-glassmorphism p-6 bg-gradient-to-br from-deep-space-black/50 to-plasma-violet/10 border-plasma-violet/50">        <div className="flex items-center space-x-3 mb-4">
          <Sun className="h-6 w-6 text-solar-gold stellar-icon animate-stellarRotation" />
          <h2 className="aurora-text text-xl font-semibold">Solar Data Loader</h2>
          <Flame className="h-5 w-5 text-coronal-orange stellar-icon animate-flareEruption" />
        </div>
        
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
            dragActive
              ? 'border-aurora-blue bg-gradient-to-br from-aurora-blue/30 to-plasma-violet/20 shadow-lg shadow-aurora-blue/20'
              : 'border-white/30 hover:border-solar-gold/60 bg-gradient-to-br from-white/5 to-white/10 hover:from-solar-gold/10 hover:to-coronal-orange/10'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center space-y-4">
            <Upload className={`h-12 w-12 stellar-icon transition-all duration-300 ${
              dragActive 
                ? 'text-aurora-blue animate-solarPulse scale-110' 
                : 'text-solar-gold animate-coronalWave hover:scale-105'
            }`} />
            <div>
              <p className="aurora-text text-lg font-medium">
                Drop GOES solar data files here or{' '}
                <label className="text-aurora-blue hover:text-plasma-violet cursor-pointer underline transition-colors duration-300">
                  browse cosmic files
                  <input
                    type="file"
                    multiple
                    accept=".nc,.h5,.hdf5,.csv,.txt"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                </label>
              </p>
              <p className="text-gray-300 text-sm mt-2 flex items-center justify-center space-x-2">
                <Star className="h-4 w-4 text-plasma-violet stellar-icon animate-auroraShimmer" />
                <span>Supports: NetCDF (.nc), HDF5 (.h5), CSV (.csv), and text files</span>
                <Star className="h-4 w-4 text-plasma-violet stellar-icon animate-auroraShimmer" />
              </p>
            </div>
          </div>
        </div>

        {/* Cosmic Supported Formats */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="solar-card border-aurora-blue bg-gradient-to-br from-aurora-blue/20 to-aurora-blue/5 hover:from-aurora-blue/30 hover:to-aurora-blue/10 transition-all duration-300">
            <Database className="h-6 w-6 text-aurora-blue mx-auto mb-2 stellar-icon animate-solarPulse" />
            <span className="aurora-text text-sm font-semibold">NetCDF</span>
            <p className="text-gray-300 text-xs">GOES/EXIS solar data</p>
          </div>
          <div className="solar-card border-solar-gold bg-gradient-to-br from-solar-gold/20 to-solar-gold/5 hover:from-solar-gold/30 hover:to-solar-gold/10 transition-all duration-300">
            <BarChart3 className="h-6 w-6 text-solar-gold mx-auto mb-2 stellar-icon animate-stellarRotation" />
            <span className="aurora-text text-sm font-semibold">HDF5</span>
            <p className="text-gray-300 text-xs">Scientific cosmic data</p>
          </div>
          <div className="solar-card border-coronal-orange bg-gradient-to-br from-coronal-orange/20 to-coronal-orange/5 hover:from-coronal-orange/30 hover:to-coronal-orange/10 transition-all duration-300">
            <FileText className="h-6 w-6 text-coronal-orange mx-auto mb-2 stellar-icon animate-coronalWave" />
            <span className="aurora-text text-sm font-semibold">CSV</span>
            <p className="text-gray-300 text-xs">Structured solar data</p>
          </div>
          <div className="solar-card border-plasma-violet bg-gradient-to-br from-plasma-violet/20 to-plasma-violet/5 hover:from-plasma-violet/30 hover:to-plasma-violet/10 transition-all duration-300">
            <FileText className="h-6 w-6 text-plasma-violet mx-auto mb-2 stellar-icon animate-plasmaFlow" />
            <span className="aurora-text text-sm font-semibold">Text</span>
            <p className="text-gray-300 text-xs">Raw flare data</p>
          </div>        </div>
      </div>

      {/* Cosmic Error Display */}
      {error && (
        <div className="solar-card bg-gradient-to-br from-solar-red/20 to-solar-red/5 border-solar-red flex items-center space-x-3">
          <AlertCircle className="h-5 w-5 text-solar-red stellar-icon animate-flareEruption" />
          <span className="text-solar-red font-medium">{error}</span>
        </div>
      )}

      {/* Cosmic File List */}
      {files.length > 0 && (
        <div className="solar-glassmorphism p-6 bg-gradient-to-br from-deep-space-black/50 to-aurora-blue/10 border-aurora-blue/50">
          <div className="flex items-center space-x-3 mb-4">
            <Sparkles className="h-5 w-5 text-aurora-blue stellar-icon animate-auroraShimmer" />
            <h3 className="aurora-text text-lg font-semibold">Selected Cosmic Files</h3>
            <Star className="h-4 w-4 text-plasma-violet stellar-icon animate-stellarRotation" />
          </div>
          <div className="space-y-3">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 solar-glassmorphism bg-gradient-to-r from-white/10 to-white/5 border border-white/20 hover:from-white/20 hover:to-white/10 transition-all duration-300">
                <div className="flex items-center space-x-3">
                  {getFileIcon(file.name)}
                  <div>
                    <span className="aurora-text font-medium">{file.name}</span>
                    <p className="text-gray-300 text-sm">
                      {formatFileSize(file.size)} • Modified {new Date(file.lastModified).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                {processing ? (
                  <Loader className="h-5 w-5 text-aurora-blue animate-spin stellar-icon" />
                ) : results[index] ? (
                  results[index].success ? (
                    <CheckCircle className="h-5 w-5 text-solar-gold stellar-icon animate-solarPulse" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-solar-red stellar-icon animate-flareEruption" />
                  )
                ) : null}
              </div>
            ))}
          </div>        </div>
      )}

      {/* Cosmic Processing Results */}
      {results.length > 0 && (
        <div className="solar-glassmorphism p-6 bg-gradient-to-br from-deep-space-black/50 to-solar-gold/10 border-solar-gold/50">
          <div className="flex items-center space-x-3 mb-4">
            <Zap className="h-5 w-5 text-solar-gold stellar-icon animate-flareEruption" />
            <h3 className="aurora-text text-lg font-semibold">Processing Results</h3>
            <Flame className="h-4 w-4 text-coronal-orange stellar-icon animate-coronalWave" />
          </div>
          <div className="space-y-4">
            {results.map((result, index) => (
              <div
                key={index}
                className={`solar-card p-4 ${
                  result.success
                    ? 'border-solar-gold bg-gradient-to-br from-solar-gold/20 to-solar-gold/5'
                    : 'border-solar-red bg-gradient-to-br from-solar-red/20 to-solar-red/5'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {result.success ? (
                      <CheckCircle className="h-5 w-5 text-solar-gold stellar-icon animate-solarPulse" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-solar-red stellar-icon animate-flareEruption" />
                    )}
                    <span className="aurora-text font-medium">{files[index]?.name}</span>
                  </div>
                  <span className={`text-sm font-semibold ${result.success ? 'text-solar-gold' : 'text-solar-red'}`}>
                    {result.processing_time.toFixed(2)}s
                  </span>
                </div>

                <p className={`text-sm mb-3 font-medium ${result.success ? 'text-solar-gold' : 'text-solar-red'}`}>
                  {result.message}
                </p>

                {result.success && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="solar-glassmorphism p-3 bg-gradient-to-br from-aurora-blue/20 to-transparent border-aurora-blue/30">
                      <span className="text-aurora-blue text-sm">Data Points:</span>
                      <div className="aurora-text font-bold">{result.data_points.toLocaleString()}</div>
                    </div>
                    <div className="solar-glassmorphism p-3 bg-gradient-to-br from-plasma-violet/20 to-transparent border-plasma-violet/30">
                      <span className="text-plasma-violet text-sm">Time Range:</span>
                      <div className="aurora-text text-xs font-semibold">{result.time_range[0]} to {result.time_range[1]}</div>
                    </div>                    <div className="solar-glassmorphism p-3 bg-gradient-to-br from-coronal-orange/20 to-transparent border-coronal-orange/30">
                      <span className="text-coronal-orange text-sm">Flares Detected:</span>
                      <div className="aurora-text font-bold">{result.flares_detected}</div>
                    </div>
                    <div className="solar-glassmorphism p-3 bg-gradient-to-br from-solar-gold/20 to-transparent border-solar-gold/30">
                      <span className="text-solar-gold text-sm">Format:</span>
                      <div className="aurora-text font-bold">{result.file_format}</div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Cosmic Summary */}
          {results.length > 0 && (
            <div className="mt-6 solar-card p-4 bg-gradient-to-br from-plasma-violet/20 to-aurora-blue/20 border-aurora-blue">
              <h4 className="aurora-text font-medium mb-3 flex items-center space-x-2">
                <Star className="h-4 w-4 stellar-icon animate-auroraShimmer" />
                <span>Processing Summary</span>
                <Sparkles className="h-4 w-4 stellar-icon animate-stellarRotation" />
              </h4>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <span className="text-gray-300">Total Files:</span>
                  <div className="aurora-text font-bold text-lg">{results.length}</div>
                </div>
                <div className="text-center">
                  <span className="text-gray-300">Successful:</span>
                  <div className="text-solar-gold font-bold text-lg">{results.filter(r => r.success).length}</div>
                </div>
                <div className="text-center">
                  <span className="text-gray-300">Failed:</span>
                  <div className="text-solar-red font-bold text-lg">{results.filter(r => !r.success).length}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
