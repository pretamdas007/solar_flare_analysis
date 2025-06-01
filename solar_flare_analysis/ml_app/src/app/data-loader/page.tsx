'use client';

import { useState, useCallback } from 'react';
import { Upload, Database, FileText, CheckCircle, AlertTriangle, Calendar, Activity } from 'lucide-react';
import { useNotification } from '@/components/NotificationSystem';

interface FileInfo {
  name: string;
  size: number;
  lastModified: number;
  type: string;
  status: 'pending' | 'uploading' | 'processing' | 'complete' | 'error';
  progress: number;
  metadata?: {
    time_range: string;
    data_points: number;
    instruments: string[];
    quality: string;
  };
}

export default function DataLoader() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const { showNotification } = useNotification();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    processFiles(droppedFiles);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      processFiles(selectedFiles);
    }
  }, []);

  const processFiles = (fileList: File[]) => {
    const newFiles: FileInfo[] = fileList.map(file => ({
      name: file.name,
      size: file.size,
      lastModified: file.lastModified,
      type: file.type,
      status: 'pending',
      progress: 0
    }));

    setFiles(prev => [...prev, ...newFiles]);
    
    // Simulate file processing
    newFiles.forEach((file, index) => {
      simulateFileProcessing(files.length + index);
    });
  };

  const simulateFileProcessing = async (fileIndex: number) => {
    setIsProcessing(true);
    
    // Update status to uploading
    setFiles(prev => prev.map((file, idx) => 
      idx === fileIndex ? { ...file, status: 'uploading' } : file
    ));

    // Simulate upload progress
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setFiles(prev => prev.map((file, idx) => 
        idx === fileIndex ? { ...file, progress } : file
      ));
    }

    // Update to processing
    setFiles(prev => prev.map((file, idx) => 
      idx === fileIndex ? { ...file, status: 'processing', progress: 0 } : file
    ));

    // Simulate processing
    for (let progress = 0; progress <= 100; progress += 20) {
      await new Promise(resolve => setTimeout(resolve, 300));
      setFiles(prev => prev.map((file, idx) => 
        idx === fileIndex ? { ...file, progress } : file
      ));
    }

    // Complete with mock metadata
    const mockMetadata = {
      time_range: '2024-01-01 to 2024-01-31',
      data_points: Math.floor(Math.random() * 100000) + 50000,
      instruments: ['EXIS', 'XRS-A', 'XRS-B'],
      quality: Math.random() > 0.5 ? 'High' : 'Medium'
    };

    setFiles(prev => prev.map((file, idx) => 
      idx === fileIndex ? { 
        ...file, 
        status: 'complete', 
        progress: 100,
        metadata: mockMetadata
      } : file
    ));

    setIsProcessing(false);
    showNotification('File processed successfully!', 'success');
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-red-400" />;
      case 'uploading':
      case 'processing':
        return <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400" />;
      default:
        return <FileText className="h-5 w-5 text-gray-400" />;
    }
  };

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
          <Database className="h-8 w-8 mr-3 text-blue-400" />
          Load GOES XRS Data
        </h1>
        <p className="text-gray-300">
          Upload and validate GOES/EXIS satellite data files. Supports netCDF (.nc) and CSV formats.
        </p>
      </div>

      {/* Upload Area */}
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h2 className="text-xl font-semibold text-white mb-4">Upload Data Files</h2>
        
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
            isDragging 
              ? 'border-blue-400 bg-blue-400/10' 
              : 'border-gray-400 hover:border-gray-300'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-white text-lg font-medium mb-2">
            Drag and drop your files here
          </h3>
          <p className="text-gray-300 mb-4">
            or
          </p>
          <label className="inline-flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer transition-colors">
            <Upload className="h-5 w-5 mr-2" />
            Choose Files
            <input
              type="file"
              multiple
              accept=".nc,.csv,.txt"
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>
          <p className="text-gray-400 text-sm mt-4">
            Supported formats: .nc (netCDF), .csv, .txt • Max file size: 500MB
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-xl font-semibold text-white mb-4">Uploaded Files</h2>
          <div className="space-y-4">
            {files.map((file, index) => (
              <div key={index} className="bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(file.status)}
                    <div>
                      <h3 className="text-white font-medium">{file.name}</h3>
                      <p className="text-gray-400 text-sm">
                        {formatFileSize(file.size)} • {new Date(file.lastModified).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-white text-sm capitalize">{file.status}</p>
                    {(file.status === 'uploading' || file.status === 'processing') && (
                      <p className="text-gray-400 text-sm">{file.progress}%</p>
                    )}
                  </div>
                </div>

                {/* Progress Bar */}
                {(file.status === 'uploading' || file.status === 'processing') && (
                  <div className="w-full bg-gray-700 rounded-full h-2 mb-3">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${file.progress}%` }}
                    />
                  </div>
                )}

                {/* Metadata */}
                {file.metadata && file.status === 'complete' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-4 pt-4 border-t border-white/10">
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-4 w-4 text-blue-400" />
                      <div>
                        <p className="text-gray-400 text-xs">Time Range</p>
                        <p className="text-white text-sm">{file.metadata.time_range}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Activity className="h-4 w-4 text-green-400" />
                      <div>
                        <p className="text-gray-400 text-xs">Data Points</p>
                        <p className="text-white text-sm">{file.metadata.data_points.toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Database className="h-4 w-4 text-purple-400" />
                      <div>
                        <p className="text-gray-400 text-xs">Instruments</p>
                        <p className="text-white text-sm">{file.metadata.instruments.join(', ')}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-yellow-400" />
                      <div>
                        <p className="text-gray-400 text-xs">Quality</p>
                        <p className="text-white text-sm">{file.metadata.quality}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Preview */}
      {files.some(f => f.status === 'complete') && (
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <h2 className="text-xl font-semibold text-white mb-4">Data Preview</h2>
          <div className="bg-black/30 rounded-lg p-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-300 border-b border-gray-600">
                  <th className="text-left py-2">Timestamp</th>
                  <th className="text-left py-2">XRS-A (W/m²)</th>
                  <th className="text-left py-2">XRS-B (W/m²)</th>
                  <th className="text-left py-2">Quality</th>
                </tr>
              </thead>
              <tbody className="text-gray-300">
                <tr className="border-b border-gray-700">
                  <td className="py-2">2024-01-01 00:00:00</td>
                  <td className="py-2">1.23e-08</td>
                  <td className="py-2">5.67e-09</td>
                  <td className="py-2">Good</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-2">2024-01-01 00:00:02</td>
                  <td className="py-2">1.25e-08</td>
                  <td className="py-2">5.71e-09</td>
                  <td className="py-2">Good</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-2">2024-01-01 00:00:04</td>
                  <td className="py-2">1.28e-08</td>
                  <td className="py-2">5.75e-09</td>
                  <td className="py-2">Good</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
