import { NextRequest, NextResponse } from 'next/server';

export async function GET() {  try {
    // Check if Python backend is available
    let pythonAvailable = false;
    let mlAvailable = false;
    let backendData = null;
    
    try {
      const response = await fetch('http://localhost:5000/health', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (response.ok) {
        backendData = await response.json();
        pythonAvailable = true;
        mlAvailable = backendData.ml_models_loaded || false;
      }
    } catch (error) {
      // Python backend not available
      pythonAvailable = false;
      mlAvailable = false;
    }

    return NextResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      api_version: '1.0.0',
      services: {
        frontend: true,
        backend_python: pythonAvailable,
        ml_models: mlAvailable
      },
      api_health: true,
      python_backend: pythonAvailable,
      model_status: mlAvailable ? 'trained' : 'not_trained',
      last_training: mlAvailable ? backendData?.timestamp : null,
      data_loaded: backendData?.data_loaded || false,
      ml_available: mlAvailable,
      python_backend_available: pythonAvailable,
      fallback_mode: !mlAvailable,
      backend_response: backendData
    });
    
  } catch (error) {
    console.error('Health check failed:', error);
    
    return NextResponse.json({
      status: 'degraded',
      timestamp: new Date().toISOString(),
      api_version: '1.0.0',
      services: {
        frontend: true,
        backend_python: false,
        ml_models: false
      },
      ml_available: false,
      python_backend_available: false,
      fallback_mode: true,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 503 });
  }
}
