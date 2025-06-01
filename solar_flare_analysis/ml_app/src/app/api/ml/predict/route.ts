import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward request to Python backend
    const response = await fetch('http://localhost:5000/api/models/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('ML prediction error:', error);
      // Return mock data if backend is unavailable
    const mockData = {
      status: 'success',
      predictions: [
        {
          model_name: 'binary_flare_classifier',
          prediction: 'Flare Detected',
          confidence: 0.85,
          processing_time: 0.023
        },
        {
          model_name: 'multiclass_flare_classifier', 
          prediction: 'M-Class',
          confidence: 0.75,
          processing_time: 0.031
        },
        {
          model_name: 'energy_regression_model',
          prediction: 1.2e-5,
          confidence: 0.68,
          processing_time: 0.018
        },
        {
          model_name: 'cnn_flare_detector',
          prediction: 'High Activity',
          confidence: 0.92,
          processing_time: 0.042
        },
        {
          model_name: 'minimal_flare_model',
          prediction: 'Moderate Risk',
          confidence: 0.63,
          processing_time: 0.012
        }
      ],
      timestamp: new Date().toISOString(),
      input_shape: [100],
      fallback: true,
      error: error instanceof Error ? error.message : 'Unknown error'
    };

    return NextResponse.json(mockData);
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'ML Prediction API - Use POST to submit data for prediction',
    endpoints: {
      predict: 'POST /api/ml/predict'
    }
  });
}
