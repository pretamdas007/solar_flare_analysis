import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  try {
    // Forward request to Python backend
    const response = await fetch('http://localhost:5000/api/analysis/flares', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(10000) // 10 second timeout
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Flare analysis error:', error);
    
    // Return mock data if backend is unavailable
    const mockData = {
      status: 'success',
      flares: [
        {
          id: 1,
          timestamp: '2024-06-01T10:30:00Z',
          class: 'M',
          peak_flux: 1.2e-5,
          duration_minutes: 15,
          confidence: 0.92
        },
        {
          id: 2,
          timestamp: '2024-06-01T14:45:00Z',
          class: 'C',
          peak_flux: 3.4e-6,
          duration_minutes: 8,
          confidence: 0.78
        },
        {
          id: 3,
          timestamp: '2024-06-01T18:20:00Z',
          class: 'X',
          peak_flux: 5.2e-4,
          duration_minutes: 25,
          confidence: 0.95
        }
      ],
      total_count: 3,
      analysis_time: new Date().toISOString(),
      fallback: true,
      error: error instanceof Error ? error.message : 'Unknown error'
    };

    return NextResponse.json(mockData);
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward request to Python backend
    const response = await fetch('http://localhost:5000/api/analysis/detect-flares', {
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
    console.error('Flare detection error:', error);
    
    return NextResponse.json({
      error: error instanceof Error ? error.message : 'Unknown error',
      fallback: true
    }, { status: 503 });
  }
}
