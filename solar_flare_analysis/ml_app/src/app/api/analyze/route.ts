import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    try {
      // Try Python backend first
      const pythonFormData = new FormData();
      pythonFormData.append('file', file);
      
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: pythonFormData,
        signal: AbortSignal.timeout(30000)
      });

      if (response.ok) {
        const result = await response.json();
        return NextResponse.json(result);
      }
    } catch (error) {
      console.log('Python backend unavailable, using fallback data');
    }

    // Return mock data as fallback
    const mockResults = {
      success: true,
      separated_flares: generateMockFlares(50),
      nanoflares: generateMockFlares(15, true),
      energy_analysis: {
        total_energy: 2.5e32,
        average_energy: 1.8e28,
        median_energy: 8.2e27,
        power_law_index: -1.85,
        nanoflare_energy_fraction: 0.35,
        energy_range: [1e24, 1e30] as [number, number]
      },
      statistics: {
        total_flares: 50,
        nanoflare_count: 15,
        nanoflare_percentage: 30,
        average_energy: 1.8e28,
        power_law_index: -1.85,
        flare_types: { nano: 15, micro: 20, minor: 12, major: 3 }
      },
      visualizations: {
        energy_histogram: generateEnergyHistogram(),
        flare_timeline: generateMockFlares(50)
      },
      metadata: {
        file_processed: file.name,
        processing_time: "2.3s",
        data_points: 86400,
        model_version: "mock-v1.0"
      },
      fallback_data: true
    };

    return NextResponse.json(mockResults);
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Analysis failed', message: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

function generateMockFlares(count: number, nanoOnly: boolean = false): any[] {
  const flares: any[] = [];
  const now = new Date();
  
  for (let i = 0; i < count; i++) {
    const hours = Math.random() * 24;
    const timestamp = new Date(now.getTime() - hours * 60 * 60 * 1000);
    const energy = nanoOnly ? Math.random() * 1e26 + 1e24 : Math.random() * 1e30 + 1e24;
    const intensity = Math.log10(energy) + Math.random() * 2 - 1;
    const alpha = nanoOnly ? Math.random() * 2 + 2.1 : Math.random() * 4 - 2;
    
    let flareType: string;
    if (energy < 1e25) flareType = 'nano';
    else if (energy < 1e27) flareType = 'micro';
    else if (energy < 1e29) flareType = 'minor';
    else flareType = 'major';
    
    flares.push({
      timestamp: timestamp.toISOString(),
      intensity: Math.max(0.1, intensity),
      energy: energy,
      alpha: alpha,
      flare_type: flareType,
      confidence: Math.random() * 0.3 + 0.7,
      is_nanoflare: Math.abs(alpha) > 2
    });
  }
  
  return flares.sort((a: any, b: any) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
}

function generateEnergyHistogram(): any[] {
  const histogram: any[] = [];
  for (let i = 24; i <= 30; i++) {
    const energy = Math.pow(10, i);
    const count = Math.floor(Math.random() * 20 + 1) * Math.pow(10, 30 - i);
    histogram.push({ energy, count });
  }
  return histogram;
}
