
import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:5000';

interface BayesianAnalysisRequest {
  data: number[];
  n_chains?: number;
  n_iterations?: number;
  target_acceptance?: number;
  regularization_strength?: number;
}

interface BayesianResults {
  posterior_samples: {
    weights: number[][];
    bias: number[];
    variance: number[];
  };
  credible_intervals: {
    lower_95: number[];
    upper_95: number[];
    lower_68: number[];
    upper_68: number[];
  };
  model_comparison: {
    waic: number;
    loo: number;
    r_hat: number[];
    effective_sample_size: number[];
  };
  uncertainty_metrics: {
    epistemic_uncertainty: number[];
    aleatoric_uncertainty: number[];
    total_uncertainty: number[];
    prediction_variance: number[];
  };
  convergence_diagnostics: {
    trace_plots: string[];
    autocorrelation: number[];
    split_r_hat: number[];
    bulk_ess: number[];
    tail_ess: number[];
  };
  predictive_distribution: {
    mean: number[];
    std: number[];
    quantiles: {
      q05: number[];
      q25: number[];
      q50: number[];
      q75: number[];
      q95: number[];
    };
  };
  model_evidence: {
    log_marginal_likelihood: number;
    bayes_factor: number;
    bridge_sampling_estimate: number;
  };
}

// Generate comprehensive mock Bayesian analysis data
function generateMockBayesianResults(): BayesianResults {
  const n_samples = 1000;
  const n_features = 10;
  
  // Generate posterior samples
  const weights = Array.from({ length: n_samples }, () =>
    Array.from({ length: n_features }, () => Math.random() * 2 - 1)
  );
  
  const bias = Array.from({ length: n_samples }, () => Math.random() * 0.5 - 0.25);
  const variance = Array.from({ length: n_samples }, () => Math.random() * 0.1 + 0.01);
  
  // Generate credible intervals
  const generateCredibleIntervals = () => {
    const values = Array.from({ length: n_features }, () => Math.random() * 2 - 1);
    return {
      lower_95: values.map(v => v - 1.96 * 0.1),
      upper_95: values.map(v => v + 1.96 * 0.1),
      lower_68: values.map(v => v - 1.0 * 0.1),
      upper_68: values.map(v => v + 1.0 * 0.1),
    };
  };
  
  // Generate uncertainty metrics
  const uncertainty_metrics = {
    epistemic_uncertainty: Array.from({ length: n_features }, () => Math.random() * 0.1 + 0.01),
    aleatoric_uncertainty: Array.from({ length: n_features }, () => Math.random() * 0.05 + 0.005),
    total_uncertainty: Array.from({ length: n_features }, () => Math.random() * 0.12 + 0.02),
    prediction_variance: Array.from({ length: n_features }, () => Math.random() * 0.08 + 0.01),
  };
  
  // Generate convergence diagnostics
  const convergence_diagnostics = {
    trace_plots: [
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
    ],
    autocorrelation: Array.from({ length: n_features }, () => Math.random() * 0.1),
    split_r_hat: Array.from({ length: n_features }, () => 1 + Math.random() * 0.05),
    bulk_ess: Array.from({ length: n_features }, () => 800 + Math.random() * 400),
    tail_ess: Array.from({ length: n_features }, () => 700 + Math.random() * 500),
  };
  
  // Generate predictive distribution
  const predictive_distribution = {
    mean: Array.from({ length: n_features }, () => Math.random() * 2 - 1),
    std: Array.from({ length: n_features }, () => Math.random() * 0.3 + 0.05),
    quantiles: {
      q05: Array.from({ length: n_features }, () => Math.random() * 0.5 - 1.5),
      q25: Array.from({ length: n_features }, () => Math.random() * 0.5 - 0.75),
      q50: Array.from({ length: n_features }, () => Math.random() * 0.5 - 0.25),
      q75: Array.from({ length: n_features }, () => Math.random() * 0.5 + 0.25),
      q95: Array.from({ length: n_features }, () => Math.random() * 0.5 + 1.0),
    },
  };
  
  return {
    posterior_samples: {
      weights,
      bias,
      variance,
    },
    credible_intervals: generateCredibleIntervals(),
    model_comparison: {
      waic: -45.2 + Math.random() * 10,
      loo: -47.8 + Math.random() * 12,
      r_hat: Array.from({ length: n_features }, () => 1 + Math.random() * 0.02),
      effective_sample_size: Array.from({ length: n_features }, () => 850 + Math.random() * 300),
    },
    uncertainty_metrics,
    convergence_diagnostics,
    predictive_distribution,
    model_evidence: {
      log_marginal_likelihood: -52.3 + Math.random() * 8,
      bayes_factor: 2.1 + Math.random() * 3,
      bridge_sampling_estimate: -51.7 + Math.random() * 9,
    },
  };
}

export async function POST(request: NextRequest) {
  try {
    const body: BayesianAnalysisRequest = await request.json();
    
    const requestData = {
      data: body.data,
      n_chains: body.n_chains || 4,
      n_iterations: body.n_iterations || 2000,
      target_acceptance: body.target_acceptance || 0.8,
      regularization_strength: body.regularization_strength || 0.01,
    };
    
    try {
      // Try to connect to Python backend first
      const pythonResponse = await fetch(`${PYTHON_API_URL}/api/bayesian/analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: AbortSignal.timeout(30000), // 30 second timeout
      });
      
      if (pythonResponse.ok) {
        const pythonData = await pythonResponse.json();
        return NextResponse.json({
          success: true,
          source: 'python_backend',
          results: pythonData,
          timestamp: new Date().toISOString(),
        });
      }
    } catch (pythonError) {
      console.log('Python backend unavailable, using mock data:', pythonError);
    }
    
    // Fallback to mock data for development
    const mockResults = generateMockBayesianResults();
    
    return NextResponse.json({
      success: true,
      source: 'mock_data',
      results: mockResults,
      timestamp: new Date().toISOString(),
      message: 'Using simulated Bayesian analysis results for development',
    });
    
  } catch (error) {
    console.error('Bayesian analysis error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to perform Bayesian analysis',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    // Return example parameters and capabilities
    return NextResponse.json({
      success: true,
      capabilities: {
        mcmc_methods: ['NUTS', 'HMC', 'Gibbs'],
        model_types: ['linear_regression', 'logistic_regression', 'neural_network'],
        diagnostics: ['r_hat', 'effective_sample_size', 'autocorrelation', 'trace_plots'],
        uncertainty_types: ['epistemic', 'aleatoric', 'total'],
        credible_intervals: ['68%', '95%', 'custom'],
        model_comparison: ['WAIC', 'LOO', 'Bayes_Factor'],
      },
      default_parameters: {
        n_chains: 4,
        n_iterations: 2000,
        n_warmup: 1000,
        target_acceptance: 0.8,
        regularization_strength: 0.01,
      },
      timestamp: new Date().toISOString(),
    });
    
  } catch (error) {
    console.error('Bayesian API info error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to get Bayesian analysis info' 
      },
      { status: 500 }
    );
  }
}
