"""
Enhanced ML Models Package for Solar Flare Analysis

This package contains state-of-the-art machine learning models for solar flare
detection, classification, and analysis including:
- Traditional enhanced models with nanoflare detection
- Monte Carlo models with uncertainty quantification
- Transformer-based models with attention mechanisms
- Graph Neural Networks for complex relationship modeling
- Self-supervised learning models
- Flare decomposition models
"""

# Traditional enhanced models
from .enhanced_flare_analysis import NanoflareDetector, FlareEnergyAnalyzer
from .flare_decomposition import FlareDecompositionModel

# Advanced Monte Carlo models
from .monte_carlo_enhanced_model import MonteCarloSolarFlareModel

# Modern deep learning models
from .transformer_flare_model import (
    TransformerFlareModel,
    ConvolutionalTransformerModel
)

# Graph-based models
from .graph_neural_model import (
    GraphNeuralFlareModel,
    HybridGraphTransformerModel,
    GraphAttentionLayer
)

# Self-supervised learning models
from .self_supervised_models import (
    ContrastiveLearningModel
)

__all__ = [
    # Traditional models
    'NanoflareDetector',
    'FlareEnergyAnalyzer', 
    'FlareDecompositionModel',
    
    # Monte Carlo models
    'MonteCarloSolarFlareModel',
    
    # Transformer models
    'TransformerFlareModel',
    'ConvolutionalTransformerModel',
    
    # Graph models
    'GraphNeuralFlareModel',
    'HybridGraphTransformerModel',
    'GraphAttentionLayer',
    
    # Self-supervised models
    'ContrastiveLearningModel'
]

# Model categories for easy access
TRADITIONAL_MODELS = [
    'NanoflareDetector',
    'FlareEnergyAnalyzer',
    'FlareDecompositionModel'
]

ADVANCED_MODELS = [
    'MonteCarloSolarFlareModel',
    'TransformerFlareModel',
    'ConvolutionalTransformerModel',
    'GraphNeuralFlareModel',
    'HybridGraphTransformerModel'
]

SELF_SUPERVISED_MODELS = [
    'ContrastiveLearningModel'
]

def get_model_by_name(model_name: str):
    """
    Get a model class by its name
    
    Parameters
    ----------
    model_name : str
        Name of the model class
        
    Returns
    -------
    class
        The requested model class
        
    Raises
    ------
    ValueError
        If model name is not found
    """
    import sys
    current_module = sys.modules[__name__]
    
    if hasattr(current_module, model_name):
        return getattr(current_module, model_name)
    else:
        raise ValueError(f"Model '{model_name}' not found. Available models: {__all__}")

def list_available_models():
    """
    List all available model classes
    
    Returns
    -------
    dict
        Dictionary with model categories and their models
    """
    return {
        'traditional': TRADITIONAL_MODELS,
        'advanced': ADVANCED_MODELS,
        'self_supervised': SELF_SUPERVISED_MODELS,
        'all': __all__
    }
