"""
Initialization file for the evaluation module.

This module provides utilities for evaluating solar flare detection 
and ML model performance.
"""

from .model_evaluation import (
    evaluate_flare_reconstruction,
    evaluate_flare_segmentation,
    plot_learning_curves,
    plot_flare_segmentation_results,
    calculate_flare_separation_metrics
)
