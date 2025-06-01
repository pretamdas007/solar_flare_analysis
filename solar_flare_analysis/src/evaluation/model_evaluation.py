"""
Model evaluation utilities for assessing ML model performance.

This module provides functions to evaluate machine learning models 
for solar flare separation and detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    mean_squared_error, r2_score, precision_recall_curve, 
    roc_curve, auc, f1_score, confusion_matrix
)


def evaluate_flare_reconstruction(original, reconstructed, individual=None):
    """
    Evaluate how well a reconstructed signal matches the original.
    
    Parameters
    ----------
    original : numpy.ndarray
        Original signal
    reconstructed : numpy.ndarray
        Reconstructed signal
    individual : numpy.ndarray, optional
        Individual components of the reconstructed signal
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Mean squared error
    metrics['mse'] = mean_squared_error(original, reconstructed)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Normalized RMSE
    if np.max(original) - np.min(original) > 0:
        metrics['nrmse'] = metrics['rmse'] / (np.max(original) - np.min(original))
    else:
        metrics['nrmse'] = 0
    
    # R² score
    metrics['r2'] = r2_score(original, reconstructed)
    
    # Mean absolute error
    metrics['mae'] = np.mean(np.abs(original - reconstructed))
    
    # Peak error
    original_peak = np.max(original)
    reconstructed_peak = np.max(reconstructed)
    metrics['peak_error'] = abs(original_peak - reconstructed_peak)
    metrics['relative_peak_error'] = metrics['peak_error'] / original_peak if original_peak > 0 else 0
    
    # Energy conservation    original_energy = np.trapezoid(original)
    reconstructed_energy = np.trapezoid(reconstructed)
    metrics['energy_error'] = abs(original_energy - reconstructed_energy)
    metrics['relative_energy_error'] = metrics['energy_error'] / original_energy if original_energy > 0 else 0
    
    # Component energy distribution
    if individual is not None:
        component_energies = []
        for i in range(individual.shape[1]):
            component_energies.append(np.trapezoid(individual[:, i]))
        metrics['component_energies'] = component_energies
        metrics['energy_distribution'] = [e / reconstructed_energy if reconstructed_energy > 0 else 0 
                                          for e in component_energies]
    
    return metrics


def evaluate_flare_segmentation(model, test_data, ground_truth=None, threshold=0.5):
    """
    Evaluate model performance for flare segmentation.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained model for flare segmentation
    test_data : numpy.ndarray
        Test data for evaluation
    ground_truth : numpy.ndarray, optional
        Ground truth segmentations
    threshold : float, optional
        Threshold for binary segmentation
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics
    """
    # Make predictions
    predictions = model.predict(test_data)
    
    # Initialize metrics
    metrics = {}
    
    if ground_truth is not None:
        # Convert predictions to binary using threshold
        binary_preds = (predictions > threshold).astype(int)
        binary_truth = (ground_truth > threshold).astype(int)
        
        # Reshape for easier metric calculation
        n_samples = binary_preds.shape[0]
        sequence_length = binary_preds.shape[1]
        n_components = binary_preds.shape[2]
        
        binary_preds_flat = binary_preds.reshape(-1)
        binary_truth_flat = binary_truth.reshape(-1)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(binary_truth_flat, binary_preds_flat).ravel()
        
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) \
                        if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(binary_truth_flat, predictions.reshape(-1))
        metrics['auc'] = auc(fpr, tpr)
        
        # IoU (Intersection over Union) by component
        iou_scores = []
        for i in range(n_components):
            component_preds = binary_preds[:, :, i].reshape(-1)
            component_truth = binary_truth[:, :, i].reshape(-1)
            
            intersection = np.logical_and(component_preds, component_truth).sum()
            union = np.logical_or(component_preds, component_truth).sum()
            
            iou = intersection / union if union > 0 else 0
            iou_scores.append(iou)
        
        metrics['iou_scores'] = iou_scores
        metrics['mean_iou'] = np.mean(iou_scores)
        
        # MSE for continuous predictions
        metrics['mse'] = mean_squared_error(ground_truth.reshape(-1), predictions.reshape(-1))
        metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Calculate metrics that don't require ground truth
    metrics['reconstruction_quality'] = {}
    
    for i in range(len(test_data)):
        # Original signal
        original = test_data[i, :, 0]
        
        # Sum of components
        components = predictions[i]
        reconstructed = np.sum(components, axis=1)
        
        # Evaluate reconstruction
        sample_metrics = evaluate_flare_reconstruction(original, reconstructed, components)
        
        # Aggregate metrics
        for key, value in sample_metrics.items():
            if key not in metrics['reconstruction_quality']:
                metrics['reconstruction_quality'][key] = []
            metrics['reconstruction_quality'][key].append(value)
    
    # Average reconstruction metrics
    for key in metrics['reconstruction_quality']:
        if key != 'component_energies' and key != 'energy_distribution':
            metrics['reconstruction_quality'][key] = np.mean(metrics['reconstruction_quality'][key])
    
    return metrics


def plot_learning_curves(history):
    """
    Plot learning curves from model training history.
    
    Parameters
    ----------
    history : tensorflow.keras.callbacks.History
        History object from model training
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with learning curves
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training and validation loss
    ax1.plot(history.history['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], 'r-', label='Validation Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot training and validation metrics
    metric_keys = [key for key in history.history.keys() 
                  if key not in ['loss', 'val_loss'] and not key.startswith('val_')]
    
    if metric_keys:
        for key in metric_keys:
            ax2.plot(history.history[key], 'b-', label=f'Training {key}')
            val_key = f'val_{key}'
            if val_key in history.history:
                ax2.plot(history.history[val_key], 'r-', label=f'Validation {key}')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Training and Validation Metrics')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
    else:
        ax2.set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_flare_segmentation_results(test_data, predictions, indices=None):
    """
    Plot the results of flare segmentation.
    
    Parameters
    ----------
    test_data : numpy.ndarray
        Test data
    predictions : numpy.ndarray
        Model predictions
    indices : list, optional
        List of indices to plot; if None, first 4 samples are plotted
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with segmentation results
    """
    if indices is None:
        indices = range(min(4, len(test_data)))
    
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Plot original signal
        ax.plot(test_data[idx, :, 0], 'k-', label='Original', alpha=0.7)
        
        # Plot components
        components = predictions[idx]
        combined = np.sum(components, axis=1)
        
        # Plot each component
        for j in range(components.shape[1]):
            if np.max(components[:, j]) > 0.05 * np.max(test_data[idx, :, 0]):
                ax.plot(components[:, j], '--', label=f'Component {j+1}')
        
        # Plot combined reconstruction
        ax.plot(combined, 'r-', label='Combined', alpha=0.7)
        
        ax.set_title(f'Sample {idx}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    
    return fig


def calculate_flare_separation_metrics(true_individual, predicted_individual, threshold=0.2):
    """
    Calculate metrics specific to flare separation.
    
    Parameters
    ----------
    true_individual : numpy.ndarray
        Ground truth individual flares (sequence_length, n_components)
    predicted_individual : numpy.ndarray
        Predicted individual flares (sequence_length, n_components)
    threshold : float, optional
        Threshold for significant flare contribution
        
    Returns
    -------
    dict
        Dictionary with flare separation metrics
    """
    metrics = {}
    
    # Number of components in ground truth and prediction
    n_components_true = true_individual.shape[1]
    n_components_pred = predicted_individual.shape[1]
    
    # Count significant components (with max value above threshold)
    significant_true = []
    for i in range(n_components_true):
        if np.max(true_individual[:, i]) > threshold * np.max(true_individual):
            significant_true.append(i)
    
    significant_pred = []
    for i in range(n_components_pred):
        if np.max(predicted_individual[:, i]) > threshold * np.max(predicted_individual):
            significant_pred.append(i)
    
    metrics['n_significant_true'] = len(significant_true)
    metrics['n_significant_pred'] = len(significant_pred)
    
    # Count correct predictions (number of components matches)
    metrics['correct_component_count'] = len(significant_true) == len(significant_pred)
    
    # Component-wise metrics
    component_metrics = []
    
    # Match predicted components to ground truth components
    if len(significant_true) > 0 and len(significant_pred) > 0:
        # Create similarity matrix
        similarity_matrix = np.zeros((len(significant_true), len(significant_pred)))
        
        for i, true_idx in enumerate(significant_true):
            for j, pred_idx in enumerate(significant_pred):
                true_comp = true_individual[:, true_idx]
                pred_comp = predicted_individual[:, pred_idx]
                
                # Calculate correlation
                correlation = np.corrcoef(true_comp, pred_comp)[0, 1]
                similarity_matrix[i, j] = correlation
        
        # Greedy matching based on highest similarity
        matched_true = set()
        matched_pred = set()
        matches = []
        
        # Sort similarities in descending order
        flat_indices = np.argsort(similarity_matrix.flatten())[::-1]
        
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, similarity_matrix.shape)
            
            if i not in matched_true and j not in matched_pred:
                matched_true.add(i)
                matched_pred.add(j)
                
                true_comp = true_individual[:, significant_true[i]]
                pred_comp = predicted_individual[:, significant_pred[j]]
                
                # Calculate metrics for this match
                correlation = similarity_matrix[i, j]
                mse = mean_squared_error(true_comp, pred_comp)
                rmse = np.sqrt(mse)
                  # Calculate energy
                true_energy = np.trapezoid(true_comp)
                pred_energy = np.trapezoid(pred_comp)
                energy_ratio = pred_energy / true_energy if true_energy > 0 else float('inf')
                
                # Peak metrics
                true_peak = np.max(true_comp)
                pred_peak = np.max(pred_comp)
                peak_ratio = pred_peak / true_peak if true_peak > 0 else float('inf')
                
                # Add metrics
                component_metrics.append({
                    'true_idx': significant_true[i],
                    'pred_idx': significant_pred[j],
                    'correlation': correlation,
                    'mse': mse,
                    'rmse': rmse,
                    'energy_ratio': energy_ratio,
                    'peak_ratio': peak_ratio
                })
        
        metrics['matched_components'] = len(matches)
        metrics['unmatched_true'] = len(significant_true) - len(matches)
        metrics['unmatched_pred'] = len(significant_pred) - len(matches)
    
    # Calculate average metrics across matched components
    if component_metrics:
        metrics['avg_component_correlation'] = np.mean([m['correlation'] for m in component_metrics])
        metrics['avg_component_rmse'] = np.mean([m['rmse'] for m in component_metrics])
        metrics['avg_energy_ratio'] = np.mean([m['energy_ratio'] for m in component_metrics 
                                              if m['energy_ratio'] != float('inf')])
        metrics['avg_peak_ratio'] = np.mean([m['peak_ratio'] for m in component_metrics
                                            if m['peak_ratio'] != float('inf')])
    
    metrics['component_metrics'] = component_metrics
    
    return metrics
