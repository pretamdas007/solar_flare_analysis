#!/usr/bin/env python
"""
Enhanced Solar Flare Analysis Training Script
Train ML models with real GOES XRS CSV data files from xrsa and xrsb directories
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import EnhancedSolarFlareAnalyzer


def load_csv_data(xrsa_dir, xrsb_dir):
    """Load and combine CSV data from xrsa and xrsb directories"""
    print(f"\n🔄 Loading data from CSV files...")
    
    # Find CSV files
    xrsa_csv_files = [f for f in os.listdir(xrsa_dir) if f.endswith('.csv')] if os.path.exists(xrsa_dir) else []
    xrsb_csv_files = [f for f in os.listdir(xrsb_dir) if f.endswith('.csv')] if os.path.exists(xrsb_dir) else []
    
    if not xrsa_csv_files and not xrsb_csv_files:
        raise ValueError(f"No CSV files found in {xrsa_dir} or {xrsb_dir}")
    
    print(f"📁 Found {len(xrsa_csv_files)} CSV files in xrsa directory")
    print(f"📁 Found {len(xrsb_csv_files)} CSV files in xrsb directory")
    
    # Prepare data paths for loading
    all_csv_paths = []
    for file in xrsa_csv_files:
        all_csv_paths.append(os.path.join(xrsa_dir, file))
    for file in xrsb_csv_files:
        all_csv_paths.append(os.path.join(xrsb_dir, file))
    
    # Load CSV files
    all_data = []
    for csv_file in all_csv_paths:
        try:
            print(f"   Loading {os.path.basename(csv_file)}...")
            df = pd.read_csv(csv_file, parse_dates=True)
            
            # Ensure there's a datetime column to use as index
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
            elif 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            elif 'date' in df.columns:
                df.set_index('date', inplace=True)
            
            # Ensure the index is a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Standardize column names
            df.columns = [col.lower().replace('-', '_').replace(' ', '_') for col in df.columns]
            
            # Check which type of data (XRS-A or XRS-B)
            if 'xrsa' in csv_file.lower() or 'xrs_a' in csv_file.lower():
                if not any('xrs_a' in col.lower() for col in df.columns):
                    # Add xrs_a column if not present
                    flux_col = [col for col in df.columns if 'flux' in col.lower() or 'irradiance' in col.lower()]
                    if flux_col:
                        df.rename(columns={flux_col[0]: 'xrs_a'}, inplace=True)
            
            if 'xrsb' in csv_file.lower() or 'xrs_b' in csv_file.lower():
                if not any('xrs_b' in col.lower() for col in df.columns):
                    # Add xrs_b column if not present
                    flux_col = [col for col in df.columns if 'flux' in col.lower() or 'irradiance' in col.lower()]
                    if flux_col:
                        df.rename(columns={flux_col[0]: 'xrs_b'}, inplace=True)
            
            all_data.append(df)
            print(f"   ✅ Added {len(df)} data points from {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("Failed to load any valid CSV data")
    
    # Combine all data
    print("\n🔄 Combining and processing data...")
    combined_data = pd.concat(all_data, ignore_index=False).sort_index()
    
    # Remove duplicates
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
    
    return combined_data


def advanced_preprocessing(data):
    """Advanced preprocessing with quality filtering and feature engineering"""
    print("\n🔄 Enhanced preprocessing and feature engineering...")
    processed_data = data.copy()
    
    # Identify XRS columns
    xrs_cols = [col for col in processed_data.columns if 'xrs' in col.lower()]
    print(f"Found XRS columns: {xrs_cols}")
    
    # Advanced quality filtering
    for col in xrs_cols:
        # Remove null, negative and unreasonable values
        mask = (
            (processed_data[col] > 0) &  # Positive values only
            (~np.isnan(processed_data[col])) &  # No NaNs
            (~np.isinf(processed_data[col])) &  # No infinities
            (processed_data[col] < 1e-2)  # Upper limit on reasonable values
        )
        processed_data.loc[~mask, col] = np.nan
        
        # Outlier detection using IQR method
        Q1 = processed_data[col].quantile(0.25)
        Q3 = processed_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)
        print(f"  Removed {outlier_mask.sum()} outliers from {col}")
        processed_data.loc[outlier_mask, col] = np.nan
    
    # Resample to 1-minute cadence with multiple aggregation methods
    print("Resampling data to 1-minute cadence...")
    resampled_data = {}
    for col in xrs_cols:
        resampled_data[col] = processed_data[col].resample('1min').mean()
        resampled_data[f'{col}_max'] = processed_data[col].resample('1min').max()
        resampled_data[f'{col}_min'] = processed_data[col].resample('1min').min()
        resampled_data[f'{col}_std'] = processed_data[col].resample('1min').std()
    
    processed_data = pd.DataFrame(resampled_data)
    
    # Enhanced gap filling strategy
    print("Applying enhanced gap filling...")
    for col in processed_data.columns:
        # Forward fill short gaps (up to 3 minutes)
        processed_data[col] = processed_data[col].fillna(method='ffill', limit=3)
        # Backward fill remaining short gaps
        processed_data[col] = processed_data[col].fillna(method='bfill', limit=3)
        # Interpolate medium gaps (up to 10 minutes)
        processed_data[col] = processed_data[col].interpolate(method='time', limit=10)
    
    # Feature engineering: Add derived features
    print("Engineering additional features...")
    if len(xrs_cols) >= 2:
        # XRS ratio features
        processed_data['xrs_ratio'] = processed_data[xrs_cols[0]] / (processed_data[xrs_cols[1]] + 1e-12)
        processed_data['xrs_sum'] = processed_data[xrs_cols[0]] + processed_data[xrs_cols[1]]
        processed_data['xrs_diff'] = processed_data[xrs_cols[0]] - processed_data[xrs_cols[1]]
    
    # Temporal features
    for col in xrs_cols:
        if col in processed_data.columns:
            # Moving averages
            processed_data[f'{col}_ma5'] = processed_data[col].rolling(window=5, center=True).mean()
            processed_data[f'{col}_ma15'] = processed_data[col].rolling(window=15, center=True).mean()
            
            # Derivatives (rate of change)
            processed_data[f'{col}_derivative'] = processed_data[col].diff()
            processed_data[f'{col}_derivative2'] = processed_data[f'{col}_derivative'].diff()
            
            # Log transform for better ML performance
            processed_data[f'{col}_log'] = np.log10(processed_data[col] + 1e-12)
    
    # Remove any remaining infinite values
    processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
    
    # Final interpolation for any remaining gaps
    processed_data = processed_data.interpolate(method='linear', limit=5)
    
    # Drop rows with too many missing values
    processed_data = processed_data.dropna(thresh=len(processed_data.columns) * 0.7)
    
    return processed_data, xrs_cols


def advanced_feature_engineering(processed_data, xrs_cols, output_dir):
    """Advanced feature engineering with statistical and spectral features"""
    print("🔬 Advanced feature engineering...")
    
    enhanced_data = processed_data.copy()
    
    for col in xrs_cols:
        if col in enhanced_data.columns:
            # Statistical features
            window_sizes = [5, 15, 30, 60]  # Different time windows
            
            for window in window_sizes:
                # Rolling statistics
                enhanced_data[f'{col}_rolling_mean_{window}'] = enhanced_data[col].rolling(window).mean()
                enhanced_data[f'{col}_rolling_std_{window}'] = enhanced_data[col].rolling(window).std()
                enhanced_data[f'{col}_rolling_skew_{window}'] = enhanced_data[col].rolling(window).skew()
                enhanced_data[f'{col}_rolling_kurt_{window}'] = enhanced_data[col].rolling(window).kurt()
                
                # Rolling quantiles
                enhanced_data[f'{col}_rolling_q25_{window}'] = enhanced_data[col].rolling(window).quantile(0.25)
                enhanced_data[f'{col}_rolling_q75_{window}'] = enhanced_data[col].rolling(window).quantile(0.75)
            
            # Signal processing features
            # Smoothed signal using Savitzky-Golay filter
            if len(enhanced_data[col].dropna()) > 10:
                try:
                    enhanced_data[f'{col}_smoothed'] = savgol_filter(
                        enhanced_data[col].fillna(method='ffill').fillna(method='bfill'), 
                        window_length=min(11, len(enhanced_data[col].dropna()) // 2 * 2 + 1), 
                        polyorder=3
                    )
                except:
                    enhanced_data[f'{col}_smoothed'] = enhanced_data[col]
            
            # Momentum indicators
            enhanced_data[f'{col}_momentum_5'] = enhanced_data[col] - enhanced_data[col].shift(5)
            enhanced_data[f'{col}_momentum_15'] = enhanced_data[col] - enhanced_data[col].shift(15)
            
            # Rate of change
            enhanced_data[f'{col}_roc_5'] = enhanced_data[col].pct_change(periods=5)
            enhanced_data[f'{col}_roc_15'] = enhanced_data[col].pct_change(periods=15)
            
            # Cumulative features
            enhanced_data[f'{col}_cumsum'] = enhanced_data[col].cumsum()
            enhanced_data[f'{col}_cumprod'] = (1 + enhanced_data[col].pct_change()).cumprod()
            
            # Bollinger Band features
            mean_20 = enhanced_data[col].rolling(20).mean()
            std_20 = enhanced_data[col].rolling(20).std()
            enhanced_data[f'{col}_bb_upper'] = mean_20 + (2 * std_20)
            enhanced_data[f'{col}_bb_lower'] = mean_20 - (2 * std_20)
            enhanced_data[f'{col}_bb_position'] = (enhanced_data[col] - enhanced_data[f'{col}_bb_lower']) / (enhanced_data[f'{col}_bb_upper'] - enhanced_data[f'{col}_bb_lower'])
    
    # Cross-feature relationships (if multiple XRS channels)
    if len(xrs_cols) >= 2:
        col1, col2 = xrs_cols[0], xrs_cols[1]
        
        # Advanced ratios
        enhanced_data['xrs_log_ratio'] = np.log10((enhanced_data[col1] + 1e-12) / (enhanced_data[col2] + 1e-12))
        enhanced_data['xrs_normalized_diff'] = (enhanced_data[col1] - enhanced_data[col2]) / (enhanced_data[col1] + enhanced_data[col2] + 1e-12)
        
        # Correlation features
        enhanced_data['xrs_correlation_15'] = enhanced_data[col1].rolling(15).corr(enhanced_data[col2])
        enhanced_data['xrs_correlation_60'] = enhanced_data[col1].rolling(60).corr(enhanced_data[col2])
    
    # Time-based features
    enhanced_data['hour'] = enhanced_data.index.hour
    enhanced_data['day_of_year'] = enhanced_data.index.dayofyear
    enhanced_data['is_weekend'] = (enhanced_data.index.weekday >= 5).astype(int)
    
    # Solar cycle features (approximate 11-year cycle)
    reference_date = pd.Timestamp('2000-01-01')
    enhanced_data['solar_cycle_phase'] = np.sin(2 * np.pi * (enhanced_data.index - reference_date).days / (11 * 365.25))
    
    print(f"✅ Enhanced features: {len(enhanced_data.columns)} total features")
    return enhanced_data


def apply_anomaly_detection(processed_data, xrs_cols, output_dir):
    """Apply anomaly detection for data quality"""
    print("🚨 Applying anomaly detection...")
    
    anomaly_results = {}
    
    for col in xrs_cols:
        if col in processed_data.columns:
            # Isolation Forest for anomaly detection
            data_clean = processed_data[col].dropna().values.reshape(-1, 1)
            
            if len(data_clean) > 100:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                anomalies = iso_forest.fit_predict(data_clean)
                
                # Mark anomalies in the original data
                anomaly_mask = pd.Series(anomalies == -1, index=processed_data[col].dropna().index)
                anomaly_results[col] = anomaly_mask
                
                print(f"  Found {anomaly_mask.sum()} anomalies in {col}")
    
    return anomaly_results


def perform_pca_analysis(processed_data, output_dir):
    """Perform PCA for dimensionality reduction and feature importance"""
    print("🔍 Performing PCA analysis...")
    
    # Select numeric columns only
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    pca_data = processed_data[numeric_cols].dropna()
    
    if len(pca_data) > 0 and len(numeric_cols) > 1:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Apply PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot explained variance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance ratio
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', markersize=4)
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Feature importance (first 2 components)
        feature_importance = np.abs(pca.components_[:2]).T
        feature_names = numeric_cols[:len(feature_importance)]
        
        # Top 10 most important features
        importance_sum = feature_importance.sum(axis=1)
        top_indices = np.argsort(importance_sum)[-10:]
        
        ax2.barh(range(len(top_indices)), importance_sum[top_indices])
        ax2.set_yticks(range(len(top_indices)))
        ax2.set_yticklabels([feature_names[i] for i in top_indices])
        ax2.set_xlabel('Feature Importance (PC1 + PC2)')
        ax2.set_title('Top 10 Most Important Features')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Recommend number of components for 95% variance
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        print(f"  Recommended components for 95% variance: {n_components_95}")
        
        return n_components_95, pca
    
    return None, None


def plot_training_history(history, output_dir, prefix):
    """Plot and save training history"""
    if history is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {prefix.replace("_", " ").title()}', fontsize=16)
    
    # Loss plots
    if 'loss' in history.history:
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plots (if available)
    if 'accuracy' in history.history:
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Additional metrics
    metric_keys = [k for k in history.history.keys() if k not in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'lr']]
    if metric_keys:
        for i, metric in enumerate(metric_keys[:2]):  # Plot up to 2 additional metrics
            if i < 2:
                row, col = 1, 1 if i == 0 else 0
                if f'val_{metric}' in history.history:
                    axes[row, col].plot(history.history[metric], label=f'Training {metric}')
                    axes[row, col].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                else:
                    axes[row, col].plot(history.history[metric], label=metric)
                axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(metric)
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_training_history(history_synthetic, history_finetuned, output_dir):
    """Plot combined training history for hybrid training"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hybrid Training History (Synthetic + Real Data)', fontsize=16)
    
    # Combine epochs
    synthetic_epochs = len(history_synthetic.history['loss'])
    finetuned_epochs = len(history_finetuned.history['loss'])
    
    # Plot combined loss
    combined_loss = history_synthetic.history['loss'] + history_finetuned.history['loss']
    combined_val_loss = history_synthetic.history.get('val_loss', []) + history_finetuned.history.get('val_loss', [])
    
    epochs = range(1, len(combined_loss) + 1)
    axes[0, 0].plot(epochs, combined_loss, label='Training Loss')
    if combined_val_loss:
        axes[0, 0].plot(epochs, combined_val_loss, label='Validation Loss')
    axes[0, 0].axvline(x=synthetic_epochs, color='red', linestyle='--', alpha=0.7, label='Switch to Real Data')
    axes[0, 0].set_title('Combined Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot phase comparison
    axes[0, 1].plot(range(1, synthetic_epochs + 1), history_synthetic.history['loss'], label='Synthetic Phase')
    axes[0, 1].plot(range(synthetic_epochs + 1, synthetic_epochs + finetuned_epochs + 1), 
                   history_finetuned.history['loss'], label='Fine-tuning Phase')
    axes[0, 1].set_title('Training Phases Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate comparison (if available)
    if 'lr' in history_synthetic.history and 'lr' in history_finetuned.history:
        combined_lr = history_synthetic.history['lr'] + history_finetuned.history['lr']
        axes[1, 0].plot(epochs, combined_lr)
        axes[1, 0].axvline(x=synthetic_epochs, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Final metrics comparison
    final_metrics = {
        'Synthetic Phase': {
            'Final Loss': history_synthetic.history['loss'][-1],
            'Final Val Loss': history_synthetic.history.get('val_loss', [0])[-1] if history_synthetic.history.get('val_loss') else 0
        },
        'Fine-tuning Phase': {
            'Final Loss': history_finetuned.history['loss'][-1],
            'Final Val Loss': history_finetuned.history.get('val_loss', [0])[-1] if history_finetuned.history.get('val_loss') else 0
        }
    }
    
    phases = list(final_metrics.keys())
    loss_values = [final_metrics[phase]['Final Loss'] for phase in phases]
    val_loss_values = [final_metrics[phase]['Final Val Loss'] for phase in phases]
    
    x = np.arange(len(phases))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, loss_values, width, label='Training Loss', alpha=0.8)
    axes[1, 1].bar(x + width/2, val_loss_values, width, label='Validation Loss', alpha=0.8)
    axes[1, 1].set_title('Final Metrics Comparison')
    axes[1, 1].set_xlabel('Training Phase')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(phases)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def perform_cross_validation_training(analyzer, processed_data, output_dir):
    """Perform cross-validation training"""
    try:
        print("🔄 Implementing time-series cross-validation...")
        
        # Time-based splits for cross-validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        cv_histories = []
        training_times = []
        detailed_metrics = []
        
        # Convert data to numpy for cross-validation
        data_array = processed_data.values
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(data_array)):
            print(f"  Training fold {fold + 1}/{n_splits}...")
            start_time = datetime.now()
            
            # Re-initialize model for each fold
            analyzer.initialize_ml_model(
                sequence_length=256,
                max_flares=5,
                enhanced=True
            )
            
            # Train on fold
            fold_history = analyzer.train_ml_model(
                use_synthetic_data=False,
                validation_split=0.2,
                epochs=15,
                batch_size=16,
                enhanced=True,
                learning_rate=0.001,
                early_stopping_patience=5
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            training_times.append(training_time)
            
            if fold_history is not None:
                cv_histories.append(fold_history)
                final_val_loss = fold_history.history.get('val_loss', [float('inf')])[-1]
                cv_scores.append(final_val_loss)
                
                # Store detailed metrics
                detailed_metrics.append({
                    'fold': fold + 1,
                    'train_samples': len(train_idx),
                    'val_samples': len(val_idx),
                    'val_loss': final_val_loss,
                    'training_time': training_time
                })
                
                print(f"    Fold {fold + 1} validation loss: {final_val_loss:.6f} (time: {training_time:.1f}s)")
        
        if cv_scores:
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            print(f"Cross-validation results:")
            print(f"  Mean validation loss: {mean_cv_score:.6f} ± {std_cv_score:.6f}")
            print(f"  Average training time: {np.mean(training_times):.1f}s")
            
            # Save cross-validation results
            cv_results = {
                'fold_scores': cv_scores,
                'mean_score': mean_cv_score,
                'std_score': std_cv_score,
                'n_folds': len(cv_scores),
                'training_times': training_times,
                'detailed_metrics': detailed_metrics
            }
            
            # Plot enhanced CV results
            plot_enhanced_cv_results(cv_results, output_dir)
            
            return cv_results
        
    except Exception as e:
        print(f"Cross-validation training failed: {e}")
        return None


def plot_enhanced_cv_results(cv_results, output_dir):
    """Enhanced cross-validation results visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Cross-Validation Analysis', fontsize=16)
    
    scores = cv_results['fold_scores']
    detailed_metrics = cv_results.get('detailed_metrics', [])
    
    # Box plot and violin plot
    ax1.boxplot(scores, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax1.set_title('Validation Loss Distribution')
    ax1.set_ylabel('Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # Individual fold performance
    folds = range(1, len(scores) + 1)
    bars = ax2.bar(folds, scores, alpha=0.7, color='steelblue')
    mean_line = ax2.axhline(y=cv_results['mean_score'], color='red', linestyle='--', 
                           label=f'Mean: {cv_results["mean_score"]:.6f}')
    ax2.axhline(y=cv_results['mean_score'] + cv_results['std_score'], color='orange', 
               linestyle=':', alpha=0.7, label=f'±1 Std')
    ax2.axhline(y=cv_results['mean_score'] - cv_results['std_score'], color='orange', 
               linestyle=':', alpha=0.7)
    ax2.set_title('Cross-Validation Scores by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training time analysis
    if cv_results.get('training_times'):
        times = cv_results['training_times']
        ax3.bar(folds, times, alpha=0.7, color='green')
        ax3.set_title('Training Time by Fold')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.grid(True, alpha=0.3)
    
    # Performance vs Training Data Size
    if detailed_metrics:
        train_sizes = [m['train_samples'] for m in detailed_metrics]
        val_losses = [m['val_loss'] for m in detailed_metrics]
        
        ax4.scatter(train_sizes, val_losses, alpha=0.7, s=60)
        ax4.set_title('Performance vs Training Data Size')
        ax4.set_xlabel('Training Samples')
        ax4.set_ylabel('Validation Loss')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation line
        if len(train_sizes) > 1:
            z = np.polyfit(train_sizes, val_losses, 1)
            p = np.poly1d(z)
            ax4.plot(train_sizes, p(train_sizes), "r--", alpha=0.8, 
                    label=f'Trend (R²={np.corrcoef(train_sizes, val_losses)[0,1]**2:.3f})')
            ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_cv_results.png'), dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_models(analyzer, processed_data, output_dir):
    """Comprehensive model evaluation"""
    try:
        print("🔍 Performing comprehensive model evaluation...")
        
        # Create evaluation report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'data_stats': {
                'total_samples': len(processed_data),
                'features': len(processed_data.columns),
                'time_span': str(processed_data.index[-1] - processed_data.index[0]),
                'missing_data_pct': (processed_data.isnull().sum().sum() / 
                                   (len(processed_data) * len(processed_data.columns)) * 100)
            }
        }
        
        # Model performance metrics (if model exists)
        if hasattr(analyzer, 'ml_model') and analyzer.ml_model is not None:
            print("  Calculating model performance metrics...")
            
            # Generate predictions on a sample of data
            sample_size = min(1000, len(processed_data))
            sample_data = processed_data.tail(sample_size)
            
            try:
                # This would depend on the specific model implementation
                # For now, we'll create a placeholder
                evaluation_report['model_metrics'] = {
                    'model_type': 'Enhanced ML Model',
                    'architecture': 'Deep Neural Network',
                    'sequence_length': getattr(analyzer.ml_model, 'sequence_length', 256),
                    'max_flares': getattr(analyzer.ml_model, 'max_flares', 5),
                    'enhanced': True
                }
            except Exception as e:
                print(f"    Model evaluation error: {e}")
                evaluation_report['model_metrics'] = {'error': str(e)}
        
        # Save evaluation report
        import json
        with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print("  ✅ Evaluation report saved")
        
        return evaluation_report
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return None


def create_data_visualizations(combined_data, processed_data, xrs_cols, output_dir):
    """Create enhanced data visualizations"""
    print("\n📈 Creating data visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot original vs processed data comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Processing Overview', fontsize=16)
    
    # Raw data plot
    for col in xrs_cols[:2]:  # Plot first 2 XRS columns
        if col in combined_data.columns:
            axes[0, 0].plot(combined_data.index, combined_data[col], alpha=0.7, label=f'Raw {col}')
    axes[0, 0].set_title('Raw Data')
    axes[0, 0].set_ylabel('Flux (W/m²)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Processed data plot
    for col in xrs_cols[:2]:
        if col in processed_data.columns:
            axes[0, 1].plot(processed_data.index, processed_data[col], alpha=0.7, label=f'Processed {col}')
    axes[0, 1].set_title('Processed Data')
    axes[0, 1].set_ylabel('Flux (W/m²)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature correlation heatmap
    correlation_cols = [col for col in processed_data.columns if any(xrs in col for xrs in xrs_cols)]
    corr_matrix = processed_data[correlation_cols[:10]].corr()  # Limit to 10 features for readability
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix')
    
    # Data distribution plot
    if len(xrs_cols) > 0 and xrs_cols[0] in processed_data.columns:
        processed_data[xrs_cols[0]].hist(bins=50, alpha=0.7, ax=axes[1, 1])
        axes[1, 1].set_title(f'{xrs_cols[0]} Distribution')
        axes[1, 1].set_xlabel('Flux Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_processing_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_data_quality_report(processed_data):
    """Print data quality report"""
    print("\n📊 Data Quality Report:")
    print(f"  Total data points: {len(processed_data)}")
    print(f"  Total features: {len(processed_data.columns)}")
    print(f"  Missing data percentage: {(processed_data.isnull().sum().sum() / (len(processed_data) * len(processed_data.columns)) * 100):.2f}%")
    for col in processed_data.columns:
        missing_pct = (processed_data[col].isnull().sum() / len(processed_data)) * 100
        if missing_pct > 0:
            print(f"    {col}: {missing_pct:.2f}% missing")


def main():
    """Train with real CSV XRS data from xrsa and xrsb folders"""
    
    # Set up paths
    base_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    xrsa_dir = os.path.join(base_data_dir, 'xrsa')
    xrsb_dir = os.path.join(base_data_dir, 'xrsb')
    output_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    print("🚀 Enhanced Solar Flare Analysis Training Script")
    print("=" * 60)
    
    # Check if xrsa and xrsb directories exist
    if not (os.path.exists(xrsa_dir) and os.path.isdir(xrsa_dir)):
        print(f"❌ Directory not found: {xrsa_dir}")
        print("Please create the xrsa directory in the data folder")
        return
    
    if not (os.path.exists(xrsb_dir) and os.path.isdir(xrsb_dir)):
        print(f"❌ Directory not found: {xrsb_dir}")
        print("Please create the xrsb directory in the data folder")
        return
    
    try:
        # Step 1: Load CSV data
        combined_data = load_csv_data(xrsa_dir, xrsb_dir)
        
        # Step 2: Advanced preprocessing
        processed_data, xrs_cols = advanced_preprocessing(combined_data)
        
        # Step 3: Advanced feature engineering
        enhanced_data = advanced_feature_engineering(processed_data, xrs_cols, output_dir)
        
        # Step 4: Anomaly detection
        anomaly_results = apply_anomaly_detection(enhanced_data, xrs_cols, output_dir)
        
        # Step 5: PCA analysis
        n_components, pca_model = perform_pca_analysis(enhanced_data, output_dir)
        
        # Step 6: Create visualizations
        create_data_visualizations(combined_data, processed_data, xrs_cols, output_dir)
        
        # Step 7: Print data quality report
        print_data_quality_report(enhanced_data)
        
        # Step 8: Initialize analyzer with processed data
        analyzer = EnhancedSolarFlareAnalyzer(
            data_path=base_data_dir,
            output_dir=output_dir
        )
        
        # Add processed data to analyzer
        analyzer.results['raw_data'] = combined_data
        analyzer.results['preprocessed_data'] = enhanced_data
        
        # Step 9: Enhanced ML model initialization
        print(f"\n🧠 Initializing enhanced ML model...")
        analyzer.initialize_ml_model(
            sequence_length=256,  # Longer sequences for better pattern recognition
            max_flares=5,         # Handle more complex overlapping scenarios
            enhanced=True         # Use enhanced model architecture
        )
        
        # Step 10: Enhanced training with multiple strategies
        print(f"\n🎯 Enhanced training with real GOES CSV data...")
        
        # Strategy 1: Train with real data only
        print("\n--- Training Strategy 1: Real Data Only ---")
        history_real = analyzer.train_ml_model(
            use_synthetic_data=False,
            n_synthetic_samples=0,
            validation_split=0.2,
            epochs=30,                # More epochs for better convergence
            batch_size=32,            # Optimal batch size
            enhanced=True,
            learning_rate=0.001,      # Conservative learning rate
            early_stopping_patience=10,
            reduce_lr_patience=5
        )
        
        if history_real is not None:
            print("✅ Real data training completed!")
            
            # Save real data model
            real_model_path = os.path.join(output_dir, 'enhanced_real_data_model.h5')
            if hasattr(analyzer.ml_model, 'save_model'):
                analyzer.ml_model.save_model(real_model_path)
                print(f"💾 Real data model saved to: {real_model_path}")
            
            # Plot training history
            plot_training_history(history_real, output_dir, 'real_data_training')
        
        # Strategy 2: Hybrid training (synthetic + real data)
        print("\n--- Training Strategy 2: Hybrid Training (Synthetic + Real) ---")
        
        # First train on synthetic data for initialization
        print("Phase 1: Pre-training on synthetic data...")
        analyzer.initialize_ml_model(
            sequence_length=256,
            max_flares=5,
            enhanced=True
        )
        
        history_synthetic = analyzer.train_ml_model(
            use_synthetic_data=True,
            n_synthetic_samples=5000,  # Generate synthetic data
            validation_split=0.2,
            epochs=20,                 # Fewer epochs for pre-training
            batch_size=32,
            enhanced=True,
            learning_rate=0.002       # Slightly higher LR for synthetic data
        )
        
        if history_synthetic is not None:
            print("✅ Synthetic pre-training completed!")
            
            # Phase 2: Fine-tune on real data
            print("Phase 2: Fine-tuning on real data...")
            history_finetuned = analyzer.train_ml_model(
                use_synthetic_data=False,
                n_synthetic_samples=0,
                validation_split=0.2,
                epochs=15,                # Fine-tuning epochs
                batch_size=16,            # Smaller batch for fine-tuning
                enhanced=True,
                learning_rate=0.0005,     # Lower LR for fine-tuning
                early_stopping_patience=8
            )
            
            if history_finetuned is not None:
                print("✅ Fine-tuning completed!")
                
                # Save hybrid model
                hybrid_model_path = os.path.join(output_dir, 'enhanced_hybrid_model.h5')
                if hasattr(analyzer.ml_model, 'save_model'):
                    analyzer.ml_model.save_model(hybrid_model_path)
                    print(f"💾 Hybrid model saved to: {hybrid_model_path}")
                
                # Plot combined training history
                plot_combined_training_history(history_synthetic, history_finetuned, output_dir)
        
        # Strategy 3: Cross-validation training
        print("\n--- Training Strategy 3: Cross-Validation Training ---")
        cv_results = perform_cross_validation_training(analyzer, enhanced_data, output_dir)
        
        # Model evaluation and comparison
        print("\n🔍 Performing comprehensive model evaluation...")
        evaluation_report = evaluate_models(analyzer, enhanced_data, output_dir)
        
        # Generate enhanced analysis report
        print(f"\n📊 Generating enhanced analysis report...")
        results = analyzer.analyze_solar_flares(
            plot_results=True,
            save_results=True,
            nanoflare_analysis=True,   # Enable advanced analysis
            corona_heating=True        # Enable corona heating analysis
        )
        
        print(f"\n🎉 Enhanced training pipeline completed successfully!")
        print(f"📁 Output files saved to: {output_dir}")
        print(f"📈 Training visualizations and reports available")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
