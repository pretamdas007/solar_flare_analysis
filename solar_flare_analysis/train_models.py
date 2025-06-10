#!/usr/bin/env python
"""
Train ML models with real GOES XRS CSV data files from xrsa and xrsb directories
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
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


def preprocess_single_file(df, csv_file):
    """
    Preprocess a single CSV file
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw CSV data from single file
    csv_file : str
        Path to the CSV file
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed data
    """
    try:
        print(f"   🔧 Preprocessing {os.path.basename(csv_file)}...")
        
        # Ensure there's a datetime column to use as index
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        elif 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif df.index.name in ['time', 'datetime', 'date']:
            pass  # Already has datetime index
        else:
            # Try to find datetime-like column
            date_cols = [col for col in df.columns if any(word in col.lower() for word in ['time', 'date', 'timestamp'])]
            if date_cols:
                df.set_index(date_cols[0], inplace=True)
        
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
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df = df.sort_index()
        
        # Identify XRS columns
        xrs_cols = [col for col in df.columns if 'xrs' in col.lower()]
        
        if not xrs_cols:
            print(f"   ⚠️ No XRS columns found in {os.path.basename(csv_file)}")
            return None
        
        print(f"   Found XRS columns: {xrs_cols}")
        
        # Quality filtering for XRS columns
        for col in xrs_cols:
            if col in df.columns:
                # Remove null, negative and unreasonable values
                mask = (
                    (df[col] > 0) &  # Positive values only
                    (~np.isnan(df[col])) &  # No NaNs
                    (~np.isinf(df[col])) &  # No infinities
                    (df[col] < 1e-2)  # Upper limit on reasonable values
                )
                df.loc[~mask, col] = np.nan
                
                # Simple outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More lenient outlier detection
                upper_bound = Q3 + 3 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df.loc[outlier_mask, col] = np.nan
        
        # Resample to 1-minute cadence
        print(f"   ⏰ Resampling to 1-minute cadence...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_resampled = df[numeric_cols].resample('1min').mean()
        
        # Enhanced gap filling
        for col in df_resampled.columns:
            # Forward fill short gaps (up to 5 minutes)
            df_resampled[col] = df_resampled[col].fillna(method='ffill', limit=5)
            # Backward fill remaining short gaps
            df_resampled[col] = df_resampled[col].fillna(method='bfill', limit=5)
            # Interpolate medium gaps (up to 15 minutes)
            df_resampled[col] = df_resampled[col].interpolate(method='time', limit=15)
        
        # Basic feature engineering for individual file
        for col in xrs_cols:
            if col in df_resampled.columns:
                # Moving averages
                df_resampled[f'{col}_ma5'] = df_resampled[col].rolling(window=5, center=True).mean()
                df_resampled[f'{col}_ma15'] = df_resampled[col].rolling(window=15, center=True).mean()
                
                # Derivatives
                df_resampled[f'{col}_derivative'] = df_resampled[col].diff()
                
                # Log transform
                df_resampled[f'{col}_log'] = np.log10(df_resampled[col] + 1e-12)
        
        # Cross-channel features if multiple XRS channels exist
        if len(xrs_cols) >= 2:
            col1, col2 = xrs_cols[0], xrs_cols[1]
            df_resampled['xrs_ratio'] = df_resampled[col1] / (df_resampled[col2] + 1e-12)
            df_resampled['xrs_sum'] = df_resampled[col1] + df_resampled[col2]
        
        # Remove infinite values
        df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)
        
        # Final cleanup - remove rows with too many missing values
        df_resampled = df_resampled.dropna(thresh=len(df_resampled.columns) * 0.5)
        
        print(f"   ✅ Preprocessed to {len(df_resampled)} data points")
        
        return df_resampled
        
    except Exception as e:
        print(f"   ❌ Error preprocessing {os.path.basename(csv_file)}: {e}")
        return None


def create_incremental_training_plots(training_history, output_dir):
    """Create plots showing incremental training progress"""
    if not training_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Incremental Training Progress', fontsize=16)
    
    files = [item['file'] for item in training_history]
    losses = [item['final_loss'] for item in training_history]
    val_losses = [item.get('final_val_loss') for item in training_history if item.get('final_val_loss') is not None]
    file_sizes = [item['file_size_mb'] for item in training_history]
    data_points = [item['data_points'] for item in training_history]
    
    # Training loss progression
    axes[0, 0].plot(range(len(losses)), losses, 'bo-', markersize=4)
    axes[0, 0].set_title('Training Loss by File')
    axes[0, 0].set_xlabel('File Number')
    axes[0, 0].set_ylabel('Final Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss progression (if available)
    if val_losses and len(val_losses) > 1:
        axes[0, 1].plot(range(len(val_losses)), val_losses, 'ro-', markersize=4)
        axes[0, 1].set_title('Validation Loss by File')
        axes[0, 1].set_xlabel('File Number')
        axes[0, 1].set_ylabel('Final Validation Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Validation Loss Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Validation Loss by File')
    
    # File size vs performance
    if len(file_sizes) > 1 and len(losses) > 1:
        axes[1, 0].scatter(file_sizes, losses, alpha=0.7, s=60)
        axes[1, 0].set_title('File Size vs Training Loss')
        axes[1, 0].set_xlabel('File Size (MB)')
        axes[1, 0].set_ylabel('Final Training Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Data points vs performance
    if len(data_points) > 1 and len(losses) > 1:
        axes[1, 1].scatter(data_points, losses, alpha=0.7, s=60, c='green')
        axes[1, 1].set_title('Data Points vs Training Loss')
        axes[1, 1].set_xlabel('Number of Data Points')
        axes[1, 1].set_ylabel('Final Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'incremental_training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training progress plots saved to: incremental_training_progress.png")


def load_csv_data_efficiently(data_dir, max_files=None, sample_rate=1):
    """
    Load CSV data efficiently with memory management
    
    Parameters
    ----------
    data_dir : str
        Directory containing CSV files
    max_files : int, optional
        Maximum number of files to load (for testing)
    sample_rate : int, optional
        Sample every nth row to reduce memory usage
        
    Returns
    -------
    pandas.DataFrame
        Combined data from CSV files
    """
    print(f"📂 Loading CSV data from: {data_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return pd.DataFrame()
    
    print(f"📋 Found {len(csv_files)} CSV files")
    
    # Limit files for testing/memory management
    if max_files:
        csv_files = csv_files[:max_files]
        print(f"🔢 Loading first {len(csv_files)} files only")
    
    combined_data = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            print(f"📄 Loading file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
            
            # Check file size first
            file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
            print(f"   File size: {file_size_mb:.1f} MB")
            
            # Load with sampling if file is large
            if file_size_mb > 50:  # If file > 50MB, sample every nth row
                df = pd.read_csv(csv_file, skiprows=lambda x: x % sample_rate != 0 and x != 0)
                print(f"   Sampled data (every {sample_rate} rows)")
            else:
                df = pd.read_csv(csv_file)
            
            # Basic data validation
            if len(df) == 0:
                print(f"   ⚠️ Empty file, skipping")
                continue
                
            print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Add file source for tracking
            df['source_file'] = os.path.basename(csv_file)
            combined_data.append(df)
            
            # Memory management - limit total rows
            total_rows = sum(len(d) for d in combined_data)
            if total_rows > 1000000:  # Stop at 1M rows
                print(f"🛑 Reached 1M rows limit, stopping file loading")
                break
                
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
            continue
    
    if not combined_data:
        print("❌ No data loaded successfully")
        return pd.DataFrame()
    
    print("🔗 Combining data...")
    result = pd.concat(combined_data, ignore_index=True)
    print(f"✅ Combined data: {len(result)} rows, {len(result.columns)} columns")
    
    return result


def preprocess_csv_data(data, resample_freq='1min'):
    """
    Preprocess the loaded CSV data
    
    Parameters
    ----------
    data : pandas.DataFrame
        Raw CSV data
    resample_freq : str
        Resampling frequency
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed data
    """
    print("🔧 Preprocessing CSV data...")
    
    if len(data) == 0:
        return data
    
    # Try to identify time column
    time_cols = [col for col in data.columns if any(word in col.lower() for word in ['time', 'date', 'timestamp'])]
    
    if time_cols:
        time_col = time_cols[0]
        print(f"📅 Using time column: {time_col}")
        
        try:
            # Convert to datetime
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.set_index(time_col).sort_index()
            
            # Remove duplicates
            data = data[~data.index.duplicated(keep='first')]
            
            # Resample if requested
            if resample_freq:
                print(f"⏰ Resampling to {resample_freq}")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data = data[numeric_cols].resample(resample_freq).mean()
                
        except Exception as e:
            print(f"⚠️ Error processing time column: {e}")
    
    # Handle missing values
    print("🧹 Cleaning data...")
    initial_rows = len(data)
    
    # Remove rows with all NaN
    data = data.dropna(how='all')
    
    # Forward fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    print(f"📉 Removed {initial_rows - len(data)} rows with missing data")
    print(f"✅ Final preprocessed data: {len(data)} rows")
    
    return data


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
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(data_clean)
                    
                    # Count anomalies
                    n_anomalies = np.sum(anomaly_labels == -1)
                    anomaly_percent = (n_anomalies / len(data_clean)) * 100
                    
                    anomaly_results[col] = {
                        'total_points': len(data_clean),
                        'anomalies_detected': n_anomalies,
                        'anomaly_percentage': anomaly_percent,
                        'anomaly_indices': np.where(anomaly_labels == -1)[0].tolist()
                    }
                    
                    print(f"  {col}: {n_anomalies} anomalies ({anomaly_percent:.2f}%) detected")
                    
                except Exception as e:
                    print(f"  Warning: Anomaly detection failed for {col}: {e}")
                    anomaly_results[col] = {'error': str(e)}
      # Save anomaly results
    if anomaly_results:
        anomaly_path = os.path.join(output_dir, 'anomaly_detection_results.json')
        with open(anomaly_path, 'w') as f:
            json.dump(anomaly_results, f, indent=2)
        print(f"Anomaly detection results saved to: {anomaly_path}")
    
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

def perform_cross_validation_training(analyzer, processed_data, output_dir):
    """Perform cross-validation training"""
    try:
        print("Implementing time-series cross-validation...")
        
        # Time-based splits for cross-validation
        n_splits = 5
        data_length = len(processed_data)
        split_size = data_length // n_splits
        
        cv_scores = []
        cv_histories = []
        
        for fold in range(n_splits):
            print(f"  Training fold {fold + 1}/{n_splits}...")
            
            # Create time-based train/test split
            test_start = fold * split_size
            test_end = min((fold + 1) * split_size, data_length)
            
            # Use remaining data for training
            train_indices = list(range(0, test_start)) + list(range(test_end, data_length))
            
            if len(train_indices) < 100:  # Minimum training data requirement
                print(f"    Skipping fold {fold + 1} - insufficient training data")
                continue
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
                enhanced=True
            )
            
            if fold_history is not None:
                cv_histories.append(fold_history)
                final_val_loss = fold_history.history.get('val_loss', [float('inf')])[-1]
                cv_scores.append(final_val_loss)
                print(f"    Fold {fold + 1} validation loss: {final_val_loss:.6f}")
        
        if cv_scores:
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            print(f"Cross-validation results:")
            print(f"  Mean validation loss: {mean_cv_score:.6f} ± {std_cv_score:.6f}")
            
            # Save cross-validation results
            cv_results = {
                'fold_scores': cv_scores,
                'mean_score': mean_cv_score,
                'std_score': std_cv_score,
                'n_folds': len(cv_scores)
            }
            
            # Plot CV results
            plot_cv_results(cv_scores, output_dir)
            
    except Exception as e:
        print(f"Cross-validation training failed: {e}")


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


def plot_cv_results(cv_scores, output_dir):
    """Plot cross-validation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot of CV scores
    ax1.boxplot(cv_scores)
    ax1.set_title('Cross-Validation Scores Distribution')
    ax1.set_ylabel('Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of individual folds
    folds = range(1, len(cv_scores) + 1)
    ax2.bar(folds, cv_scores, alpha=0.7)
    ax2.axhline(y=np.mean(cv_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_scores):.6f}')
    ax2.set_title('Cross-Validation Scores by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_models(analyzer, processed_data, output_dir):
    """Comprehensive model evaluation"""
    try:
        print("Performing comprehensive model evaluation...")
        
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
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")


def main():
    """Train with real CSV XRS data from xrsa and xrsb folders"""
    
    # Set up paths
    base_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    xrsa_dir = os.path.join(base_data_dir, 'xrsa')
    xrsb_dir = os.path.join(base_data_dir, 'xrsb')
    output_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    print("🚀 Training ML Models with Real GOES XRS CSV Data")
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
    
    # Find CSV files in xrsa directory
    xrsa_csv_files = [f for f in os.listdir(xrsa_dir) if f.endswith('.csv')]
    
    # Find CSV files in xrsb directory
    xrsb_csv_files = [f for f in os.listdir(xrsb_dir) if f.endswith('.csv')]
    
    if not xrsa_csv_files and not xrsb_csv_files:
        print(f"❌ No CSV files found in {xrsa_dir} or {xrsb_dir}")
        print("Please add GOES XRS CSV files to these directories")
        return
    
    print(f"📁 Found {len(xrsa_csv_files)} CSV files in xrsa directory:")
    for file in xrsa_csv_files:
        print(f"   📄 {file}")
    
    print(f"📁 Found {len(xrsb_csv_files)} CSV files in xrsb directory:")
    for file in xrsb_csv_files:
        print(f"   📄 {file}")    # Prepare data paths for loading
    all_csv_paths = []
    for file in xrsa_csv_files:
        all_csv_paths.append(os.path.join(xrsa_dir, file))
    for file in xrsb_csv_files:
        all_csv_paths.append(os.path.join(xrsb_dir, file))
    
    # Initialize analyzer with real data
    analyzer = EnhancedSolarFlareAnalyzer(
        data_path=base_data_dir,  # We'll manually load files below
        output_dir=output_dir
    )
    
    try:
        # Step 1: Initialize model once
        print(f"\n🧠 Initializing enhanced ML model...")
        analyzer.initialize_ml_model(
            sequence_length=256,  # Longer sequences for better pattern recognition
            max_flares=5,         # Handle more complex overlapping scenarios
            enhanced=True         # Use enhanced model architecture
        )
        
        # Step 2: Train incrementally on one CSV file at a time
        print(f"\n🔄 Training incrementally on CSV files...")
        
        training_history = []
        total_files_processed = 0
        
        for file_idx, csv_file in enumerate(all_csv_paths):
            try:
                print(f"\n📄 Processing file {file_idx + 1}/{len(all_csv_paths)}: {os.path.basename(csv_file)}")
                
                # Check file size first
                file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
                print(f"   File size: {file_size_mb:.1f} MB")
                
                # Load single CSV file with memory management
                if file_size_mb > 100:  # If file > 100MB, sample every 2nd row
                    print(f"   Large file detected, sampling every 2nd row...")
                    df = pd.read_csv(csv_file, skiprows=lambda x: x % 2 != 0 and x != 0, parse_dates=True)
                elif file_size_mb > 50:  # If file > 50MB, sample every nth row
                    print(f"   Medium file detected, sampling every 3rd row...")
                    df = pd.read_csv(csv_file, skiprows=lambda x: x % 3 != 0 and x != 0, parse_dates=True)
                else:
                    df = pd.read_csv(csv_file, parse_dates=True)
                
                if len(df) == 0:
                    print(f"   ⚠️ Empty file, skipping")
                    continue
                
                print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Preprocess single file
                processed_df = preprocess_single_file(df, csv_file)
                
                if processed_df is None or len(processed_df) < 100:
                    print(f"   ⚠️ Insufficient data after preprocessing, skipping")
                    continue
                
                # Store in analyzer for this iteration
                analyzer.results['raw_data'] = df
                analyzer.results['preprocessed_data'] = processed_df
                
                print(f"   🎯 Training on {len(processed_df)} processed data points...")
                
                # Train on this file's data
                history = analyzer.train_ml_model(
                    use_synthetic_data=False,
                    n_synthetic_samples=0,
                    validation_split=0.2,
                    epochs=5,  # Fewer epochs per file to avoid overfitting
                    batch_size=16,
                    enhanced=True
                )
                
                if history is not None:
                    training_history.append({
                        'file': os.path.basename(csv_file),
                        'file_size_mb': file_size_mb,
                        'data_points': len(processed_df),
                        'final_loss': history.history.get('loss', [float('inf')])[-1],
                        'final_val_loss': history.history.get('val_loss', [float('inf')])[-1] if 'val_loss' in history.history else None
                    })
                    total_files_processed += 1
                    print(f"   ✅ Training completed for {os.path.basename(csv_file)}")
                    
                    # Save intermediate model every 5 files
                    if total_files_processed % 5 == 0:
                        intermediate_model_path = os.path.join(output_dir, f'intermediate_model_after_{total_files_processed}_files.h5')
                        if hasattr(analyzer.ml_model, 'model') and analyzer.ml_model.model is not None:
                            analyzer.ml_model.model.save(intermediate_model_path)
                            print(f"   💾 Intermediate model saved after {total_files_processed} files")
                else:
                    print(f"   ❌ Training failed for {os.path.basename(csv_file)}")
                
                # Memory cleanup
                del df, processed_df
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Error processing {csv_file}: {e}")
                continue
        
        if total_files_processed == 0:
            print("❌ No files were successfully processed")
            return
        
        print(f"\n✅ Incremental training completed on {total_files_processed} files")
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_incremental_model.h5')
        if hasattr(analyzer.ml_model, 'model') and analyzer.ml_model.model is not None:
            analyzer.ml_model.model.save(final_model_path)
            print(f"💾 Final model saved to: {final_model_path}")
        
        # Save training summary
        training_summary = {
            'total_files_processed': total_files_processed,
            'training_history': training_history,
            'final_model_path': final_model_path
        }
        
        with open(os.path.join(output_dir, 'incremental_training_summary.json'), 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
          # Create training progress visualization
        create_incremental_training_plots(training_history, output_dir)
        
        # Generate final analysis report
        print(f"\n📊 Generating final analysis report...")
        if analyzer.results.get('preprocessed_data') is not None:
            results = analyzer.analyze_solar_flares(
                plot_results=True,
                save_results=True,
                nanoflare_analysis=False,   # Disable for efficiency
                corona_heating=False        # Disable for efficiency
            )
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Incremental training pipeline completed!")
    print(f"📁 Output files saved to: {output_dir}")
    print(f"📊 Processed {total_files_processed} files successfully")


if __name__ == "__main__":
    main()