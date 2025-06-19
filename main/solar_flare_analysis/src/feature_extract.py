"""
Feature Extraction Module for Solar Flare Analysis

This module extracts features from fitted flare parameters and prepares data for ML models.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import argparse
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class FlareFeatureExtractor:
    """
    Extract features from fitted flare parameters for ML models
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = None
    
    def extract_basic_features(self, parameters):
        """
        Extract basic features from fitted parameters
        
        Parameters:
        -----------
        parameters : dict
            Dictionary containing fitted parameters A, B, C, D
            
        Returns:
        --------
        dict
            Dictionary of extracted features
        """
        features = {}
        
        # Direct parameters
        features['amplitude'] = parameters['A']
        features['peak_time'] = parameters['B']
        features['width'] = parameters['C']
        features['decay_rate'] = parameters['D']
        
        # Derived physical quantities
        features['rise_time'] = parameters['B']  # Time to peak
        features['decay_time'] = 1.0 / parameters['D'] if parameters['D'] > 0 else np.inf
        features['total_duration'] = features['rise_time'] + 3 * features['decay_time']
        
        # Energy-related features
        features['peak_flux'] = parameters['A']
        features['integrated_flux'] = parameters['A'] * parameters['C'] * np.sqrt(np.pi)
        
        # Shape characteristics
        features['asymmetry'] = features['decay_time'] / features['rise_time'] if features['rise_time'] > 0 else np.inf
        features['sharpness'] = parameters['A'] / parameters['C'] if parameters['C'] > 0 else 0
        
        # Logarithmic features (often better for ML)
        for param in ['A', 'C', 'D']:
            if parameters[param] > 0:
                features[f'log_{param.lower()}'] = np.log10(parameters[param])
            else:
                features[f'log_{param.lower()}'] = -np.inf
        
        return features
    
    def extract_statistical_features(self, flux_data, fitted_flux):
        """
        Extract statistical features from the fit quality
        """
        features = {}
        
        residuals = flux_data - fitted_flux
        
        # Fit quality metrics
        features['r_squared'] = 1 - np.var(residuals) / np.var(flux_data)
        features['rmse'] = np.sqrt(np.mean(residuals**2))
        features['mae'] = np.mean(np.abs(residuals))
        
        # Residual statistics
        features['residual_mean'] = np.mean(residuals)
        features['residual_std'] = np.std(residuals)
        features['residual_skewness'] = stats.skew(residuals)
        features['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # Relative errors
        rel_error = np.abs(residuals) / (flux_data + 1e-10)
        features['mean_relative_error'] = np.mean(rel_error)
        features['max_relative_error'] = np.max(rel_error)
        
        return features
    
    def extract_temporal_features(self, time_data, flux_data):
        """
        Extract features related to temporal characteristics
        """
        features = {}
        
        # Basic temporal stats
        features['duration'] = time_data[-1] - time_data[0]
        features['sampling_rate'] = len(time_data) / features['duration']
        
        # Peak timing
        peak_idx = np.argmax(flux_data)
        features['peak_position_fraction'] = peak_idx / len(flux_data)
        features['time_to_peak'] = time_data[peak_idx] - time_data[0]
        
        # Flux statistics
        features['max_flux'] = np.max(flux_data)
        features['min_flux'] = np.min(flux_data)
        features['mean_flux'] = np.mean(flux_data)
        features['std_flux'] = np.std(flux_data)
        features['flux_range'] = features['max_flux'] - features['min_flux']
        
        # Flux distribution characteristics
        features['flux_skewness'] = stats.skew(flux_data)
        features['flux_kurtosis'] = stats.kurtosis(flux_data)
        
        # Rise and decay phases
        rise_flux = flux_data[:peak_idx+1]
        decay_flux = flux_data[peak_idx:]
        
        if len(rise_flux) > 1:
            features['rise_slope'] = (features['max_flux'] - features['min_flux']) / features['time_to_peak']
        else:
            features['rise_slope'] = 0
        
        if len(decay_flux) > 1:
            # Exponential decay fit
            try:
                decay_time = time_data[peak_idx:] - time_data[peak_idx]
                log_decay_flux = np.log(decay_flux + 1e-10)
                slope, _, _, _, _ = stats.linregress(decay_time, log_decay_flux)
                features['decay_slope'] = -slope
            except:
                features['decay_slope'] = 0
        else:
            features['decay_slope'] = 0
        
        return features
    
    def create_synthetic_features(self, features):
        """
        Create synthetic features through combinations of basic features
        """
        synthetic = {}
        
        # Energy density features
        if 'integrated_flux' in features and 'duration' in features:
            synthetic['energy_density'] = features['integrated_flux'] / features['duration']
        
        # Efficiency metrics
        if 'peak_flux' in features and 'integrated_flux' in features:
            synthetic['flux_efficiency'] = features['peak_flux'] / features['integrated_flux']
        
        # Temporal ratios
        if 'time_to_peak' in features and 'duration' in features:
            synthetic['rise_fraction'] = features['time_to_peak'] / features['duration']
            synthetic['decay_fraction'] = 1 - synthetic['rise_fraction']
        
        # Shape complexity
        if 'width' in features and 'decay_rate' in features:
            synthetic['shape_complexity'] = features['width'] * features['decay_rate']
        
        # Power law indicators (for α prediction)
        if 'amplitude' in features and 'width' in features:
            synthetic['power_indicator'] = np.log10(features['amplitude']) / np.log10(features['width'])
        
        return synthetic
    
    def extract_all_features(self, flare_data):
        """
        Extract all features from a single flare fit result
        
        Parameters:
        -----------
        flare_data : dict
            Dictionary containing fit results from fit_flare_model.py
            
        Returns:
        --------
        dict
            Complete feature dictionary
        """
        all_features = {}
        
        # Basic parameter features
        if 'parameters' in flare_data:
            basic_features = self.extract_basic_features(flare_data['parameters'])
            all_features.update(basic_features)
        
        # Statistical features from fit quality
        if 'flux' in flare_data and 'fitted_flux' in flare_data:
            stat_features = self.extract_statistical_features(
                np.array(flare_data['flux']), 
                np.array(flare_data['fitted_flux'])
            )
            all_features.update(stat_features)
        
        # Temporal features
        if 'normalized_time' in flare_data and 'flux' in flare_data:
            temporal_features = self.extract_temporal_features(
                np.array(flare_data['normalized_time']), 
                np.array(flare_data['flux'])
            )
            all_features.update(temporal_features)
        
        # Synthetic features
        synthetic_features = self.create_synthetic_features(all_features)
        all_features.update(synthetic_features)
        
        # Add metadata
        all_features['flare_id'] = flare_data.get('flare_id', 0)
        all_features['file'] = flare_data.get('file', 'unknown')
        all_features['fit_success'] = flare_data.get('success', False)
        
        return all_features
    
    def process_fit_results(self, fit_results):
        """
        Process multiple flare fit results and extract features
        
        Parameters:
        -----------
        fit_results : list
            List of fit result dictionaries
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with extracted features
        """
        all_features = []
        
        for flare_data in fit_results:
            if flare_data.get('success', False):
                features = self.extract_all_features(flare_data)
                all_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
          # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['flare_id', 'file', 'fit_success']]
        
        return df
    
    def add_synthetic_labels(self, df, alpha_range=(1.5, 3.0), nanoflare_threshold=1e-9):
        """
        Add synthetic labels for training (since real labels may not be available)
        For real GOES XRS data, use realistic thresholds and relationships
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Feature DataFrame
        alpha_range : tuple
            Range for synthetic alpha values
        nanoflare_threshold : float
            Threshold for nanoflare classification (adjusted for real data)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added synthetic labels
        """
        df = df.copy()
        
        # For real GOES XRS data, use more realistic nanoflare threshold
        # GOES XRS A-class threshold is typically around 1e-8 W/m²
        real_nanoflare_threshold = 1e-8  # A-class flare threshold
        
        # Synthetic alpha based on flux and duration characteristics
        # Based on observed relationships in solar flare studies
        if 'peak_flux' in df.columns and 'duration' in df.columns:
            # Normalize features for alpha calculation
            log_flux = np.log10(df['peak_flux'].clip(lower=1e-10))
            log_duration = np.log10(df['duration'].clip(lower=1))
            
            # Alpha relationship based on physical models:
            # Smaller, weaker flares tend to have steeper (larger) alpha values
            # This is based on observed power-law distributions
            alpha_base = 2.0
            flux_factor = (log_flux + 9) / 5  # Normalize around typical range
            duration_factor = (log_duration - 2) / 2  # Normalize around typical range
            
            # Inverse relationship with flux, positive with duration
            df['alpha'] = alpha_base - 0.3 * flux_factor + 0.2 * duration_factor
            
            # Add realistic noise
            df['alpha'] += np.random.normal(0, 0.15, len(df))
            df['alpha'] = np.clip(df['alpha'], alpha_range[0], alpha_range[1])
        
        # Realistic nanoflare classification for GOES XRS data
        if 'peak_flux' in df.columns:
            df['is_nanoflare'] = (df['peak_flux'] < real_nanoflare_threshold).astype(int)
        
        # GOES flare class based on real classification scheme
        if 'peak_flux' in df.columns:
            conditions = [
                df['peak_flux'] < 1e-8,        # Below A-class (nanoflares)
                (df['peak_flux'] >= 1e-8) & (df['peak_flux'] < 1e-7),   # A-class
                (df['peak_flux'] >= 1e-7) & (df['peak_flux'] < 1e-6),   # B-class
                (df['peak_flux'] >= 1e-6) & (df['peak_flux'] < 1e-5),   # C-class
                (df['peak_flux'] >= 1e-5) & (df['peak_flux'] < 1e-4),   # M-class
                df['peak_flux'] >= 1e-4                                  # X-class
            ]
            choices = ['Nano', 'A', 'B', 'C', 'M', 'X']
            df['flare_class'] = np.select(conditions, choices, default='A')
        
        return df
    
    def prepare_ml_data(self, df, target_column='alpha', test_size=0.2, random_state=42):
        """
        Prepare data for machine learning
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Feature DataFrame
        target_column : str
            Target variable name
        test_size : float
            Fraction of data for testing
        random_state : int
            Random seed
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, feature_names)
        """
        from sklearn.model_selection import train_test_split
        
        # Select features (exclude metadata and target)
        feature_cols = [col for col in df.columns if col not in 
                       ['flare_id', 'file', 'fit_success', target_column, 'alpha', 'is_nanoflare', 'flare_class']]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def load_all_fits(fits_dir):
    """
    Load all fit results from JSON files
    
    Parameters:
    -----------
    fits_dir : str or Path
        Directory containing fit JSON files
        
    Returns:
    --------
    list
        Combined list of all fit results
    """
    fits_dir = Path(fits_dir)
    all_fits = []
    
    for json_file in fits_dir.glob('*_fits.json'):
        try:
            with open(json_file, 'r') as f:
                fits = json.load(f)
                all_fits.extend(fits)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_fits

def main():
    """
    Main function to extract features from fitted flare data
    """
    parser = argparse.ArgumentParser(description='Extract features from fitted flare data')
    parser.add_argument('--fits_dir', default='fits', help='Directory containing fit JSON files')
    parser.add_argument('--output_dir', default='output', help='Output directory for feature data')
    parser.add_argument('--target', default='alpha', choices=['alpha', 'is_nanoflare'], 
                       help='Target variable for ML')
    
    args = parser.parse_args()
    
    fits_dir = Path(args.fits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load all fit results
    print("Loading fit results...")
    all_fits = load_all_fits(fits_dir)
    
    if not all_fits:
        print(f"No fit results found in {fits_dir}")
        print("Please run fit_flare_model.py first")
        return
    
    print(f"Loaded {len(all_fits)} flare fit results")
    
    # Extract features
    print("Extracting features...")
    extractor = FlareFeatureExtractor()
    feature_df = extractor.process_fit_results(all_fits)
    
    if feature_df.empty:
        print("No successful fits found for feature extraction")
        return
    
    print(f"Extracted {len(feature_df)} feature vectors with {len(extractor.feature_names)} features")
    
    # Add synthetic labels for demonstration
    print("Adding synthetic labels...")
    feature_df = extractor.add_synthetic_labels(feature_df)
    
    # Prepare ML data
    print(f"Preparing data for {args.target} prediction...")
    X_train, X_test, y_train, y_test, feature_names = extractor.prepare_ml_data(
        feature_df, target_column=args.target
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save processed data
    np.savez(
        output_dir / f'ml_data_{args.target}.npz',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names
    )
    
    # Save feature DataFrame
    feature_df.to_csv(output_dir / 'flare_features.csv', index=False)
    
    # Save feature importance ranking (based on correlation with target)
    if args.target in feature_df.columns:
        feature_importance = feature_df[feature_names].corrwith(feature_df[args.target]).abs().sort_values(ascending=False)
        feature_importance.to_csv(output_dir / f'feature_importance_{args.target}.csv')
        
        print(f"\nTop 10 features correlated with {args.target}:")
        print(feature_importance.head(10))
    
    print(f"\nFeature extraction complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
