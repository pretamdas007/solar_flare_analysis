"""
Generate synthetic GOES XRS data for testing the solar flare analysis pipeline.
This creates realistic-looking solar flare time series data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_flare_profile(t, A, B, C, D):
    """
    Generate flare profile using the Gryciuk model
    """
    from scipy.special import erf
    
    Z = (2 * B + C**2 * D) / (2 * C)
    exp_term = np.exp(D * (B - t) + (C**2 * D**2) / 4)
    erf_term1 = erf(Z)
    erf_term2 = erf((Z - t) / C)
    flux = 0.5 * np.sqrt(np.pi) * A * C * exp_term * (erf_term1 - erf_term2)
    
    return np.where(np.isfinite(flux), flux, 0)

def generate_sample_data():
    """
    Generate sample GOES XRS data with multiple flares
    """
    # Time parameters
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    duration_hours = 24  # 24 hours of data
    sampling_interval = 60  # 1 minute sampling
    
    n_points = int(duration_hours * 3600 / sampling_interval)
    time_seconds = np.arange(n_points) * sampling_interval
    time_datetime = [start_time + timedelta(seconds=t) for t in time_seconds]
    
    # Background flux level
    background_flux = 1e-9  # W/m²
    noise_level = background_flux * 0.1
    
    # Initialize flux array with background
    flux = np.full(n_points, background_flux)
    
    # Add noise
    flux += np.random.normal(0, noise_level, n_points)
    
    # Flare parameters for different classes
    flare_events = [
        # (start_time_hours, A, B_duration, C, D) 
        (2.0, 1e-8, 600, 300, 1e-4),   # Small A-class flare
        (6.5, 5e-8, 900, 450, 8e-5),   # B-class flare
        (12.2, 2e-7, 1200, 600, 6e-5), # C-class flare
        (18.8, 8e-7, 1500, 800, 4e-5), # M-class flare
        (20.5, 1e-8, 400, 200, 1.5e-4), # Another small flare
    ]
    
    # Add flares to the time series
    for start_hour, A, B_dur, C, D in flare_events:
        start_idx = int(start_hour * 3600 / sampling_interval)
        flare_duration_points = int(4 * B_dur / sampling_interval)  # 4x the characteristic time
        
        if start_idx + flare_duration_points < n_points:
            t_flare = np.arange(flare_duration_points) * sampling_interval
            flare_profile = generate_flare_profile(t_flare, A, B_dur/2, C, D)
            
            # Add flare to background
            end_idx = start_idx + len(flare_profile)
            flux[start_idx:end_idx] += flare_profile
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': time_datetime,
        'xrs_flux': flux
    })
    
    return data

def create_sample_data_files(output_dir='data', n_files=3):
    """
    Create multiple sample data files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i in range(n_files):
        # Generate different data for each file
        np.random.seed(42 + i)  # Different seed for each file
        
        data = generate_sample_data()
        
        # Add some variation to each dataset
        if i == 1:
            # Second file: add more smaller flares
            additional_flares = [
                (4.0, 3e-9, 300, 150, 2e-4),
                (8.0, 5e-9, 400, 200, 1.8e-4),
                (14.0, 2e-9, 250, 120, 2.5e-4),
            ]
            
            for start_hour, A, B_dur, C, D in additional_flares:
                start_idx = int(start_hour * 3600 / 60)  # 60s sampling
                flare_duration_points = int(4 * B_dur / 60)
                
                if start_idx + flare_duration_points < len(data):
                    t_flare = np.arange(flare_duration_points) * 60
                    flare_profile = generate_flare_profile(t_flare, A, B_dur/2, C, D)
                    
                    end_idx = start_idx + len(flare_profile)
                    data.loc[start_idx:end_idx-1, 'xrs_flux'] += flare_profile
        
        elif i == 2:
            # Third file: add one large X-class flare
            start_hour = 10.0
            A, B_dur, C, D = 3e-6, 2000, 1000, 3e-5  # X-class parameters
            
            start_idx = int(start_hour * 3600 / 60)
            flare_duration_points = int(4 * B_dur / 60)
            
            if start_idx + flare_duration_points < len(data):
                t_flare = np.arange(flare_duration_points) * 60
                flare_profile = generate_flare_profile(t_flare, A, B_dur/2, C, D)
                
                end_idx = start_idx + len(flare_profile)
                data.loc[start_idx:end_idx-1, 'xrs_flux'] += flare_profile
        
        # Save to CSV
        filename = f'goes_xrs_sample_{i+1}.csv'
        filepath = output_dir / filename
        data.to_csv(filepath, index=False)
        
        print(f"Created sample file: {filepath}")
    
    print(f"\nSample data files created in {output_dir}/")
    print("These files contain synthetic GOES XRS data with embedded solar flares.")

if __name__ == '__main__':
    create_sample_data_files()
