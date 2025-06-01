#!/usr/bin/env python
"""
Test real GOES data processing and prepare data for training.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def load_and_process_goes_data():
    """Load and process real GOES XRS data."""
    print("📊 Loading and processing real GOES XRS data...")
    
    try:
        from src.data_processing.data_loader import load_goes_data, preprocess_xrs_data, remove_background
        from config.settings import DATA_DIR, OUTPUT_DIR
        
        # Find data files
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.nc')]
        
        if not data_files:
            print(f"❌ No .nc files found in {DATA_DIR}")
            return None
        
        print(f"📂 Found {len(data_files)} data files:")
        for file in data_files:
            print(f"   - {file}")
        
        all_data = []
        
        for file in data_files:
            file_path = os.path.join(DATA_DIR, file)
            
            try:
                print(f"\n📈 Processing {file}...")
                
                # Load GOES dataset
                dataset = load_goes_data(file_path)
                
                if dataset is not None:
                    print(f"   ✅ Dataset loaded successfully")
                    print(f"   📅 Time range: {dataset.time.min().values} to {dataset.time.max().values}")
                    
                    # Process both XRS channels
                    for channel in ['A', 'B']:
                        print(f"   📡 Processing XRS channel {channel}...")
                        
                        df = preprocess_xrs_data(dataset, channel=channel)
                        
                        if df is not None and len(df) > 100:
                            print(f"      ✅ {len(df)} data points processed")
                            print(f"      🌟 Flux range: {df['flux'].min():.2e} - {df['flux'].max():.2e} W/m²")
                            
                            # Remove background
                            df_clean = remove_background(df, window_size='1H', quantile=0.1)
                            
                            if df_clean is not None:
                                df_clean['channel'] = channel
                                df_clean['file_source'] = file
                                all_data.append(df_clean)
                                print(f"      ✅ Background removed, {len(df_clean)} points remaining")
                            else:
                                print(f"      ⚠️ Background removal failed")
                        else:
                            print(f"      ⚠️ No usable data for channel {channel}")
                
            except Exception as e:
                print(f"   ❌ Error processing {file}: {e}")
                continue
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_index()
            
            print(f"\n✅ Combined dataset:")
            print(f"   📊 Total points: {len(combined_df)}")
            print(f"   📅 Time span: {combined_df.index.min()} to {combined_df.index.max()}")
            print(f"   🌟 Flux range: {combined_df['flux'].min():.2e} to {combined_df['flux'].max():.2e} W/m²")
            
            # Save processed data
            output_file = os.path.join(OUTPUT_DIR, 'processed_goes_data.csv')
            combined_df.to_csv(output_file)
            print(f"   💾 Processed data saved to: {output_file}")
            
            return combined_df
        else:
            print("❌ No usable data found")
            return None
            
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_flares_in_real_data(df):
    """Detect flares in real GOES data."""
    print("\n🎯 Detecting flares in real data...")
    
    try:
        from src.flare_detection.traditional_detection import detect_flare_peaks, define_flare_bounds
        
        # Detect peaks for each channel
        all_flares = []
        
        for channel in df['channel'].unique():
            channel_data = df[df['channel'] == channel].copy()
            
            if len(channel_data) < 100:
                continue
                
            print(f"\n📡 Analyzing channel {channel}:")
            print(f"   📊 {len(channel_data)} data points")
            
            # Detect peaks
            peaks = detect_flare_peaks(
                channel_data, 
                flux_column='flux',
                threshold_factor=3.0,
                window_size=25
            )
            
            print(f"   🔍 Found {len(peaks)} potential flare peaks")
            
            if len(peaks) > 0:
                # Define flare bounds
                flares = define_flare_bounds(
                    channel_data,
                    flux_column='flux',
                    peak_indices=peaks['peak_index'].values,
                    start_threshold=0.5,
                    end_threshold=0.5,
                    min_duration='2min',
                    max_duration='6H'
                )
                
                print(f"   📊 Defined {len(flares)} flare events")
                
                # Add channel information
                flares['channel'] = channel
                all_flares.append(flares)
                
                # Print flare details
                for i, flare in flares.iterrows():
                    duration_min = flare['duration'].total_seconds() / 60
                    peak_flux = channel_data.iloc[flare['peak_index']]['flux']
                    
                    # Classify flare
                    if peak_flux >= 1e-4:
                        flare_class = 'X'
                    elif peak_flux >= 1e-5:
                        flare_class = 'M'
                    elif peak_flux >= 1e-6:
                        flare_class = 'C'
                    else:
                        flare_class = 'B'
                    
                    print(f"      🌟 Flare {i+1}: {flare_class}-class, "
                          f"Duration: {duration_min:.1f} min, "
                          f"Peak: {peak_flux:.2e} W/m²")
        
        if all_flares:
            combined_flares = pd.concat(all_flares, ignore_index=True)
            
            # Save flare catalog
            from config.settings import OUTPUT_DIR
            flare_file = os.path.join(OUTPUT_DIR, 'detected_flares.csv')
            combined_flares.to_csv(flare_file, index=False)
            print(f"\n💾 Flare catalog saved to: {flare_file}")
            
            return combined_flares
        else:
            print("\n📊 No flares detected in the data")
            return None
            
    except Exception as e:
        print(f"❌ Flare detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_training_sequences(df, sequence_length=128):
    """Create training sequences from real data."""
    print(f"\n🔄 Creating training sequences (length={sequence_length})...")
    
    try:
        sequences = []
        labels = []
        
        for channel in df['channel'].unique():
            channel_data = df[df['channel'] == channel].copy()
            flux_values = channel_data['flux'].values
            
            print(f"📡 Processing channel {channel}: {len(flux_values)} points")
            
            # Create overlapping sequences
            for i in range(len(flux_values) - sequence_length + 1):
                sequence = flux_values[i:i + sequence_length]
                
                # Simple labeling: check if max flux in sequence exceeds threshold
                max_flux = np.max(sequence)
                
                if max_flux > 5e-7:  # Above background
                    label = 1  # Potential flare
                else:
                    label = 0  # Background
                
                # Normalize sequence
                normalized_seq = sequence / 1e-7  # Normalize to background level
                
                sequences.append(normalized_seq)
                labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"✅ Created {len(sequences)} sequences:")
        print(f"   📊 Shape: {sequences.shape}")
        print(f"   🎯 Positive samples (potential flares): {np.sum(labels)}")
        print(f"   🎯 Negative samples (background): {len(labels) - np.sum(labels)}")
        
        # Save sequences
        from config.settings import OUTPUT_DIR
        sequences_file = os.path.join(OUTPUT_DIR, 'real_data_sequences.npz')
        np.savez(sequences_file, X=sequences, y=labels)
        print(f"   💾 Sequences saved to: {sequences_file}")
        
        return sequences, labels
        
    except Exception as e:
        print(f"❌ Sequence creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_data_sample(df):
    """Create visualizations of the processed data."""
    print("\n📈 Creating data visualizations...")
    
    try:
        from config.settings import OUTPUT_DIR
        plots_dir = os.path.join(OUTPUT_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot time series for each channel
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        for i, channel in enumerate(['A', 'B']):
            channel_data = df[df['channel'] == channel]
            
            if len(channel_data) > 0:
                axes[i].plot(channel_data.index, channel_data['flux'], 
                           linewidth=0.8, alpha=0.8, color='blue')
                axes[i].set_ylabel(f'XRS-{channel} Flux (W/m²)')
                axes[i].set_yscale('log')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_title(f'GOES XRS Channel {channel} - Solar X-ray Flux')
                
                # Add flare detection thresholds
                background_level = channel_data['flux'].quantile(0.1)
                flare_threshold = background_level * 3
                axes[i].axhline(flare_threshold, color='red', linestyle='--', 
                              alpha=0.7, label=f'Flare threshold ({flare_threshold:.2e})')
                axes[i].legend()
        
        axes[1].set_xlabel('Time')
        plt.tight_layout()
        
        plot_file = os.path.join(plots_dir, 'goes_xrs_timeseries.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Time series plot saved to: {plot_file}")
        
        # Plot flux distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, channel in enumerate(['A', 'B']):
            channel_data = df[df['channel'] == channel]
            
            if len(channel_data) > 0:
                flux_values = channel_data['flux'].values
                flux_values = flux_values[flux_values > 0]  # Remove zeros
                
                axes[i].hist(np.log10(flux_values), bins=50, alpha=0.7, 
                           color='skyblue', edgecolor='black')
                axes[i].set_xlabel('log₁₀(Flux) [W/m²]')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'XRS-{channel} Flux Distribution')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        hist_file = os.path.join(plots_dir, 'flux_distribution.png')
        plt.savefig(hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Flux distribution plot saved to: {hist_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def main():
    """Main data processing function."""
    print("🌞 Real GOES Data Processing for ML Training")
    print("=" * 60)
    
    # Load and process GOES data
    df = load_and_process_goes_data()
    
    if df is None:
        print("❌ Failed to load GOES data")
        return
    
    # Detect flares
    flares = detect_flares_in_real_data(df)
    
    # Create training sequences
    sequences, labels = create_training_sequences(df)
    
    # Create visualizations
    visualize_data_sample(df)
    
    print("\n🎉 Real data processing completed!")
    
    if flares is not None:
        print(f"📊 Summary:")
        print(f"   📈 Total data points: {len(df)}")
        print(f"   🌟 Detected flares: {len(flares)}")
        if sequences is not None:
            print(f"   🔄 Training sequences: {len(sequences)}")
            print(f"   🎯 Positive samples: {np.sum(labels)} ({100*np.sum(labels)/len(labels):.1f}%)")

if __name__ == "__main__":
    main()
