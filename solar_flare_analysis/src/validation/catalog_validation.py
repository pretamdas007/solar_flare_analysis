"""
Validation module for comparing detected flares against known catalogs.

This module provides functions to:
1. Download flare catalogs from NOAA SWPC and other sources
2. Parse and standardize flare information
3. Compare detected flares with catalog entries
4. Calculate performance metrics (precision, recall, etc.)
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from io import StringIO
import re


def download_noaa_flare_catalog(start_date, end_date, output_file=None):
    """
    Download the NOAA SWPC flare events catalog.
    
    Parameters
    ----------
    start_date : str or datetime
        Start date in YYYY-MM-DD format or as datetime object
    end_date : str or datetime
        End date in YYYY-MM-DD format or as datetime object
    output_file : str, optional
        Path to save the downloaded catalog
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the flare catalog
    """
    # Convert dates to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # NOAA SWPC requires dates in YYYYMMDD format
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    # Build URL
    url = f"https://www.swpc.noaa.gov/ftpdir/indices/events/{start_date.year}_{end_date.year}_events.txt"
    
    try:
        # Download the catalog
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the text data
        df = parse_noaa_event_list(response.text, start_date, end_date)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Saved flare catalog to {output_file}")
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading NOAA flare catalog: {e}")
        
        # Try to download individual monthly files if available
        current_date = start_date.replace(day=1)
        end_of_month = end_date.replace(day=1)
        
        all_data = []
        
        while current_date <= end_of_month:
            year_month = current_date.strftime('%Y%m')
            monthly_url = f"https://www.swpc.noaa.gov/ftpdir/indices/events/{year_month}_events.txt"
            
            try:
                monthly_response = requests.get(monthly_url)
                monthly_response.raise_for_status()
                monthly_df = parse_noaa_event_list(monthly_response.text, start_date, end_date)
                all_data.append(monthly_df)
                print(f"Downloaded catalog for {current_date.strftime('%Y-%m')}")
            except requests.exceptions.RequestException:
                print(f"No data available for {current_date.strftime('%Y-%m')}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year+1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month+1)
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            
            if output_file:
                df.to_csv(output_file, index=False)
                print(f"Saved combined flare catalog to {output_file}")
            
            return df
        else:
            print("Failed to download any flare data")
            return pd.DataFrame()


def parse_noaa_event_list(text_data, start_date, end_date):
    """
    Parse NOAA SWPC event list format into a DataFrame.
    
    Parameters
    ----------
    text_data : str
        Raw text data from NOAA SWPC event list
    start_date : datetime
        Start date for filtering events
    end_date : datetime
        End date for filtering events
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing parsed flare events
    """
    # Regular expression for event lines
    # Format is typically:
    # YYYY MM DD BBBB  NNNN   EOD     LOC      PARTICULARS           OBS
    # 2022 06 01 0045  0101   XRA 1-8   G15 5.0E-06   2.7E-06   1.1       5
    
    # Initialize lists to store data
    event_data = []
    
    # Parse lines
    for line in text_data.splitlines():
        line = line.strip()
        
        # Skip header or empty lines
        if not line or line.startswith('#') or line.startswith(':'):
            continue
        
        # Try to parse date fields
        try:
            # Check if line starts with a year (YYYY MM DD format)
            match = re.match(r'^(\d{4})\s+(\d{1,2})\s+(\d{1,2})', line)
            if match:
                year, month, day = match.groups()
                year, month, day = int(year), int(month), int(day)
                
                # Extract X-ray flare events (XRA)
                if 'XRA' in line and ('1-8' in line or '0.1-0.8' in line):
                    # Extract start and end times
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    
                    # Extract start and end times (BBBB NNNN format)
                    try:
                        start_time = parts[3]
                        end_time = parts[4]
                        
                        start_hour, start_min = int(start_time[:2]), int(start_time[2:])
                        end_hour, end_min = int(end_time[:2]), int(end_time[2:])
                        
                        # Create datetime objects
                        start_datetime = datetime(year, month, day, start_hour, start_min)
                        
                        # Handle events that cross midnight
                        if end_hour < start_hour or (end_hour == start_hour and end_min < start_min):
                            end_datetime = datetime(year, month, day, end_hour, end_min) + timedelta(days=1)
                        else:
                            end_datetime = datetime(year, month, day, end_hour, end_min)
                        
                        # Skip if outside requested date range
                        if end_datetime < start_date or start_datetime > end_date:
                            continue
                        
                        # Find peak flux value
                        peak_flux = None
                        for i, p in enumerate(parts):
                            if p == 'XRA' and i + 3 < len(parts):
                                try:
                                    peak_flux = float(parts[i+3])
                                except ValueError:
                                    try:
                                        # Try scientific notation with 'E' format
                                        peak_flux = float(re.sub(r'(\d+)E(-?\d+)', r'\1e\2', parts[i+3]))
                                    except ValueError:
                                        peak_flux = None
                        
                        # Determine flare class and magnitude
                        flare_class = None
                        magnitude = None
                        
                        if peak_flux is not None:
                            if peak_flux >= 1e-4:
                                flare_class = 'X'
                                magnitude = peak_flux / 1e-4
                            elif peak_flux >= 1e-5:
                                flare_class = 'M'
                                magnitude = peak_flux / 1e-5
                            elif peak_flux >= 1e-6:
                                flare_class = 'C'
                                magnitude = peak_flux / 1e-6
                            elif peak_flux >= 1e-7:
                                flare_class = 'B'
                                magnitude = peak_flux / 1e-7
                            else:
                                flare_class = 'A'
                                magnitude = peak_flux / 1e-8
                        
                        # Store event
                        event_data.append({
                            'start_time': start_datetime,
                            'end_time': end_datetime,
                            'peak_flux': peak_flux,
                            'flare_class': flare_class,
                            'magnitude': magnitude,
                            'classification': f"{flare_class}{magnitude:.1f}" if flare_class and magnitude else None,
                            'source': 'NOAA SWPC'
                        })
                    
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line}. Error: {e}")
        
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}. Error: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(event_data)
    
    # Sort by start time
    if not df.empty:
        df.sort_values('start_time', inplace=True)
    
    return df


def compare_detected_flares(detected_flares, catalog_flares, time_tolerance='5min'):
    """
    Compare detected flares with catalog entries.
    
    Parameters
    ----------
    detected_flares : pandas.DataFrame
        DataFrame with detected flares, must have 'start_time', 'peak_time', 'end_time', 'peak_flux' columns
    catalog_flares : pandas.DataFrame
        DataFrame with catalog flares, must have 'start_time', 'end_time', 'peak_flux' columns
    time_tolerance : str, optional
        Time tolerance for matching flares, default is 5 minutes
        
    Returns
    -------
    dict
        Dictionary with comparison results including:
        - matched_flares: DataFrame with matched flares
        - unmatched_detected: DataFrame with detected flares not in catalog
        - unmatched_catalog: DataFrame with catalog flares not detected
        - metrics: Dictionary with performance metrics
    """
    # Convert time_tolerance to timedelta
    if isinstance(time_tolerance, str):
        if 'min' in time_tolerance:
            minutes = int(time_tolerance.replace('min', ''))
            time_tolerance = timedelta(minutes=minutes)
        elif 'sec' in time_tolerance:
            seconds = int(time_tolerance.replace('sec', ''))
            time_tolerance = timedelta(seconds=seconds)
        else:
            time_tolerance = timedelta(minutes=5)  # Default
    
    # Initialize matches
    matches = []
    detected_indices_matched = set()
    catalog_indices_matched = set()
    
    # Find matches
    for i, detected in detected_flares.iterrows():
        for j, catalog in catalog_flares.iterrows():
            # Check if peak times are close
            peak_time_diff = None
            if 'peak_time' in detected.index and 'peak_time' in catalog.index:
                peak_time_diff = abs(detected['peak_time'] - catalog['peak_time'])
                if peak_time_diff <= time_tolerance:
                    matches.append({
                        'detected_idx': i,
                        'catalog_idx': j,
                        'detected_peak_flux': detected['peak_flux'],
                        'catalog_peak_flux': catalog['peak_flux'],
                        'flux_ratio': detected['peak_flux'] / catalog['peak_flux'] if catalog['peak_flux'] > 0 else float('inf'),
                        'detected_start': detected['start_time'],
                        'catalog_start': catalog['start_time'],
                        'detected_end': detected['end_time'],
                        'catalog_end': catalog['end_time'],
                        'time_diff': peak_time_diff.total_seconds() / 60  # Time difference in minutes
                    })
                    detected_indices_matched.add(i)
                    catalog_indices_matched.add(j)
                    continue
            
            # If peak_time not available or not matched, try overlapping time ranges
            detected_start = detected['start_time']
            detected_end = detected['end_time']
            catalog_start = catalog['start_time']
            catalog_end = catalog['end_time']
            
            # Check if time ranges overlap with tolerance
            if (detected_start <= catalog_end + time_tolerance and 
                detected_end + time_tolerance >= catalog_start):
                matches.append({
                    'detected_idx': i,
                    'catalog_idx': j,
                    'detected_peak_flux': detected['peak_flux'],
                    'catalog_peak_flux': catalog['peak_flux'],
                    'flux_ratio': detected['peak_flux'] / catalog['peak_flux'] if catalog['peak_flux'] > 0 else float('inf'),
                    'detected_start': detected_start,
                    'catalog_start': catalog_start,
                    'detected_end': detected_end,
                    'catalog_end': catalog_end,
                    'time_diff': peak_time_diff.total_seconds() / 60 if peak_time_diff else None
                })
                detected_indices_matched.add(i)
                catalog_indices_matched.add(j)
    
    # Create DataFrames for matched and unmatched flares
    matches_df = pd.DataFrame(matches)
    
    # Get unmatched flares
    unmatched_detected = detected_flares.loc[~detected_flares.index.isin(detected_indices_matched)]
    unmatched_catalog = catalog_flares.loc[~catalog_flares.index.isin(catalog_indices_matched)]
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(unmatched_detected)
    false_negatives = len(unmatched_catalog)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    return {
        'matched_flares': matches_df,
        'unmatched_detected': unmatched_detected,
        'unmatched_catalog': unmatched_catalog,
        'metrics': metrics
    }


def calculate_detection_quality(comparison_results):
    """
    Calculate additional quality metrics for flare detection.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_detected_flares()
        
    Returns
    -------
    dict
        Dictionary with additional quality metrics
    """
    metrics = comparison_results['metrics'].copy()
    matched = comparison_results['matched_flares']
    
    if len(matched) > 0:
        # Calculate time difference statistics
        if 'time_diff' in matched.columns:
            time_diffs = matched['time_diff'].dropna().abs()
            metrics['mean_time_diff'] = time_diffs.mean()
            metrics['median_time_diff'] = time_diffs.median()
            metrics['max_time_diff'] = time_diffs.max()
        
        # Calculate flux ratio statistics
        flux_ratios = matched['flux_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        metrics['mean_flux_ratio'] = flux_ratios.mean()
        metrics['median_flux_ratio'] = flux_ratios.median()
        
        # Calculate duration difference statistics
        if all(col in matched.columns for col in ['detected_start', 'detected_end', 'catalog_start', 'catalog_end']):
            detected_durations = (matched['detected_end'] - matched['detected_start']).dt.total_seconds() / 60
            catalog_durations = (matched['catalog_end'] - matched['catalog_start']).dt.total_seconds() / 60
            duration_diffs = abs(detected_durations - catalog_durations)
            
            metrics['mean_duration_diff'] = duration_diffs.mean()
            metrics['median_duration_diff'] = duration_diffs.median()
    
    return metrics


def get_flare_class_distribution(flares):
    """
    Get the distribution of flare classes.
    
    Parameters
    ----------
    flares : pandas.DataFrame
        DataFrame with flare information, must have 'flare_class' column
        
    Returns
    -------
    pandas.Series
        Series with count of flares by class
    """
    if 'flare_class' in flares.columns:
        return flares['flare_class'].value_counts().sort_index()
    elif 'classification' in flares.columns:
        # Extract first character of classification
        flare_classes = flares['classification'].str[0]
        return flare_classes.value_counts().sort_index()
    else:
        return pd.Series()
