#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download sample GOES XRS data for solar flare analysis.

This script downloads GOES XRS data from the NOAA NCEI repository for a specified date range.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings
from config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download GOES XRS data')
    parser.add_argument('--satellite', type=str, choices=['goes16', 'goes17', 'goes18'],
                        default='goes16', help='GOES satellite number')
    parser.add_argument('--start_date', type=str, default='20220601',
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date in YYYYMMDD format (default: start_date)')
    parser.add_argument('--resolution', type=str, choices=['avg1m', 'avg5m'], 
                        default='avg1m', help='Data resolution')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: project data directory)')
    return parser.parse_args()


def download_file(url, output_path):
    """
    Download a file from a URL with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download
    output_path : str
        Path to save the file
    
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress bar
        with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        print(f"Downloaded {os.path.basename(output_path)} successfully")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_goes_xrs_data(satellite, date, resolution, output_dir):
    """
    Download GOES XRS data for a specific date.
    
    Parameters
    ----------
    satellite : str
        GOES satellite identifier (e.g., 'goes16')
    date : str
        Date in YYYYMMDD format
    resolution : str
        Data resolution ('avg1m' or 'avg5m')
    output_dir : str
        Directory to save downloaded files
    
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    # Parse date
    date_obj = datetime.strptime(date, '%Y%m%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')
    
    # Construct URL
    base_url = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    url = f"{base_url}/goes/goes{satellite[-2:]}/l2/{resolution}/{year}/{month}/{satellite}/"
    file_name = f"{satellite}_xrs_{resolution}_{date}_{date}.nc"
    file_url = f"{url}/{file_name}"
    
    # Set output path
    output_path = os.path.join(output_dir, file_name)
    
    # Download file
    return download_file(file_url, output_path)


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set default end date if not specified
    end_date = args.end_date if args.end_date else args.start_date
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else settings.DATA_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    # Download data for each date in the range
    current_date = start_date
    successful_downloads = 0
    total_days = (end_date - start_date).days + 1
    
    print(f"Downloading {args.satellite} {args.resolution} data from {args.start_date} to {end_date.strftime('%Y%m%d')}")
    print(f"Saving to {output_dir}")
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        print(f"\nDownloading data for {date_str} ({current_date.strftime('%Y-%m-%d')})")
        
        if download_goes_xrs_data(args.satellite, date_str, args.resolution, output_dir):
            successful_downloads += 1
        
        current_date += timedelta(days=1)
    
    # Print summary
    print("\nDownload summary:")
    print(f"  Total days: {total_days}")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Failed downloads: {total_days - successful_downloads}")
    
    if successful_downloads > 0:
        print(f"\nData files saved to {output_dir}")
        print("You can now use these files with the solar flare analysis pipeline.")


if __name__ == "__main__":
    main()
