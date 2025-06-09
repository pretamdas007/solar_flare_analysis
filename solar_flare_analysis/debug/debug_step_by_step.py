#!/usr/bin/env python3
"""
Step-by-step import debugging
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

print("Python path:")
for p in sys.path[:5]:  # Show first 5 entries
    print(f"  {p}")

print("\nTesting imports step by step...")

try:
    print("1. Importing solar_flare_analysis...")
    import solar_flare_analysis
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")

try:
    print("2. Importing solar_flare_analysis.src...")
    from solar_flare_analysis import src
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")

try:
    print("3. Importing solar_flare_analysis.src.data_processing...")
    from solar_flare_analysis.src import data_processing
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")

try:
    print("4. Importing GOESDataLoader...")
    from solar_flare_analysis.src.data_processing.data_loader import GOESDataLoader
    print("   SUCCESS")
except Exception as e:
    print(f"   FAILED: {e}")

print("\nTest complete.")
