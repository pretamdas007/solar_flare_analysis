"""
__init__ file for validation package.
"""

from .catalog_validation import (
    download_noaa_flare_catalog, 
    parse_noaa_event_list,
    compare_detected_flares,
    calculate_detection_quality,
    get_flare_class_distribution
)
