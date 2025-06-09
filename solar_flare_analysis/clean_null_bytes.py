#!/usr/bin/env python3
"""
Script to remove null bytes from data_loader.py file
"""

import os

def clean_null_bytes(file_path):
    """Remove null bytes from a file"""
    try:
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        print(f"Original file size: {len(content)} bytes")
        print(f"Null bytes found: {content.count(b'\\x00')}")
        
        # Remove null bytes
        cleaned_content = content.replace(b'\\x00', b'')
        
        print(f"Cleaned file size: {len(cleaned_content)} bytes")
        
        # Write back the cleaned content
        with open(file_path, 'wb') as f:
            f.write(cleaned_content)
        
        print(f"File cleaned successfully: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error cleaning file: {e}")
        return False

if __name__ == "__main__":
    file_path = r"c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\src\data_processing\data_loader.py"
    clean_null_bytes(file_path)
