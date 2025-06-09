#!/usr/bin/env python3
"""
Check specific file for null bytes
"""

import os

def check_file_bytes(file_path):
    """Check file for null bytes and other issues"""
    try:
        # Read in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        print(f"File: {file_path}")
        print(f"Size: {len(content)} bytes")
        print(f"Content (hex): {content.hex()}")
        print(f"Content (repr): {repr(content)}")
        
        # Count null bytes
        null_count = content.count(b'\x00')
        print(f"Null bytes: {null_count}")
        
        if null_count > 0:
            print("Found null bytes! Cleaning...")
            clean_content = content.replace(b'\x00', b'')
            with open(file_path, 'wb') as f:
                f.write(clean_content)
            print("File cleaned.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_path = r"c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\src\__init__.py"
    check_file_bytes(file_path)
