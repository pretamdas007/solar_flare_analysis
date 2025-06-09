#!/usr/bin/env python3
"""
Check encoding and characters in data_loader.py
"""

import os

def analyze_file(file_path):
    """Analyze file for encoding issues"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        print(f"File: {file_path}")
        print(f"Size: {len(content)} bytes")
        
        # Check for various problematic bytes
        null_bytes = content.count(b'\x00')
        print(f"Null bytes (\\x00): {null_bytes}")
        
        # Check for other control characters that might cause issues
        for i in range(32):
            if i not in [9, 10, 13]:  # Skip tab, newline, carriage return
                count = content.count(bytes([i]))
                if count > 0:
                    print(f"Control char \\x{i:02x}: {count}")
        
        # Try to decode as UTF-8
        try:
            text_content = content.decode('utf-8')
            print("UTF-8 decode: SUCCESS")
        except UnicodeDecodeError as e:
            print(f"UTF-8 decode: FAILED - {e}")
            
        # Try to compile as Python
        try:
            compile(content, file_path, 'exec')
            print("Python compile: SUCCESS")
        except SyntaxError as e:
            print(f"Python compile: FAILED - {e}")
            
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    file_path = r"c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\src\data_processing\data_loader.py"
    analyze_file(file_path)
