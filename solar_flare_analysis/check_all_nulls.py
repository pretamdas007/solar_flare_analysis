#!/usr/bin/env python3
"""
Script to check and clean null bytes from all __init__.py files in the import path
"""

import os

def check_and_clean_file(file_path):
    """Check and clean null bytes from a file"""
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
            
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        null_count = content.count(b'\\x00')
        print(f"{file_path}: {len(content)} bytes, {null_count} null bytes")
        
        if null_count > 0:
            # Remove null bytes
            cleaned_content = content.replace(b'\\x00', b'')
            
            # Write back the cleaned content
            with open(file_path, 'wb') as f:
                f.write(cleaned_content)
            
            print(f"  -> Cleaned {null_count} null bytes")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

if __name__ == "__main__":
    base_path = r"c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis"
    
    # Check all __init__.py files in the import path
    files_to_check = [
        os.path.join(base_path, "__init__.py"),
        os.path.join(base_path, "src", "__init__.py"),
        os.path.join(base_path, "src", "data_processing", "__init__.py"),
        os.path.join(base_path, "src", "data_processing", "data_loader.py"),
        os.path.join(base_path, "src", "ml_models", "__init__.py"),
        os.path.join(base_path, "src", "visualization", "__init__.py")
    ]
    
    print("Checking files for null bytes...")
    for file_path in files_to_check:
        check_and_clean_file(file_path)
