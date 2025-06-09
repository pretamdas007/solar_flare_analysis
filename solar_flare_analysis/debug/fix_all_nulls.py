#!/usr/bin/env python3
"""
Comprehensive null byte detector and cleaner for the solar flare analysis project
"""

import os
import glob

def scan_and_clean_file(file_path):
    """Scan and clean a single file for null bytes"""
    try:
        # Read file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Check for null bytes
        null_count = content.count(b'\x00')
        
        if null_count > 0:
            print(f"FOUND NULL BYTES: {file_path} - {null_count} null bytes")
            
            # Remove null bytes
            cleaned_content = content.replace(b'\x00', b'')
            
            # Write cleaned content back
            with open(file_path, 'wb') as f:
                f.write(cleaned_content)
                
            print(f"  -> Cleaned {null_count} null bytes from {file_path}")
            return True
        else:
            print(f"CLEAN: {file_path}")
            return False
            
    except Exception as e:
        print(f"ERROR: {file_path} - {e}")
        return False

def main():
    """Main function to scan all Python files"""
    base_path = r"c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis"
    
    # Find all Python files
    python_files = []
    
    # All __init__.py files
    init_files = glob.glob(os.path.join(base_path, "**", "__init__.py"), recursive=True)
    python_files.extend(init_files)
    
    # All .py files in src directory
    src_files = glob.glob(os.path.join(base_path, "src", "**", "*.py"), recursive=True)
    python_files.extend(src_files)
    
    # Main directory Python files
    main_files = glob.glob(os.path.join(base_path, "*.py"))
    python_files.extend(main_files)
    
    # Remove duplicates
    python_files = list(set(python_files))
    
    print(f"Scanning {len(python_files)} Python files for null bytes...")
    print("=" * 60)
    
    cleaned_files = 0
    for file_path in sorted(python_files):
        if scan_and_clean_file(file_path):
            cleaned_files += 1
    
    print("=" * 60)
    print(f"Summary: Cleaned {cleaned_files} files with null bytes")
    
    # Now test import
    print("\nTesting import after cleaning...")
    try:
        import sys
        sys.path.insert(0, base_path)
        
        print("1. Testing solar_flare_analysis import...")
        import solar_flare_analysis
        print("   SUCCESS!")
        
        print("2. Testing src import...")
        from solar_flare_analysis import src
        print("   SUCCESS!")
        
        print("3. Testing data_processing import...")
        from solar_flare_analysis.src import data_processing
        print("   SUCCESS!")
        
        print("4. Testing GOESDataLoader import...")
        from solar_flare_analysis.src.data_processing.data_loader import GOESDataLoader
        print("   SUCCESS!")
        
        print("\nAll imports successful! 🎉")
        
    except Exception as e:
        print(f"Import still failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
