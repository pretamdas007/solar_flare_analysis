"""
Run All Algorithm Flowcharts - Simplified Version
Generates only professional algorithm flowcharts for all solar flare ML models
"""

import subprocess
import sys
from pathlib import Path
import time

def run_flowchart_script(script_name):
    """Run a flowchart script and return success status"""
    try:
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run all flowchart scripts"""
    
    print("🚀 Generating Professional Algorithm Flowcharts for Solar Flare ML Models")
    print("=" * 80)
    
    # List of flowchart scripts
    scripts = [
        "transformer_flowchart_only.py",
        "monte_carlo_flowchart_only.py",
        "gnn_flowchart_only.py",
        "contrastive_learning_flowchart_only.py",
        "bayesian_flowchart_only.py"
    ]
    
    successful_runs = 0
    total_scripts = len(scripts)
    start_time = time.time()
    
    print(f"Found {total_scripts} flowchart scripts to run...")
    print()
    
    # Run each script
    for i, script_name in enumerate(scripts, 1):
        print(f"[{i}/{total_scripts}] {script_name}")
        if run_flowchart_script(script_name):
            successful_runs += 1
        print()
    
    # Summary
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"🎯 FLOWCHART GENERATION COMPLETE!")
    print(f"Successfully generated {successful_runs}/{total_scripts} flowcharts")
    print(f"Total execution time: {total_time:.1f} seconds")
    
    # List generated files
    current_dir = Path.cwd()
    png_files = list(current_dir.glob("*_algorithm_flowchart.png"))
    
    if png_files:
        print(f"\n📊 Generated {len(png_files)} professional algorithm flowcharts:")
        for png_file in sorted(png_files):
            print(f"  ✅ {png_file.name}")
    else:
        print("\n⚠️  No flowchart files found")
    
    print(f"\n🎉 All professional algorithm flowcharts are ready!")
    print("All files saved in current directory with 300 DPI resolution.")

if __name__ == "__main__":
    main()
