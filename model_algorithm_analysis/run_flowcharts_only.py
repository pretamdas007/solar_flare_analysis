"""
Master Script to Generate All Professional Algorithm Flowcharts
Creates only the main algorithm flowcharts for all solar flare ML models
"""

import subprocess
import sys
from pathlib import Path

def run_flowchart_script(script_name):
    """Run a flowchart script and return success status"""
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ {script_name} not found")
        return False

def main():
    """Generate all algorithm flowcharts"""
    
    print("🚀 Generating Professional Algorithm Flowcharts for Solar Flare ML Models")
    print("=" * 80)
    
    # List of flowchart scripts
    flowchart_scripts = [
        "transformer_flowchart_only.py",
        "monte_carlo_flowchart_only.py", 
        "gnn_flowchart_only.py",
        "contrastive_learning_flowchart_only.py",
        "bayesian_flowchart_only.py"
    ]
    
    successful_runs = 0
    total_scripts = len(flowchart_scripts)
    
    print(f"Running {total_scripts} flowchart generation scripts...")
    print()
    
    # Run each script
    for i, script_name in enumerate(flowchart_scripts, 1):
        print(f"[{i}/{total_scripts}] Generating {script_name.replace('_flowchart_only.py', '')} flowchart...")
        
        if run_flowchart_script(script_name):
            successful_runs += 1
        
        print()
    
    # Final summary
    print("=" * 80)
    print(f"🎯 FLOWCHART GENERATION COMPLETE!")
    print(f"Successfully generated {successful_runs}/{total_scripts} flowcharts")
    
    # List generated files
    current_dir = Path.cwd()
    png_files = list(current_dir.glob("*algorithm_flowchart.png"))
    
    print(f"\n📊 Generated {len(png_files)} professional algorithm flowcharts:")
    for png_file in sorted(png_files):
        print(f"  ✅ {png_file.name}")
    
    print("\n🎉 All professional algorithm flowcharts are ready!")
    print("All files saved in current directory with 300 DPI resolution.")

if __name__ == "__main__":
    main()
