"""
Master Model Testing Suite
Professional testing suite that runs all individual model testers
and the comprehensive comparison dashboard
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime

class MasterModelTester:
    """
    Master controller for running all model testing suites
    """
    
    def __init__(self):
        self.test_scripts = {
            'transformer': 'transformer_model_tester.py',
            'monte_carlo': 'monte_carlo_model_tester.py',
            'graph': 'graph_neural_model_tester.py',
            'contrastive': 'contrastive_learning_model_tester.py',
            'bayesian': 'bayesian_model_tester.py',
            'comparison': 'comprehensive_model_comparator.py'
        }
        
        self.results_dir = Path("model_testing_results")
        
    def setup_results_directory(self):
        """Create results directory if it doesn't exist"""
        self.results_dir.mkdir(exist_ok=True)
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def check_scripts_exist(self):
        """Check if all testing scripts exist"""
        missing_scripts = []
        for name, script in self.test_scripts.items():
            if not Path(script).exists():
                missing_scripts.append(f"{name}: {script}")
        
        if missing_scripts:
            print("❌ Missing testing scripts:")
            for script in missing_scripts:
                print(f"   • {script}")
            return False
        
        print("✅ All testing scripts found!")
        return True
    
    def run_individual_tester(self, script_name, model_name):
        """Run an individual model tester"""
        print(f"\n{'='*60}")
        print(f"🔬 Running {model_name} Model Tester")
        print(f"{'='*60}")
        
        try:
            # Run the script
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {model_name} testing completed successfully!")
                
                # Move generated files to results directory
                self.organize_output_files(model_name.lower().replace(' ', '_'))
                
                return True
            else:
                print(f"❌ {model_name} testing failed!")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} testing timed out after 5 minutes!")
            return False
        except Exception as e:
            print(f"❌ Error running {model_name} tester: {str(e)}")
            return False
    
    def organize_output_files(self, model_prefix):
        """Organize output files into the results directory"""
        # Common output file patterns
        patterns = [
            f"*{model_prefix}*.png",
            f"*{model_prefix}*.txt",
            f"*{model_prefix}*.json",
            f"*{model_prefix}*report*",
            f"*{model_prefix}*analysis*"
        ]
        
        moved_files = []
        for pattern in patterns:
            for file_path in Path(".").glob(pattern):
                if file_path.is_file():
                    dest_path = self.results_dir / file_path.name
                    try:
                        file_path.rename(dest_path)
                        moved_files.append(dest_path.name)
                    except Exception as e:
                        print(f"⚠️ Could not move {file_path}: {str(e)}")
        
        if moved_files:
            print(f"📁 Moved {len(moved_files)} output files to {self.results_dir}")
    
    def run_all_tests(self, models_to_test=None):
        """Run all model tests or specific ones"""
        if models_to_test is None:
            models_to_test = list(self.test_scripts.keys())
        
        print("🚀 Master Model Testing Suite")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧪 Models to test: {', '.join(models_to_test)}")
        
        # Setup
        self.setup_results_directory()
        
        if not self.check_scripts_exist():
            return
        
        # Run tests
        successful_tests = []
        failed_tests = []
        
        for model_name in models_to_test:
            if model_name in self.test_scripts:
                script_name = self.test_scripts[model_name]
                model_display_name = model_name.replace('_', ' ').title()
                
                success = self.run_individual_tester(script_name, model_display_name)
                
                if success:
                    successful_tests.append(model_display_name)
                else:
                    failed_tests.append(model_display_name)
            else:
                print(f"⚠️ Unknown model: {model_name}")
                failed_tests.append(model_name)
        
        # Summary
        self.print_summary(successful_tests, failed_tests)
    
    def print_summary(self, successful_tests, failed_tests):
        """Print testing summary"""
        print(f"\n{'='*60}")
        print("📊 TESTING SUMMARY")
        print(f"{'='*60}")
        
        print(f"✅ Successful Tests ({len(successful_tests)}):")
        for test in successful_tests:
            print(f"   • {test}")
        
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test}")
        
        print(f"\n📁 All results saved to: {self.results_dir}")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # List generated files
        result_files = list(self.results_dir.glob("*"))
        if result_files:
            print(f"\n📄 Generated Files ({len(result_files)}):")
            for file_path in sorted(result_files):
                print(f"   • {file_path.name}")

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Master Model Testing Suite for Solar Flare ML Models')
    parser.add_argument('--models', nargs='+', 
                       choices=['transformer', 'monte_carlo', 'graph', 'contrastive', 'bayesian', 'comparison'],
                       help='Specific models to test (default: all)')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip the comprehensive comparison (run only individual tests)')
    
    args = parser.parse_args()
    
    # Initialize master tester
    tester = MasterModelTester()
    
    # Determine which models to test
    if args.models:
        models_to_test = args.models
    else:
        # Test all models
        models_to_test = ['transformer', 'monte_carlo', 'graph', 'contrastive', 'bayesian']
        
        # Add comparison unless skipped
        if not args.skip_comparison:
            models_to_test.append('comparison')
    
    # Run the tests
    tester.run_all_tests(models_to_test)

if __name__ == "__main__":
    main()
