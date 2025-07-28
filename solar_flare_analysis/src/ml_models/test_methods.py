import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_flare_model import SolarFlareStormDetector

# Create instance
detector = SolarFlareStormDetector(
    data_dir=r'c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\data\XRS'
)

# Check available methods
print("Available methods:")
methods = [method for method in dir(detector) if not method.startswith('_')]
for method in sorted(methods):
    print(f"  - {method}")

# Check if build_model exists
if hasattr(detector, 'build_model'):
    print("✅ build_model method exists")
else:
    print("❌ build_model method NOT found")
