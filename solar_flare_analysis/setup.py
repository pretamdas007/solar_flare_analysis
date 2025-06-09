#!/usr/bin/env python3
"""
Setup script for Solar Flare Analysis package
Supports optional dependency groups for different use cases
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    with open(os.path.join(os.path.dirname(__file__), '__init__.py'), 'r') as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    return "0.1.0"

# Read README for long description
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Advanced solar flare analysis with machine learning"

# Core dependencies (essential for basic functionality)
CORE_REQUIREMENTS = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.3.0",
    "netCDF4>=1.6.4",
    "xarray>=2023.1.0",
    "requests>=2.31.0",
    "tqdm>=4.65.0",
]

# Machine Learning dependencies
ML_REQUIREMENTS = [
    "tensorflow>=2.13.0",
    "tf-keras>=2.13.0",
    "tensorflow-probability>=0.20.0",
    "numba>=0.57.0",
    "lmfit>=1.2.0",
]

# Web API dependencies
WEB_REQUIREMENTS = [
    "flask>=2.3.0",
    "flask-cors>=4.0.0",
    "werkzeug>=2.3.0",
    "gunicorn>=21.0.0",
    "waitress>=2.1.0",
    "flasgger>=0.9.7",
    "pydantic>=2.0.0",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "ipython>=8.14.0",
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.4.0",
    "pre-commit>=3.3.0",
    "memory-profiler>=0.61.0",
]

# Scientific analysis dependencies
SCIENCE_REQUIREMENTS = [
    "astropy>=5.3.0",
    "powerlaw>=1.5",
    "statsmodels>=0.14.0",
    "seaborn>=0.12.0",
    "h5py>=3.9.0",
    "sympy>=1.12",
    "networkx>=3.1",
]

# Visualization dependencies
VIZ_REQUIREMENTS = [
    "plotly>=5.15.0",
    "bokeh>=3.2.0",
    "pillow>=10.0.0",
]

# Performance dependencies
PERFORMANCE_REQUIREMENTS = [
    "dask>=2023.5.0",
    "psutil>=5.9.0",
    # "cupy>=12.0.0",  # Requires CUDA - commented out for compatibility
]

# Documentation dependencies
DOCS_REQUIREMENTS = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]

# All optional dependencies combined
ALL_REQUIREMENTS = (
    ML_REQUIREMENTS + 
    WEB_REQUIREMENTS + 
    DEV_REQUIREMENTS + 
    SCIENCE_REQUIREMENTS + 
    VIZ_REQUIREMENTS + 
    PERFORMANCE_REQUIREMENTS + 
    DOCS_REQUIREMENTS
)

setup(
    name="solar-flare-analysis",
    version=get_version(),
    author="Solar Flare Analysis Team",
    author_email="support@solarflareanalysis.org",
    description="Advanced solar flare analysis with machine learning capabilities",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/pretamdas007/solar_flare_analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "ml": ML_REQUIREMENTS,
        "web": WEB_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
        "science": SCIENCE_REQUIREMENTS,
        "viz": VIZ_REQUIREMENTS,
        "performance": PERFORMANCE_REQUIREMENTS,
        "docs": DOCS_REQUIREMENTS,
        "all": ALL_REQUIREMENTS,
    },
    entry_points={
        "console_scripts": [
            "solar-flare-analysis=main:main",
            "goes-download=scripts.download_goes_data:main",
            "flare-server=ml_app.enhanced_python_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "solar_flare_analysis": [
            "config/*.py",
            "data/sample_data/*",
            "models/*.h5",
            "notebooks/*.ipynb",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/pretamdas007/solar_flare_analysis/issues",
        "Source": "https://github.com/pretamdas007/solar_flare_analysis",
        "Documentation": "https://solar-flare-analysis.readthedocs.io/",
    },
)
