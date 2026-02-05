"""
Setup script for Composite BKI C++ package.

This creates a pip-installable package with both:
1. Python library API (import composite_bki_cpp)
2. Command-line tool (composite-bki command)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os

# Package metadata  
PACKAGE_NAME = "composite-bki"
VERSION = "2.0.0"
DESCRIPTION = "High-performance C++-accelerated semantic segmentation for LiDAR point clouds"

# C++ extension module
extensions = [
    Extension(
        name="composite_bki_cpp",  # Module name
        sources=[
            "composite_bki_cpp/composite_bki_wrapper.pyx",
            "composite_bki_cpp/composite_bki.cpp",
            "composite_bki_cpp/composite_bki_utils.cpp"
        ],
        include_dirs=[
            "composite_bki_cpp",
            np.get_include()
        ],
        language="c++",
        extra_compile_args=[
            "-std=c++11",
            "-O3",
            "-march=native",
            "-ffast-math",
            "-fopenmp",
            "-DUSE_OPENMP"
        ],
        extra_link_args=[
            "-std=c++11",
            "-fopenmp"
        ],
    )
]

# Long description from README
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Composite BKI Team",
    author_email="your.email@example.com",
    url="https://github.com/yourrepo/composite-bki",
    
    # Python modules (cli.py, __init__.py at root)
    py_modules=["cli", "__init__"],
    package_data={
        '': ['composite_bki_cpp/*.cpp', 
             'composite_bki_cpp/*.hpp', 
             'composite_bki_cpp/*.pyx',
             'composite_bki_cpp/*.cu',
             'configs/*.yaml']
    },
    include_package_data=True,
    
    # Cython extensions
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'embedsignature': True,
        }
    ),
    
    # Dependencies
    install_requires=[
        "numpy>=1.19.0",
    ],
    
    # Build dependencies
    setup_requires=[
        "Cython>=0.29.0",
        "numpy>=1.19.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },
    
    # Command-line scripts
    entry_points={
        'console_scripts': [
            'composite-bki=cli:main',
        ],
    },
    
    # Metadata
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
    keywords="lidar point-cloud segmentation bayesian osm semantic machine-learning",
    project_urls={
        "Documentation": "https://github.com/yourrepo/composite-bki/blob/main/README.md",
        "Source": "https://github.com/yourrepo/composite-bki",
        "Bug Tracker": "https://github.com/yourrepo/composite-bki/issues",
    },
    
    zip_safe=False,  # Required for Cython extensions
)
