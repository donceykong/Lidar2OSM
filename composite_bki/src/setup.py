from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Define the extension module
extensions = [
    Extension(
        "composite_bki_cpp",
        sources=[
            "composite_bki_cpp/bindings.pyx",
            "composite_bki_cpp/continuous_bki.cpp",
        ],
        include_dirs=[
            np.get_include(), 
            "composite_bki_cpp",
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="composite-bki",
    version="2.1.0",
    description="Composite BKI - Unified Continuous/Single Semantic Mapping",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    install_requires=[
        "numpy", 
        "cython"
    ],
    py_modules=["cli"],
    entry_points={
        'console_scripts': [
            'composite-bki=cli:main',
        ],
    },
)
