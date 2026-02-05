#!/bin/bash
# Build script for multi-threaded Composite BKI

set -e  # Exit on error

echo "=================================="
echo "Building Multi-Threaded Composite BKI"
echo "=================================="
echo ""

# Check if in virtual environment
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "Consider activating your environment first:"
    echo "  conda activate lidar2osm_env"
    echo ""
fi

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "❌ g++ not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y build-essential g++ libgomp1
else
    echo "✅ g++ found: $(g++ --version | head -1)"
fi

# Check for OpenMP
if echo | gcc -fopenmp -E -dM - | grep -q OPENMP; then
    echo "✅ OpenMP available"
else
    echo "⚠️  OpenMP not detected, installing libgomp1..."
    sudo apt-get install -y libgomp1
fi

echo ""
echo "Installing Python dependencies..."
pip install numpy cython

echo ""
echo "Cleaning previous build..."
make clean 2>/dev/null || rm -rf build/ *.so composite_bki_wrapper.cpp 2>/dev/null

echo ""
echo "Building extension..."
python setup.py build_ext --inplace

echo ""
echo "=================================="
if [ -f composite_bki_cpp*.so ]; then
    echo "✅ BUILD SUCCESSFUL!"
    echo "=================================="
    echo ""
    echo "Extension file:"
    ls -lh composite_bki_cpp*.so
    echo ""
    echo "Testing import..."
    python -c "import composite_bki_cpp; print('✅ Import successful!')" || echo "❌ Import failed"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: make test"
    echo "  2. Or run example: python example_usage.py"
else
    echo "❌ BUILD FAILED"
    echo "=================================="
    exit 1
fi
