# Makefile for Composite BKI C++ Implementation

.PHONY: build install clean test help

help:
	@echo "Composite BKI C++ Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build Cython extension in-place"
	@echo "  install     - Install package"
	@echo "  clean       - Remove build artifacts"
	@echo "  test        - Run example tests"
	@echo "  help        - Show this help message"

build:
	@echo "Building Cython extension..."
	python setup.py build_ext --inplace
	@echo "Build complete! Extension ready to use."

install:
	@echo "Installing package..."
	pip install .
	@echo "Installation complete!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f composite_bki_cpp*.so
	rm -f composite_bki_cpp*.pyd
	rm -f composite_bki_wrapper.cpp
	rm -f *.o
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

test: build
	@echo "Running example tests..."
	python example_usage.py
	@echo "Tests complete!"

rebuild: clean build
	@echo "Rebuild complete!"

# Development helpers
format:
	@echo "Formatting C++ code..."
	clang-format -i *.cpp *.hpp
	@echo "Format complete!"

check-deps:
	@echo "Checking dependencies..."
	@python -c "import numpy; print('numpy:', numpy.__version__)" || echo "numpy not found!"
	@python -c "import Cython; print('Cython:', Cython.__version__)" || echo "Cython not found!"
	@which g++ > /dev/null && echo "g++ found" || echo "g++ not found!"
