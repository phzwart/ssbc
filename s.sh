#!/bin/bash

# Create ssbc project structure
PROJECT_NAME="ssbc"

# Create main directory

# Create directory structure
mkdir -p src/ssbc
mkdir -p tests
mkdir -p examples
mkdir -p docs

# Create root-level files
touch pyproject.toml
touch setup.py
touch README.md
touch LICENSE
touch requirements.txt
touch .gitignore

# Create src/ssbc files
touch src/ssbc/__init__.py
touch src/ssbc/core.py
touch src/ssbc/conformal.py
touch src/ssbc/simulation.py
touch src/ssbc/statistics.py
touch src/ssbc/hyperparameter.py
touch src/ssbc/visualization.py

# Create test files
touch tests/__init__.py
touch tests/test_core.py
touch tests/test_conformal.py
touch tests/test_simulation.py
touch tests/test_statistics.py
touch tests/test_visualization.py

# Create example files
touch examples/basic_usage.py
touch examples/hyperparameter_sweep.ipynb

# Create docs files
touch docs/conf.py
touch docs/index.rst

echo "✅ Project structure created successfully!"
echo ""
echo "Directory structure:"
tree -L 3 $PROJECT_NAME 2>/dev/null || find $PROJECT_NAME -type f -o -type d | sort

cd ..
