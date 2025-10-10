# Badges for README.md

Add these badges to the top of your README.md to show project status:

```markdown
# SSBC - Small Sample Beta Correction

[![CI](https://github.com/phzwart/ssbc/workflows/CI/badge.svg)](https://github.com/phzwart/ssbc/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/phzwart/ssbc/workflows/Pre-commit/badge.svg)](https://github.com/phzwart/ssbc/actions/workflows/pre-commit.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- After setting up Codecov, add this: -->
<!-- [![codecov](https://codecov.io/gh/phzwart/ssbc/branch/main/graph/badge.svg)](https://codecov.io/gh/phzwart/ssbc) -->

<!-- After publishing to PyPI, add this: -->
<!-- [![PyPI version](https://badge.fury.io/py/ssbc.svg)](https://badge.fury.io/py/ssbc) -->
<!-- [![Downloads](https://pepy.tech/badge/ssbc)](https://pepy.tech/project/ssbc) -->
```

## Optional: Add Contributing Section

Add this section to your README.md:

```markdown
## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies:
   ```bash
   uv sync --extra dev --extra test
   pre-commit install
   ```
4. Make your changes
5. Run quality checks:
   ```bash
   just qa
   ```
6. Commit your changes (pre-commit hooks will run automatically)
7. Push to your fork and submit a Pull Request

See [CI_CD_GUIDE.md](CI_CD_GUIDE.md) for more details on our development workflow.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/phzwart/ssbc.git
cd ssbc

# Run the setup script
./setup-hooks.sh

# Or manually:
uv sync --extra dev --extra test
pre-commit install
```

### Code Quality

This project uses:
- **Ruff** for linting and formatting
- **Ty** for type checking  
- **Pytest** for testing
- **Pre-commit** hooks for automated checks
- **GitHub Actions** for CI/CD

All checks must pass before merging.
```

