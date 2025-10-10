# CI/CD Setup Guide

This document explains the CI/CD setup for the SSBC project.

## 🚀 Quick Start

### 1. Install Pre-commit Hooks (Local Development)

Pre-commit hooks run automatically before each commit to catch issues early:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run hooks on all files manually
pre-commit run --all-files
```

### 2. What Gets Checked Locally

When you commit, the following checks run automatically:
- **Ruff** - Linting and code formatting (replaces flake8, black, isort)
- **Ty** - Type checking
- **Trailing whitespace** - Removes trailing spaces
- **File endings** - Ensures files end with newline
- **YAML/TOML validation** - Checks syntax
- **Large files check** - Prevents committing files >1MB
- **Security** - Bandit security checks
- **Spelling** - Codespell for common typos

## 📋 GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)
Runs on every push and pull request to `main` or `develop`:

1. **Lint and Format Check**
   - Checks code formatting with `ruff format --check`
   - Runs linting with `ruff check`

2. **Type Checking**
   - Validates types with `ty check`

3. **Tests**
   - Runs pytest on Python 3.10, 3.11, 3.12, and 3.13
   - Ensures compatibility across all supported versions

4. **Coverage**
   - Generates coverage report
   - Uploads to Codecov (if configured)

5. **Build**
   - Builds the package to ensure it's installable
   - Uploads artifacts for inspection

### Pre-commit Workflow (`.github/workflows/pre-commit.yml`)
Runs all pre-commit hooks in CI to catch issues:
- Useful for contributors who don't have pre-commit installed locally
- Ensures all code meets quality standards

### CodeQL Workflow (`.github/workflows/codeql.yml`)
Security scanning:
- Runs on push/PR to main branches
- Weekly scheduled scan
- Identifies potential security vulnerabilities

### Release Workflow (`.github/workflows/release.yml`)
Automated publishing when you create a git tag:

```bash
# Create and push a release tag
just tag  # Uses version from pyproject.toml
# OR manually:
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

This will:
- Run tests
- Build the package
- Publish to PyPI (when configured)
- Create a GitHub release with notes

### Dependabot (`.github/dependabot.yml`)
Automatically creates PRs to:
- Update Python dependencies weekly
- Update GitHub Actions weekly

## 🔧 Configuration Files

### `.pre-commit-config.yaml`
Configures local pre-commit hooks. Update versions periodically:
```bash
pre-commit autoupdate
```

### `pyproject.toml`
Contains configuration for:
- **Ruff**: Line length (120), enabled rules
- **Bandit**: Security exclusions and skips
- **Ty**: Type checking rules

## 🎯 Usage Examples

### Run QA checks locally (mimics CI)
```bash
just qa
```

### Run specific checks
```bash
# Format code
uv run --extra test ruff format .

# Lint code
uv run --extra test ruff check . --fix

# Type check
uv run --extra test ty check .

# Run tests
uv run --extra test pytest

# Coverage
just coverage
```

### Test all Python versions
```bash
just testall
```

### Skip pre-commit hooks (not recommended)
```bash
git commit --no-verify -m "message"
```

## 📦 PyPI Publishing Setup

To enable automatic PyPI publishing on releases:

1. **Set up Trusted Publishing** (recommended):
   - Go to PyPI → Your Account → Publishing
   - Add a new publisher:
     - PyPI Project: `ssbc`
     - Owner: `phzwart`
     - Repository: `ssbc`
     - Workflow: `release.yml`
     - Environment: `release`

2. **Alternative: Using API Token**:
   - Generate PyPI API token
   - Add to GitHub Secrets as `PYPI_API_TOKEN`
   - Uncomment token auth in `release.yml`

3. **Uncomment the publish step** in `.github/workflows/release.yml`

## 🔍 Monitoring

- **GitHub Actions**: Check the "Actions" tab on GitHub
- **Coverage**: Results uploaded to Codecov (configure with token)
- **Security**: CodeQL alerts in "Security" tab
- **Dependencies**: Dependabot PRs appear automatically

## 🛠️ Customization

### Add more Python versions
Edit the matrix in `.github/workflows/ci.yml`:
```yaml
matrix:
  python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]
```

### Add more pre-commit hooks
Browse available hooks at https://pre-commit.com/hooks.html

### Adjust ruff rules
Edit `[tool.ruff.lint]` in `pyproject.toml`

## 📚 Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [uv documentation](https://docs.astral.sh/uv/)
- [Codecov setup](https://docs.codecov.com/docs)

