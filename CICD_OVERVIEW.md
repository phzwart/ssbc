# CI/CD Overview - SSBC Project

## 🎯 What Was Set Up

Your repository now has a **comprehensive, production-ready CI/CD pipeline** with:

### 🔄 Automated Quality Checks
- **Pre-commit hooks** that run locally before each commit
- **GitHub Actions** that validate every push and pull request
- **Security scanning** to catch vulnerabilities
- **Dependency updates** via Dependabot

### 📊 What Gets Checked

#### Local (Pre-commit Hooks)
Every time you commit, these checks run automatically:
- ✨ **Ruff** - Code formatting & linting (fast!)
- 🔍 **Ty** - Type checking
- 🔒 **Bandit** - Security vulnerability scanning
- 📝 File quality checks (trailing spaces, EOF, etc.)
- 📖 **Codespell** - Catch typos

#### CI Pipeline (GitHub Actions)
On every push/PR, these workflows run:
- 🧹 **Lint & Format** - Code style validation
- 🔍 **Type Check** - Static type analysis
- 🧪 **Tests** - Run on Python 3.10, 3.11, 3.12, 3.13
- 📊 **Coverage** - Track test coverage
- 📦 **Build** - Verify package builds correctly
- 🔒 **CodeQL** - Advanced security analysis

## 📁 File Structure

```
.github/
├── workflows/
│   ├── ci.yml              # Main CI pipeline
│   ├── pre-commit.yml      # Pre-commit validation
│   ├── codeql.yml          # Security scanning
│   └── release.yml         # Automated releases
├── ISSUE_TEMPLATE/
│   ├── bug_report.md       # Bug report template
│   └── feature_request.md  # Feature request template
├── PULL_REQUEST_TEMPLATE.md
└── dependabot.yml          # Dependency updates

.pre-commit-config.yaml     # Local hooks config
pyproject.toml             # Updated with bandit & dev deps
setup-hooks.sh             # Quick setup script

Documentation:
├── CI_CD_GUIDE.md         # Detailed usage guide
├── CICD_SETUP_SUMMARY.md  # Setup checklist
├── README_BADGES.md       # Badge snippets
└── CICD_OVERVIEW.md       # This file
```

## 🚀 Getting Started

### 1. Initial Setup (One Time)

Run the setup script to install and configure pre-commit hooks:

```bash
./setup-hooks.sh
```

Or manually:
```bash
uv sync --extra dev --extra test
uv run --extra dev pre-commit install
uv run --extra dev pre-commit run --all-files
```

### 2. Daily Workflow

Once set up, your workflow becomes:

```bash
# Make changes to your code
vim src/ssbc/core.py

# Add files (pre-commit hooks run automatically on commit)
git add .
git commit -m "feat: Add new feature"

# If hooks find issues, they'll auto-fix what they can
# Review the changes and commit again
git add .
git commit -m "feat: Add new feature"

# Push to trigger CI
git push
```

### 3. Before Committing

Run quality checks manually:
```bash
just qa  # Runs all QA checks defined in justfile
```

Or individually:
```bash
uv run --extra test ruff format .        # Format code
uv run --extra test ruff check . --fix   # Lint and fix
uv run --extra test ty check .           # Type check
uv run --extra test pytest               # Run tests
```

## 🏗️ CI/CD Workflows Explained

### Main CI (`ci.yml`)
**Triggers:** Push to main/develop, PRs to main/develop

**Jobs:**
1. **Lint** - Check code formatting and style
2. **Type Check** - Validate type annotations
3. **Test** - Run pytest on all supported Python versions
4. **Coverage** - Generate coverage report
5. **Build** - Verify package builds

**Duration:** ~3-5 minutes

### Pre-commit (`pre-commit.yml`)
**Triggers:** Push and PRs

Runs all pre-commit hooks in CI to catch issues from contributors who don't have pre-commit installed locally.

### CodeQL Security (`codeql.yml`)
**Triggers:** Push, PR, weekly schedule (Mondays)

Advanced security analysis to detect vulnerabilities and coding errors.

### Release (`release.yml`)
**Triggers:** Git tags (e.g., `v0.1.0`)

**What it does:**
1. Runs all tests
2. Builds package
3. Publishes to PyPI (when configured)
4. Creates GitHub release with notes

**Usage:**
```bash
# Update version in pyproject.toml, then:
just tag  # Creates tag from pyproject.toml version and pushes
```

## 🔧 Configuration

### Pre-commit Hooks (`.pre-commit-config.yaml`)

Current hooks:
- `ruff` - Linting & formatting
- `ruff-format` - Code formatting
- `ty-check` - Type checking (local hook)
- `bandit` - Security checks
- `codespell` - Spell checking
- Various file checks

**Update hooks:**
```bash
uv run --extra dev pre-commit autoupdate
```

### Ruff Configuration (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP"]
```

### Bandit Security (`pyproject.toml`)

```toml
[tool.bandit]
exclude_dirs = ["tests", "htmlcov", ".venv", "venv"]
skips = ["B101"]  # Skip assert_used in tests
```

## 📋 Checklist for Production

- [x] Pre-commit hooks configured
- [x] GitHub Actions workflows set up
- [x] Security scanning enabled
- [x] Dependency updates automated
- [x] Issue/PR templates created
- [ ] **Configure Codecov** (optional)
  - Sign up at https://codecov.io
  - Add `CODECOV_TOKEN` to GitHub secrets
- [ ] **Configure PyPI publishing**
  - Set up Trusted Publishing on PyPI
  - Uncomment publish step in `release.yml`
- [ ] **Add badges to README**
  - See `README_BADGES.md` for snippets

## 🛠️ Common Tasks

### Skip Pre-commit (Emergency Only)
```bash
git commit --no-verify -m "Emergency fix"
```

### Run Specific Hook
```bash
uv run --extra dev pre-commit run ruff --all-files
```

### Update Dependencies
Dependabot will create PRs automatically, or manually:
```bash
uv lock --upgrade
```

### Create a Release
```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add pyproject.toml
git commit -m "chore: Bump version to 0.2.0"

# 3. Tag and push
just tag

# GitHub Actions will handle the rest!
```

## 📊 Monitoring

### GitHub Actions
View workflow runs: https://github.com/phzwart/ssbc/actions

### Security Alerts
View CodeQL findings: https://github.com/phzwart/ssbc/security/code-scanning

### Dependabot
View dependency PRs: https://github.com/phzwart/ssbc/pulls

### Coverage (after Codecov setup)
View coverage: https://codecov.io/gh/phzwart/ssbc

## 🔍 Troubleshooting

### Pre-commit fails
- Most failures auto-fix issues
- Review changes: `git diff`
- Commit auto-fixes: `git add . && git commit -m "style: Auto-fix from pre-commit"`

### CI fails on GitHub but passes locally
- Ensure you've committed all changes
- Check Python version compatibility
- Review GitHub Actions logs for details

### Type checking errors
- Fix type annotations in your code
- Or adjust `[tool.ty]` rules in `pyproject.toml`

### Can't push due to hooks
- Fix the issues (recommended)
- Or use `--no-verify` (not recommended)

## 📚 Documentation Files

- **CI_CD_GUIDE.md** - Comprehensive usage guide
- **CICD_SETUP_SUMMARY.md** - Setup checklist and configuration
- **README_BADGES.md** - Badge snippets for README
- **CICD_OVERVIEW.md** - This file (high-level overview)

## 🎉 What's Next?

1. **Run setup:**
   ```bash
   ./setup-hooks.sh
   ```

2. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: Add comprehensive CI/CD setup"
   git push origin main
   ```

3. **Watch CI run:**
   Visit https://github.com/phzwart/ssbc/actions

4. **Optional enhancements:**
   - Set up Codecov for coverage tracking
   - Configure PyPI publishing for releases
   - Add badges to README.md

---

**Your code is now professionally maintained with automated quality checks!** 🚀

