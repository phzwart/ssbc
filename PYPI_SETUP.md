# PyPI Publishing Setup Guide

This guide will help you set up automated PyPI publishing for the SSBC package.

## 🎯 Quick Start - Trusted Publishing (Recommended)

Trusted Publishing is the **modern, secure way** to publish to PyPI without API tokens.

### Step 1: Create PyPI Account (if needed)

1. Go to https://pypi.org/account/register/
2. Create an account with your email
3. Verify your email address

### Step 2: Register Your Package Name

**IMPORTANT:** You need to do this ONCE before setting up trusted publishing:

1. Build your package locally:
   ```bash
   just build
   # or
   uv build
   ```

2. Upload manually the FIRST time using twine:
   ```bash
   # Install twine in your environment
   mamba run -n ssbc pip install twine

   # Upload to PyPI (will prompt for username/password or token)
   mamba run -n ssbc twine upload dist/*
   ```

3. Or create an API token for first upload:
   - Go to https://pypi.org/manage/account/token/
   - Create a token for "Entire account (all projects)"
   - Use it when prompted by twine:
     - Username: `__token__`
     - Password: `<your-token>`

### Step 3: Set Up Trusted Publishing on PyPI

Once your package exists on PyPI:

1. Go to https://pypi.org/manage/project/ssbc/settings/publishing/

2. Add a new "pending publisher":
   - **PyPI Project Name**: `ssbc`
   - **Owner**: `phzwart`
   - **Repository name**: `ssbc`
   - **Workflow name**: `release.yml`
   - **Environment name**: `release`

3. Click "Add"

4. **Done!** Future releases will publish automatically.

### Step 4: Test Your Release Workflow

Create and push a tag:

```bash
# Option 1: Using justfile (reads version from pyproject.toml)
just tag

# Option 2: Manual tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

The workflow will automatically:
1. ✅ Run all tests
2. ✅ Build the package
3. ✅ Publish to PyPI
4. ✅ Create GitHub release with notes

## 📋 Release Checklist

Before creating a release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `HISTORY.md` or changelog
- [ ] Ensure all tests pass locally: `just qa`
- [ ] Ensure CI is green: https://github.com/phzwart/ssbc/actions
- [ ] Commit all changes
- [ ] Create and push tag: `just tag`

## 🔧 Alternative: API Token Method (Not Recommended)

If you can't use Trusted Publishing:

1. Create a PyPI API token:
   - Go to https://pypi.org/manage/account/token/
   - Create a token scoped to the `ssbc` project

2. Add to GitHub Secrets:
   - Go to https://github.com/phzwart/ssbc/settings/secrets/actions
   - Add secret: `PYPI_API_TOKEN`
   - Paste your token

3. Update `.github/workflows/release.yml`:
   ```yaml
   - name: Publish package to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
       skip-existing: true
   ```

**Note:** Trusted Publishing is more secure (no tokens to manage)!

## 🚀 Release Process

### Regular Release

```bash
# 1. Update version in pyproject.toml
# Example: version = "0.1.0" -> "0.2.0"

# 2. Commit the version bump
git add pyproject.toml
git commit -m "chore: Bump version to 0.2.0"
git push origin main

# 3. Create and push tag (using justfile)
just tag

# 4. GitHub Actions will automatically:
#    - Run tests
#    - Build package
#    - Publish to PyPI
#    - Create GitHub release
```

### Pre-release (Beta/Alpha)

```bash
# Use version like: "0.2.0b1" or "0.2.0a1"
# Tag with: v0.2.0b1

# To mark as pre-release in GitHub:
# Edit .github/workflows/release.yml:
prerelease: true  # for beta/alpha versions
```

## 📦 What Gets Published

Your package will be available at:
- **PyPI**: https://pypi.org/project/ssbc/
- **Install**: `pip install ssbc`
- **GitHub**: https://github.com/phzwart/ssbc/releases

## 🔍 Verify Package Before Publishing

Test your package locally:

```bash
# Build the package
just build

# Check the built files
ls -lh dist/

# Install locally and test
pip install dist/ssbc-0.1.0-py3-none-any.whl
python -c "import ssbc; print(ssbc.__version__)"

# Test in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install dist/ssbc-0.1.0-py3-none-any.whl
python -c "from ssbc import ssbc_correct; print('Success!')"
deactivate
rm -rf test_env
```

## ⚠️ Important Notes

1. **Version numbers are immutable** - Once published, you can't change a version
2. **Test on TestPyPI first** (optional): https://test.pypi.org/
3. **Semantic versioning**: MAJOR.MINOR.PATCH (e.g., 0.1.0 → 0.2.0 → 1.0.0)
4. **Tags must match version**: If `version = "0.1.0"`, tag should be `v0.1.0`

## 🆘 Troubleshooting

### Release workflow fails to publish

1. Check you've set up Trusted Publishing on PyPI
2. Verify the environment name is exactly `release`
3. Check workflow permissions are correct
4. View logs: https://github.com/phzwart/ssbc/actions

### Package name already taken

If someone else has `ssbc`:
1. Choose a different name in `pyproject.toml` (e.g., `ssbc-pac`)
2. Update the PyPI project URL in release.yml
3. Register the new name on PyPI

### First upload fails

The very first upload might need manual intervention:
1. Build locally: `uv build`
2. Upload with twine: `twine upload dist/*`
3. Then set up Trusted Publishing
4. Future releases will be automatic

## 📚 Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions PyPI Publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Python Packaging Guide](https://packaging.python.org/)

## ✅ Current Status

Your release workflow is:
- ✅ Configured for Trusted Publishing
- ✅ Environment: `release`
- ✅ Ready to use
- ⏳ Needs: PyPI Trusted Publisher setup (one-time)

Once you complete the PyPI setup, just run:
```bash
just tag
```

And your package will be on PyPI! 🚀
