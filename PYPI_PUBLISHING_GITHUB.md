# PyPI Publishing via GitHub Actions (No Twine Needed!)

This guide shows how to publish to PyPI using ONLY GitHub Actions - no manual uploads needed!

## 🎯 Quick Start - Publish via GitHub

### Step 1: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create an account and verify your email

### Step 2: Set Up "Pending Publisher" (BEFORE First Release)

PyPI allows you to set up Trusted Publishing **before** your package exists!

1. Go to: https://pypi.org/manage/account/publishing/

2. Click **"Add a new pending publisher"**

3. Fill in these EXACT values:
   ```
   PyPI Project Name:    ssbc
   Owner:                phzwart
   Repository name:      ssbc
   Workflow name:        release.yml
   Environment name:     release
   ```

4. Click **"Add"**

5. **Done!** PyPI is now ready to accept your first automated release.

### Step 3: Create Your First Release (via GitHub)

```bash
# No code changes needed - just tag and push!
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# Or use the justfile shortcut:
just tag
```

That's it! GitHub Actions will:
1. ✅ Run all tests
2. ✅ Build the package
3. ✅ Publish to PyPI (automatically via Trusted Publishing)
4. ✅ Create GitHub release with notes

### Step 4: Verify Publication

Within 2-3 minutes, your package will be live:
- **PyPI**: https://pypi.org/project/ssbc/
- **Install**: `pip install ssbc`

## 🔄 Future Releases

For every future release:

```bash
# 1. Update version in pyproject.toml
# Example: "0.1.0" -> "0.2.0"

# 2. Commit the version bump
git add pyproject.toml
git commit -m "chore: Bump version to 0.2.0"
git push origin main

# 3. Create and push tag
just tag

# GitHub Actions handles the rest automatically!
```

## 📋 Pre-Release Checklist

Before pushing a tag:

- [ ] ✅ All CI checks are green: https://github.com/phzwart/ssbc/actions
- [ ] Version updated in `pyproject.toml`
- [ ] `HISTORY.md` updated (optional)
- [ ] All changes committed and pushed
- [ ] Local tests pass: `just qa`

## 🔍 What Happens When You Push a Tag

The release workflow (`.github/workflows/release.yml`) automatically:

1. **Checks out code** from the tagged commit
2. **Installs dependencies** with uv
3. **Runs all tests** to ensure quality
4. **Builds package** (creates .tar.gz and .whl files)
5. **Publishes to PyPI** using Trusted Publishing (no tokens!)
6. **Creates GitHub release** with auto-generated notes

All within 3-5 minutes!

## 🎯 Workflow File

Your release workflow is already configured:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'  # Triggers on tags like v0.1.0, v1.2.3, etc.

jobs:
  build-and-publish:
    environment:
      name: release  # This must match PyPI Trusted Publisher config
    permissions:
      id-token: write  # Required for Trusted Publishing
      contents: write  # Required for GitHub releases

    steps:
      - Run tests
      - Build package
      - Publish to PyPI (automatic via Trusted Publishing)
      - Create GitHub release
```

## 🔒 Security Benefits of Trusted Publishing

✅ **No API tokens** - No secrets to manage or leak
✅ **No credentials** - PyPI verifies GitHub's cryptographic signature
✅ **More secure** - Uses OpenID Connect (OIDC) authentication
✅ **Recommended** - Official recommendation from PyPI and GitHub

## ⚠️ Important Notes

### Version and Tag Must Match

If `pyproject.toml` has `version = "0.1.0"`, the tag should be `v0.1.0`.

The `justfile` handles this automatically:
```bash
just tag  # Reads version from pyproject.toml and creates matching tag
```

### Version Numbers are Immutable

Once published to PyPI, you **cannot** change or delete a version. Always test locally first!

### Test Before Publishing

```bash
# Build locally
mamba run -n ssbc uv build

# Check built files
ls -lh dist/

# Test installation locally
pip install dist/ssbc-0.1.0-py3-none-any.whl
python -c "import ssbc; print(ssbc.__version__)"
```

## 🚀 Test Release (Optional)

You can test the entire workflow with TestPyPI first:

1. Create account at https://test.pypi.org/
2. Set up pending publisher there
3. Temporarily modify `.github/workflows/release.yml`:
   ```yaml
   - name: Publish to TestPyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```
4. Push a test tag: `git tag v0.1.0-test && git push origin v0.1.0-test`
5. Verify it works, then publish to real PyPI

## 📊 Monitoring Your Release

Watch the workflow run:
- **Actions**: https://github.com/phzwart/ssbc/actions
- **Releases**: https://github.com/phzwart/ssbc/releases
- **PyPI**: https://pypi.org/project/ssbc/

## 🆘 Troubleshooting

### "Environment protection rules not satisfied"

If you see this error, you need to create the `release` environment in GitHub:
1. Go to: https://github.com/phzwart/ssbc/settings/environments
2. Click "New environment"
3. Name it exactly: `release`
4. Click "Configure environment"
5. (Optional) Add protection rules like requiring reviews

### "Trusted publishing exchange failure"

This means PyPI doesn't have your publisher configured:
1. Verify you set up the pending publisher on PyPI
2. Check all values match EXACTLY (owner, repo, workflow, environment)
3. Wait a few minutes and try again (can take time to propagate)

### "Package name already taken"

If someone else registered `ssbc`:
1. Choose a different name (e.g., `ssbc-pac`, `small-sample-beta`)
2. Update `name` in `pyproject.toml`
3. Update PyPI project URL in `release.yml`
4. Set up pending publisher with new name

## ✅ Current Status

Your repository is configured for:
- ✅ **Automatic PyPI publishing** via GitHub Actions
- ✅ **Trusted Publishing** (secure, no tokens)
- ✅ **Single version source** (pyproject.toml only)
- ✅ **Automated releases** (tag and push)

## 🚀 Ready to Publish?

1. Set up pending publisher on PyPI (5 minutes)
2. Run: `just tag`
3. Watch it publish automatically!

Your package will be on PyPI and installable via `pip install ssbc` within minutes! 🎉
