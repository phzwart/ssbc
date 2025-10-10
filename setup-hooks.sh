#!/bin/bash
# Setup script for CI/CD hooks

set -e

echo "🚀 Setting up CI/CD hooks for SSBC..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dev dependencies
echo "📦 Installing development dependencies..."
uv sync --extra dev --extra test

# Install pre-commit
echo "🔧 Setting up pre-commit hooks..."
uv run --extra dev pre-commit install

# Run pre-commit on all files to check everything
echo "✅ Running pre-commit hooks on all files (this may take a moment)..."
if uv run --extra dev pre-commit run --all-files; then
    echo "✨ All pre-commit hooks passed!"
else
    echo "⚠️  Some hooks failed. Don't worry, they've auto-fixed what they could."
    echo "   Please review the changes and commit them."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review any auto-fixed changes: git diff"
echo "2. Read the CI/CD guide: cat CI_CD_GUIDE.md"
echo "3. Commit your changes: git add . && git commit -m 'Add CI/CD setup'"
echo "4. Push to GitHub: git push"
echo ""
echo "From now on, pre-commit hooks will run automatically before each commit."
echo "To run QA checks manually: just qa"

