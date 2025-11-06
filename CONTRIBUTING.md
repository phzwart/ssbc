# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/phzwart/ssbc/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

ssbc could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/phzwart/ssbc/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `ssbc` for local development.

1. Fork the `ssbc` repo on GitHub.
2. Clone your fork locally:

   ```sh
   git clone git@github.com:your_name_here/ssbc.git
   ```

3. Install your local copy. The project uses `uv` for package management. Install in development mode:

   ```sh
   cd ssbc/
   uv pip install -e .
   ```

   This installs the package in editable mode with all dependencies.

4. Create a branch for local development:

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and tests:

   ```sh
   # Run all quality checks (formatting, linting, type checking, tests)
   just qa

   # Or run tests only
   just test

   # Run tests for all supported Python versions (3.10, 3.11, 3.12, 3.13)
   just testall
   ```

   The project uses `just` for task running. Install it with `cargo install just` or see [just documentation](https://github.com/casey/just).

6. Commit your changes and push your branch to GitHub:

   ```sh
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The pull request should work for Python 3.10, 3.11, 3.12, and 3.13. Tests run in GitHub Actions on every pull request to the main branch, make sure that the tests pass for all supported Python versions.

## Tips

To run a subset of tests:

```sh
pytest tests.test_ssbc
```

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are committed (including an entry in HISTORY.md). Then:

1. Update the version in `pyproject.toml` (e.g., from `1.3.4` to `1.3.5`)
2. Commit the version change
3. Create and push a git tag:

```sh
git tag -a v1.3.5 -m "Release version 1.3.5"
git push origin v1.3.5
```

You can set up a [GitHub Actions workflow](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python#publishing-to-pypi) to automatically deploy your package to PyPI when you push a new tag.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
