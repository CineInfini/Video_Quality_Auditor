# Contributing to CineInfini

We welcome contributions! Please follow these guidelines:

## How to contribute

1. Fork the repository and create a new branch.
2. Make your changes with clear commit messages.
3. Write tests for new functionality.
4. Ensure the CI passes (linting, tests).
5. Submit a pull request.

## Code style

- Use [Black](https://black.readthedocs.io/) for formatting.
- Use [Ruff](https://ruff.rs/) for linting.
- Write docstrings for all public functions.

## Reporting issues

Use the GitHub issue tracker. Include relevant logs and video sample if possible.

## Development setup

```bash
pip install -e .[dev]
pre-commit install
