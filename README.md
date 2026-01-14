# MREG API [![Build Status](https://github.com/unioslo/mreg-api/actions/workflows/test.yml/badge.svg)](https://github.com/unioslo/mreg-api/actions/workflows/test.yml)

## Pre-commit Hooks

This project uses `prek` to manage pre-commit hooks for code quality and formatting. To set up the pre-commit hooks, run the following command:

```bash
uv tool install prek
```

Then, install the pre-commit hooks with:

```bash
prek install
```

Run the pre-commit checks manually with:

```bash
prek run
# Optionally for all files:
prek run --all-files
```
