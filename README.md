# MREG API [![Build Status](https://github.com/unioslo/mreg-api/actions/workflows/test.yml/badge.svg)](https://github.com/unioslo/mreg-api/actions/workflows/test.yml)

`mreg-api` is a Python client library for the [MREG](https://github.com/unioslo/mreg)
REST API. It gives you a typed, ergonomic interface to every MREG
resource through a single client
object with scoped namespaces for each resource type.

It provides optional caching of resources, automatic FQDN expansion of hostnames, and a consistent interface for all resource types. The client is compatible with Python 3.11 and later.


## Documentation

https://unioslo.github.io/mreg-api/

## Development

Set up a development environment with `uv`:

```bash
git clone git@github.com:unioslo/mreg-api.git
uv sync
```

### Pre-commit Hooks

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
