# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MREG API is a Python client library for interacting with the MREG network registry system. It provides Pydantic-based models for type-safe API interactions, caching, and configuration management.

**Current Status:** This project is being extracted from the mreg-cli codebase to create a standalone API client library. Many files have been copied with minor name changes (e.g., `cli` -> `api`), but the architecture will be significantly refactored to support API-only usage without CLI dependencies.

**Key Technologies:**

- Python 3.11+ (supports 3.11, 3.12, 3.13)
- Pydantic v2 for data validation and models
- requests for HTTP operations
- diskcache for persistent caching
- pytest for testing

## Development Commands

### Environment Setup

```bash
# Install dependencies (uses uv for fast dependency resolution)
uv pip install -e ".[dev]"
```

### Code Quality

```bash
# Lint with ruff (check only)
ruff check .

# Lint and auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Combined lint + format (recommended before commits)
ruff check --fix . && ruff format .
```

### Testing

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest path/to/test_file.py

# Run tests matching pattern
pytest -k "pattern"
```

### Building

```bash
# Build distribution packages
python -m build
```

## Code Style

### Ruff Configuration

- Line length: 99 characters
- Target: Python 3.11+
- Import style: One import per line (`force-single-line = true`)
- Required import: `from __future__ import annotations` in all files

### Enabled Lints

- flake8-builtins (A)
- flake8-unused-arguments (ARG)
- flake8-bugbear (B)
- pydocstyle (D) - requires docstrings
- Error checking (E, F)
- isort (I) for import sorting
- Pylint conventions and errors (PLC, PLE)

### Docstring Style

- Multi-line docstrings: Summary on first line (D212)
- No blank line before class docstring (D211)
- Use triple double quotes (D300)

## Important Notes

### Work in Progress

This codebase is in active development as it's being extracted from mreg-cli. The current code structure is temporary and will undergo significant architectural changes. Do not assume the current patterns and module organization are final.

### Missing/Incomplete Modules

Several modules are referenced but not yet implemented or need to be refactored:

- `mreg_api/dirs.py` - Directory/path constants
- `mreg_api/outputmanager.py` - Output formatting (may be removed or refactored)
- `mreg_api/choices.py` - Choice enums
- `mreg_api/__about__.py` - Package metadata

### Testing

- Tests use pytest with `pytest-httpserver` for mocking HTTP
- Use `inline-snapshot` for snapshot testing
- Test files should be in a `tests/` directory (to be created)
- Test files excluded from strict docstring requirements

### Dependencies

Core dependencies are minimal:

- requests (with SOCKS proxy support)
- pydantic v2.7.1+
- pydantic-extra-types
- platformdirs
- diskcache

Dev dependencies include ruff, pytest, and build tools.

### Refactoring Roadmap

#### Goal

Transform this CLI codebase into a clean, standalone API client library for programmatic use.

#### Key Changes

- Remove CLI-specific logic and output methods from models
- Preserve current CRUD semantics (`fetch*`, `create`, `update`, `delete` methods)
- Create `mreg_api/client.py` with `MregClient` class for session management
  - Handles authentication, base URL, session state
  - Replace global session state with client instance (singleton pattern recommended)
  - Migrate `get`/`post`/`patch`/`delete` from `utilities/api.py` to client methods
- Rename `MregCliConfig` to remove CLI references
- **Token file handling:** Do NOT change semantics - CLI users depend on this for auth persistence

#### Fragile Modules

- `mreg_api/api/history.py` - Complex filtering logic, brittle code
