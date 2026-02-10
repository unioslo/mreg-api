# MREG API [![Build Status](https://github.com/unioslo/mreg-api/actions/workflows/test.yml/badge.svg)](https://github.com/unioslo/mreg-api/actions/workflows/test.yml)

## Typing and IDE Support

The library is fully typed. For best IDE support and the fluent API, use client managers:

```python
from mreg_api import MregClient

client = MregClient(...)
host = client.host().get_by_id(123)
host.patch({"comment": "hi"})
```

Objects fetched through a manager are automatically bound to the client, so instance methods do not accept an explicit `client` argument. For manually constructed objects (e.g., `Host.model_validate(...)` in tests or custom tooling), call `obj.bind(client)` once before using instance methods.

If you need typing for wrappers or tests, `mreg_api.types.ClientProtocol` documents the minimal client interface expected by models.

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
