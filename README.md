# MREG API [![Build Status](https://github.com/unioslo/mreg-api/actions/workflows/test.yml/badge.svg)](https://github.com/unioslo/mreg-api/actions/workflows/test.yml)

## Typing and IDE Support

The library is fully typed. For best IDE support and the fluent API, use client managers:

```python
from mreg_api import MregClient

client = MregClient(...)
host = client.host().get_by_id(123)
if host:
    host.patch(comment="hi")
    host.patch_typed(comment="hi")  # typed helper for Host
    host.patch_raw({"comment": "hi"})  # explicit untyped fallback
```

Objects fetched through a manager are automatically bound to the client, so instance methods do not accept an explicit `client` argument. For manually constructed objects (e.g., `Host.model_validate(...)` in tests or custom tooling), call `obj.bind(client)` once before using instance methods.

If you need typing for wrappers or tests, `mreg_api.types.ClientProtocol` documents the minimal client interface expected by models.

Typed patch payloads are available in `mreg_api.models.patch_types` (also re-exported from `mreg_api.models`), and patchable models expose a `patch_typed(...)` helper in addition to `patch(...)` and `patch_raw(...)`.

## List Interface (ModelList)

Manager list methods (for example `client.host().get_list()`) return a `ModelList`, which behaves like a normal Python list but adds bulk operations:

```python
hosts = client.host().get_list()

# Regular list behavior still works:
for host in hosts:
    print(host.name)

# Bulk patch (returns a new ModelList of updated objects):
updated_hosts = hosts.patch(comment="hi")
updated_hosts = hosts.patch_typed(comment="hi")  # typed helper on ModelList
updated_hosts = hosts.patch_raw({"comment": "hi"})  # explicit untyped fallback

# Directly.
updated_hosts = client.host().get_list().patch_typed(comment="hi")

# Bulk delete (raises on failure):
hosts.delete()
```

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
