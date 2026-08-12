# MREG API [![Build Status](https://github.com/unioslo/mreg-api/actions/workflows/test.yml/badge.svg)](https://github.com/unioslo/mreg-api/actions/workflows/test.yml)


## Quick Start

### Connect

```py
import os
from mreg_api.client import MregClient

client = MregClient(url="https://mreg.example.com", domain="example.com")
client.login(
    username=os.environ.get("MREG_USERNAME", "mreguser"),
    password=os.environ.get("MREG_PASSWORD", "mregpass123"),
)
```

### Resources

Every MREG resource type is an attribute on the client, exposing the same manager interface:

```py
client.host     # manage hosts
client.network  # manage networks
client.cname    # manage CNAME records
# ...
```

### The manager interface

| Method | Purpose |
|---|---|
| `get(x)` | fetch by name, ID, or model object |
| `get_by_id(id)` | fetch by ID |
| `get_by_name(name)` | fetch by name |
| `list(**filters)` | list, optionally filtered |
| `create(**fields)` | create a resource |
| `update(target, **fields)` | update a resource |
| `delete(target)` | delete a resource |

### Fetching

`get` interprets the argument type to decide the lookup: a string (name), an int
(ID), or an existing model object (to refresh it).

```py
host = client.host.get("myhost.example.com")
host = client.host.get("myhost")  # domain appended automatically if set on client
host = client.host.get(123)       # by ID
host = client.host.get(host)      # refresh model object
```

Explicit variants are also available:

```py
host = client.host.get_by_id(123)
host = client.host.get_by_name("myhost")
```

### Listing

`list` returns all resources of that type. Keyword arguments map to model fields
for filtering.

```py
all_hosts = client.host.list()
hosts = client.host.list(name__startswith="test-")
```

### Creating

`create` takes keyword arguments matching the model's fields. Endpoints aren't
guaranteed to return the created resource, so the return type is `T | None`. Pass
`fetch_after_create=True` to attempt to fetch the newly created resource.

```py
new_host = client.host.create(
    name="newhost.example.com",
    comment="Created via the API",
    fetch_after_create=True,
)
```

### Updating

`update` accepts a model object or an identifier (name, ID, …), plus the fields
to change.

```py
client.host.update(
    new_host or "newhost.example.com",
    comment="Updated comment",
    ttl=3600,
)
```

### Deleting

`delete` accepts a model object or an identifier.

```py
client.host.delete(new_host)               # by object
client.host.delete("newhost.example.com")  # by name
client.host.delete(123)                     # by ID
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
