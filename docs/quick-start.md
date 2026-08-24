---
icon: lucide/rocket
---

# Quick start

## Connect

```python
import os
from mreg_api import MregClient

client = MregClient(url="https://mreg.example.com", domain="example.com")
client.login(
    username=os.environ.get("MREG_USERNAME", "mreguser"),
    password=os.environ.get("MREG_PASSWORD", "mregpass123"),
)
```

The `domain` argument enables automatic FQDN expansion of bare hostnames. See
[Client configuration](guides/configuration.md#domain-handling) and [Authentication](guides/authentication.md)
for more.

## Resources

Every MREG resource type is an attribute on the client, exposing the same resource manager interface:

```python
client.host     # manage hosts
client.network  # manage networks
client.cname    # manage CNAME records
# ...
```

See [Working with resources](guides/resources.md) for the complete list of available resources and their methods, and [Managers reference](reference/managers.md) for the per-resoource manager API reference.

## The resource manager interface

| Method | Purpose |
|---|---|
| `get(x)` | Fetch by name, ID, or model object |
| `get_by_id(id)` | Fetch by ID |
| `get_by_name(name)` | Fetch by name |
| `list(**filters)` | List, optionally filtered |
| `create(**fields)` | Create a resource |
| `update(target, **fields)` | Update a resource |
| `delete(target)` | Delete a resource |

This interface is consistent across most resource types, and the methods are documented in the [API reference](reference/client.md).

## Fetching

`get` interprets the argument type to decide the lookup: a string (name), an int (ID), or
an existing model object (to refresh it).

```python
host = client.host.get("myhost.example.com")
host = client.host.get("myhost")  # domain appended automatically if set on client
host = client.host.get(123)       # by ID
host = client.host.get(host)      # refresh model object
```

Pass `required=False` to get `None` instead of an error when the resource is missing:

```python
host = client.host.get("web", required=False)
if host is not None:
    client.host.update(host, name="web2")  # update the host if it exists
```

Explicit variants are also available:

```python
host = client.host.get_by_id(123)
host = client.host.get_by_name("myhost")
```

## Listing

`list` returns all resources of that type. Keyword arguments map to model fields for
filtering.

```python
all_hosts = client.host.list()
hosts = client.host.list(name__startswith="test-")
```

## Creating

`create` takes keyword arguments matching the model's fields. Endpoints aren't guaranteed
to return the created resource, so the return type is `T | None`. Pass
`fetch_after_create=True` to attempt to fetch the newly created resource.

```python
new_host = client.host.create(
    name="newhost.example.com",
    comment="Created via the API",
    fetch_after_create=True,
)
```

## Updating

`update` accepts a model object or an identifier (name, ID, …), plus the fields to change.

```python
client.host.update(
    new_host or "newhost.example.com",
    comment="Updated comment",
    ttl=3600,
)
```

## Deleting

`delete` accepts a model object or an identifier.

```python
client.host.delete(new_host)               # by object
client.host.delete("newhost.example.com")  # by name
client.host.delete(123)                     # by ID
```
