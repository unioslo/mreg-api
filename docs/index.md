---
icon: lucide/house
---

# mreg-api

`mreg-api` is a Python client library for the [MREG](https://github.com/unioslo/mreg)
REST API. It gives you a typed, ergonomic interface to every MREG
resource — hosts, networks, DNS records, zones, and more — through a single client
object.

## Install

pip:

```bash
pip install mreg-api
```

uv:

```bash
uv add mreg-api
```

## Connect

```python
import os
from mreg_api import MregClient

client = MregClient(url="https://mreg.example.com", domain="example.com")
client.login(
    username=os.environ["MREG_USERNAME"],
    password=os.environ["MREG_PASSWORD"],
)

host = client.host.get("myhost")
```

## Where to go next

- **[Quick start](quick-start.md)** — the essentials in a few minutes.
- **Guides** — [Client configuration](guides/configuration.md),
  [Authentication](guides/authentication.md), [Caching](guides/caching.md), and
  [Working with resources](guides/resources.md).
- **[API reference](reference/client.md)** — full reference for the client, managers,
  models, events, cache, and exceptions.
