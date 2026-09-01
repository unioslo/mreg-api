---
icon: lucide/settings
---

# Client configuration

Everything starts with an [`MregClient`][mreg_api.client.MregClient]. Constructing one
sets up the HTTP session but does **not** authenticate or make any network requests — see
[Authentication](authentication.md) for that.

```python
from mreg_api import MregClient

client = MregClient(
    url="https://mreg.example.com",
    domain="example.com",
)
```

## Constructor options

| Argument | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | *required* | Base URL of the MREG server. All request paths are joined onto it. |
| `domain` | `str | None` | `None` | Default DNS domain used to expand bare hostnames into FQDNs. See [Domain handling](#domain-handling). |
| `user` | `str | None` | `None` | Informational username stored on the client. Does **not** authenticate. |
| `timeout` | `int | float | None` | `60` | Request timeout in seconds, passed to the underlying HTTP client. |
| `cache` | `CacheConfig | bool` | `False` | Enable GET-response caching. `True` uses defaults; pass a [`CacheConfig`][mreg_api.cache.CacheConfig] to customize. See [Caching](caching.md). |
| `follow_redirects` | `bool` | `False` | Whether the HTTP client follows redirects. |
| `page_size` | `int | None` | `None` | Default pagination page size sent as a query parameter on GET requests. |
| `request_log_size` | `int | None` | `100` | Maximum number of request/response records kept in `client.requests`. (`history_size` is a deprecated alias.) |
| `event_log_size` | `int | None` | `100` | Maximum number of entries kept in the [event log](../reference/events.md). |
| `user_agent` | `str | None` | `None` | Custom `User-Agent` header. Defaults to `mreg-api-<version>`. |

## Domain handling

When `domain` is set, bare hostnames are expanded to FQDNs automatically by resource
methods. You can also expand a name explicitly or override the domain temporarily:

```python
client.fqdn("web")  # -> "web.example.com"

with client.domain_override("other.example.com"):
    client.fqdn("web")  # -> "web.other.example.com"
```

## Timeout

`timeout` is also available as a settable property after construction:

```python
client.timeout = 120
```

## Request log

Every request/response pair is recorded in a bounded deque, sized by `request_log_size`:

```python
for record in client.requests:
    print(record.method, record.status, record.url)

client.requests.clear()
```
