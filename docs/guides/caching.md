---
icon: lucide/database
---

# Caching

The client can cache `GET` responses on disk to avoid refetching unchanged data. Caching
is **off by default**. Any successful `POST`/`PATCH`/`DELETE` clears the cache, so writes
never leave you reading stale data.

## Enabling at construction

Pass `cache=True` for defaults, or a [`CacheConfig`][mreg_api.cache.CacheConfig] to
customize:

```python
from mreg_api import MregClient, CacheConfig

# Defaults
client = MregClient(url="https://mreg.example.com", cache=True)

# Custom configuration
client = MregClient(
    url="https://mreg.example.com",
    cache=CacheConfig(ttl=600, directory="/var/tmp/mreg-cache"),
)
```

## `CacheConfig` options

| Field | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `True` | Whether the cache is active. |
| `ttl` | `int` | `300` | Time-to-live for cached entries, in seconds. |
| `tag` | `str` | `"mreg-api"` | Tag used to namespace cache entries. |
| `timeout` | `int` | `60` | Open timeout for the underlying disk cache. |
| `directory` | `str \| None` | `None` | Cache directory. `None` uses a system temp directory. |

## Controlling the cache at runtime

```python
client.enable_cache()                       # turn caching on
client.enable_cache(CacheConfig(ttl=60))    # reconfigure and turn on
client.disable_cache()                      # turn off (clears by default)
client.disable_cache(clear=False)           # turn off, keep cached data
client.clear_cache()                        # drop all cached entries
```

Use the [`caching()`][mreg_api.client.MregClient.caching] context manager to toggle
caching for a block:

```python
with client.caching(enable=True):
    client.host.list()  # served from / stored in the cache
```

## Inspecting the cache

```python
info = client.get_cache_info()  # CacheInfo, or None if caching is disabled
if info:
    print(info.items, info.hits, info.misses, info.size)
```
