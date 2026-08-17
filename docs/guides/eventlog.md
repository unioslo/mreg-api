---
icon: lucide/chart-no-axes-gantt
---

# Event Log

The client records an event log of resolutions, mutations, and notices that happen in-between requests for complex queries and multi-step operations. Typically, such events are things like CNAME and PTR resolution that happen when resolving Host or host records.

!!! note

    The event log system is a new feature and is still under development. The API may change in future releases. As of 0.3.0, it only records CNAME and PTR resolution events, but future releases will include more event types.

## Accessing the event log

The event log is accessible via the `client.events` property, which returns an [`EventLog`][mreg_api.events.EventLog] instance. The size of the log is controlled by the `event_log_size` argument to [`MregClient`][mreg_api.client.MregClient].

```python
client = MregClient(...)
events = client.events.get()
```

## Filtering by event kind

Each event is categorized by kind ([`EventKind`][mreg_api.events.EventKind]). You can filter the event log by kind using the `get` method with the `kind` argument:

```python
from mreg_api.events import EventKind

client.events.get(kind=EventKind.RESOLUTION)
client.events.get(kind=EventKind.MUTATION)
client.events.get(kind=EventKind.NOTICE)
```

## Filtering by event level

Each event is also categorized by level ([`EventLevel`][mreg_api.events.EventLevel]). You can filter the event log by level using the `get` method with the `level` argument:

```python
from mreg_api.events import EventLevel

client.events.get(level=EventLevel.DEBUG)
client.events.get(level=EventLevel.INFO)
client.events.get(level=EventLevel.WARNING)
client.events.get(level=EventLevel.ERROR)
client.events.get(level=EventLevel.CRITICAL)
```

The `min_level` argument can be used to retrieve events at or above a certain severity level instead of exactly matching a level:

```python
client.events.get(min_level=EventLevel.WARNING)
```
