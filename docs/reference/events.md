---
icon: lucide/activity
---

# Events

!!! info

    For event log usage, see [Event Log](../guides/eventlog.md).

The client records an event log of resolutions, mutations, and notices that happen in-between requests for complex queries and other operations. Typically, such events are things like CNAME and PTR resolution that happen when resolving Host or host records.

The client exposes its event log via `client.events` (an [`EventLog`][mreg_api.events.EventLog]).
The size of the log is controlled by the `event_log_size` argument to [`MregClient`][mreg_api.client.MregClient].


::: mreg_api.events.EventLog

::: mreg_api.events.Event

::: mreg_api.events.EventKind

::: mreg_api.events.EventLevel

::: mreg_api.events.ObjectRef
