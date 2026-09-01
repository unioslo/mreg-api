---
icon: lucide/plug
---

# Client

The [`MregClient`][mreg_api.client.MregClient] is the entry point to the library. It
owns the HTTP session, authentication state, cache, request log, and event log, and
exposes every resource type as a manager attribute (see [Managers](managers.md)).

For a task-oriented introduction, see [Client configuration](../guides/configuration.md)
and [Authentication](../guides/authentication.md).

::: mreg_api.client.MregClient
