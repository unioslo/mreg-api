# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `mreg_api.requestlog` module that defines request/response log data structures. This includes the `RequestLog` class for storing and retrieving request records, as well as filtering capabilities.
- `MregClient(request_log_size=...)` parameter to size the request log.
- `EventKind.TRUNCATION` event kind. Recorded on `MregClient.events` when a list GET request matches more objects than the requested `limit`, so consumers can react to dropped results.
- `MregClient.*.refresh()` method to refresh an object with the latest data from the API. Used to update the state of an object after an `update()` call.

### Changed

- Client request/response history is now stored in a `RequestLog` instance, accessible via the `MregClient.requests` property.
- **BREAKING:** `limit` semantics for list GET requests (`MregClient.get_list()` and `MregClient.get_typed()`). Previously, matching more results than `limit` raised `TooManyResults`. Now the results are truncated to `limit` and an `EventKind.TRUNCATION` event is recorded instead of raising. Passing `None` (the default) leaves results unrestricted.
  - The emitted event contains an ObjectRef to the GET request URL, so consumers can correlate the event with the request that caused it (subject to change in the future).

### Deprecated

- `MregClient.get_client_history()` method. Use `MregClient.requests.get()` instead.
- `MregClient.clear_client_history()` method. Use `MregClient.requests.clear()` instead.
- `MregClient(history_size=...)` parameter. Use `request_log_size` instead; `history_size` remains as an alias.
- `exceptions.TooManyResults`. See the truncation behaviour above.

### Removed

- **BREAKING:** `MregClient.get_list_generic()` method. Its behaviour is now part of `MregClient.get_list()`.

## [0.4.0](https://github.com/unioslo/mreg-api/releases/tag/0.4.0) - 2026-08-24

### Added

- `MregClient.community` property to access the `CommunityManager` for managing communities.
  - Still exists via `MregClient.network.community`.
- `EventLog.get(subject=..., kind=..., level=..., min_level=...)` method to support retrieving events and filtering by a combination of subject, kind, and level. This replaces the existing `get_for`, `get_by_kind`, `get_by_level`, and `get_at_or_above` methods, and allows one to combine all their filters in a single call. Currently, only intersections of filters are supported (i.e. events must match all filters to be returned). Future releases may support unions of filters (i.e. events matching any filter will be returned).
- `MregClient.policy` property that groups the existing `MregClient.atom`, `MregClient.role`, `MregClient.label` into a common namespace as `MregClient.policy.role`, `MregClient.policy.atom`, and `MregClient.policy.label`.

### Changed

- `MregClient.*.create()` semantics and return type changes:
  - POST requests now prefer to return the created object found in the response, rather than fetching the object again from the API via the location header. The manager falls back on fetching via location header if the response does not contain the created object.
  - `create()` method return type changed from `T | None` to `T`. Never returns None.
  - `create(fetch_after_create)` parameter removed.
- `models.LDAPHealth` now defines `status` as a `Literal["OK", "Down", "Unknown"]` type instead of a generic string.
- `models.UserInfo` now has defaults for all its fields.
- `MregClient.community.create()` now returns the created `Community` object instead of a boolean indicating success.
- Default limit of 500 for `list()` calls has been changed to `None` (unrestricted).

### Removed

- **BREAKING:** `create(fetch_after_create)` parameter. No longer has any meaning as each endpoint returns a valid object representation in the response body after creation.

### Deprecated

- `EventLog.get_all()` -> `EventLog.get()`
- `EventLog.get_for()` -> `EventLog.get(subject=...)`
- `EventLog.get_by_kind()` -> `EventLog.get(kind=...)`
- `EventLog.get_by_level()` -> `EventLog.get(level=...)`
- `EventLog.get_at_or_above()` -> `EventLog.get(min_level=...)`

## [0.3.0](https://github.com/unioslo/mreg-api/releases/tag/0.3.0) - 2026-08-12

### Added

- `Host.get_community_associations(name: str)` method to retrieve all ipaddress->community associations for the host with the given community name.
- `Network.get_communities(name: str)` method to retrieve all communities with the given name.

### Changed

- **BREAKING**: `MregClient` is no longer a singleton. Each instance is independent and maintains its own state. The `MregClient.get_instance()` method has been removed. Each instantiation of `MregClient` is now independent and maintains its own state, including history, events and cache.
- `Network.get_community()` now ignores case when searching for communities by name. It returns the first community with the given name, or None if not found.

### Removed

- **BREAKING**: `MregClient.reset_instance` removed because the singleton pattern has been removed.

## [0.2.3](https://github.com/unioslo/mreg-api/releases/tag/0.2.3) - 2026-05-21

### Fixed

- Fixed `NAPTR` fields missing defaults after removal of nullability.

## [0.2.2](https://github.com/unioslo/mreg-api/releases/tag/0.2.2) - 2026-04-23

### Changed

- Removed nullability of `NAPTR` fields.

### Removed

- Duplicate endpoint `Endpoints.NAPTRs`. Now only `Endpoints.Naptrs` exists.

## [0.2.1](https://github.com/unioslo/mreg-api/releases/tag/0.2.1) - 2026-04-20

### Added

- `APIMixin.endpoint_with_id` property for getting the endpoint for the object with its ID, used for operations on the object itself (PATCH, DELETE, etc.). Used by default in `APIMixin.patch()` and `APIMixin.delete()`. Subclasses can override `endpoint_with_id` to provide custom endpoint resolution for these operations.

### Removed

- `Community.patch()`. Now uses `APIMixin.patch()`.
- `Community.delete()`. Now uses `APIMixin.delete()`.
- `APIMixin.patch()` response validation.

### Deprecated

- `APIMixin.patch()` parameter `validate`. Will be removed in 0.3.0.

## [0.2.0](https://github.com/unioslo/mreg-api/releases/tag/0.2.0) - 2026-04-16

### Added

- `MregClient(user_agent=)` parameter for setting a custom user agent.

### Changed

- **Breaking:** `MregClient.{get,post,patch,delete}()`:
  - All arguments apart from `path` are now keyword-only.
  - Replace parameter `**kwargs` with an explicit `json: Json | None = None` parameter.
- **Breaking:** `APIMixin.patch()`:
  - Renamed parameter `fields` to `data` (type changed from `Mapping[str, Any]` to `Json`)
  - Added parameter `params: QueryParams | None = None`
  - Made parameter `validate` keyword-only.
- **Breaking:** `APIMixin.create()`:
  - Renamed parameter `params` to `data` (type unchanged)
  - Made parameter `fetch_after_create` keyword-only.
- **Breaking:** `Community.patch()` (Mirrors `APIMixin.patch`):
  - Renamed parameter `fields` to `data` (type changed from `Mapping[str, Any]` to `Json`)
  - Added parameter `params: QueryParams | None = None`
  - Made parameter `validate` keyword-only.

## [0.1.5](https://github.com/unioslo/mreg-api/releases/tag/0.1.5) - 2026-03-26

### Added

- `ObjectRef.field` field to declare the name of the field the `value` pertains to. Defaults to `"id"`. This allows referencing objects by fields other than their primary ID (e.g. `field="ipaddress"`).

### Changed

- Docstrings converted from Sphinx-style to Google-style.
- `Community.remove_host()` `ipaddress` parameter now defaults to `None`.
- `ObjectRef.id` renamed to `ObjectRef.value`. Since `field` can now point to non-ID fields, `value` more accurately describes the attribute as the value of whichever field is referenced.

## [0.1.4](https://github.com/unioslo/mreg-api/releases/tag/0.1.4) - 2026-03-24

### Added

- `mreg_api.events.EventLevel` enum with values DEBUG, INFO, WARNING, ERROR, CRITICAL to represent the severity level of events.
- `mreg_api.events.EventLog.get_by_level(level: EventLevel)` method to retrieve events with exactly the given level.
- `mreg_api.events.EventLog.get_at_or_above(level: EventLevel)` method to retrieve events at or above the given severity level.

### Changed

- `mreg_api.events.Event` now has a `level` attribute of type `EventLevel` in addition to the existing `kind` attribute. This allows for more granular categorization of events by severity level in handlers.
- `mreg_api.events.EventKind` now only describes the type of the event, not its severity.
- `mreg_api.events.EventKind.INFO` renamed to `EventKind.NOTICE` to avoid ambiguity with `EventLevel.INFO`.

### Removed

- `mreg_api.events.EventKind.WARNING`. Moved to `EventLevel` enum.

## [0.1.3](https://github.com/unioslo/mreg-api/releases/tag/0.1.3) - 2026-03-23

### Added

- Event log functionality to MREG API client, including event recording and retrieval, with the ability for consumers to subscribe to events.
  - Accessed via `MregClient.events`, which is an instance of `mreg_api.events.EventLog`.
  - Event log max size can be controlled via the `event_log_size` parameter to `MregClient` (default: 100). Passing in `None` means unlimited size.

### Removed

- `FrozenModel` note system, replaced with the more comprehensive and flexible event log system.

## [0.1.2](https://github.com/unioslo/mreg-api/releases/tag/0.1.2) - 2026-03-18

### Added

- All models with `get` or `fetch` methods in `mreg_api.models.models` are now accessible as attributes of `MregClient` in lowercase form. For example, `MregClient().host` gives access to the `Host` model.
- `timeout` parameter to `MregClient` to configure request timeouts.
- `page_size` parameter to `MregClient` to configure default page size for list GET requests.
- `Host.add_mx` method to add an MX record to a host.
- `Host.remove_mx` method to remove an MX from a host.
- `MX.create_mx` convenience method to create an MX record.

## [0.1.0](https://github.com/unioslo/mreg-api/releases/tag/0.1.0) - 2026-01-27

### Added

- Initial release of MREG API client library.
