# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- ## Unreleased -->

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
