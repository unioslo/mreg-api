# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- All models with `get` or `fetch` methods in `mreg_api.models.models` are now accessible as attributes of `MregClient` in lowercase form. For example, `MregClient().host` gives access to the `Host` model.
- `timeout` parameter to `MregClient` to configure request timeouts.
- `page_size` parameter to `MregClient` to configure default page size for list GET requests.
- `APIMixin.bind(client)` to explicitly bind manually constructed model instances to a client.

### Changed

- Direct classmethod calls on models that require a client (e.g. `Host.get_by_id(client, ...)`) are no longer supported; use a client manager instead (e.g. `client.host().get_by_id(...)`).
- Instance methods on models no longer accept an explicit `client` argument; models must be bound to a client via a manager or `bind()`.

## [0.1.0](https://github.com/unioslo/mreg-api/releases/tag/0.1.0) - 2026-01-27

### Added

- Initial release of MREG API client library.
