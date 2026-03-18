# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- ## Unreleased -->

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
