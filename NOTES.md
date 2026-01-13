<!-- NOTE: This NOTES/TODO file is for humans, not language models and AI assistants. It is not a step-by-step plan. Do not reference it when generating code. -->

# Mreg CLI

## Category and Location Validation

Add category and location validation to CLI commands in mreg-cli. They have been removed from model methods such as `Network.set_category()` and `Network.set_location()` in this package.

## List GET Request Limits

Adjust limit for list GET requests when using MregApiClient to be 500 (if desired). New default is None (no limit). This affects the following methods:

- `MregApiClient.get_list()`
- `MregApiClient.get_list_generic()`
- `MregApiClient.get_typed()`

## OutputManager Removal in models

OutputManager.add_line() calls have been removed from mreg_api.models.models and replaced with an internal note system using `FrozenModel.add_note()`. This allows tracking informational messages without direct CLI output.

For now, this is implemented as a `list[str]` private attribute `_notes` in `FrozenModel`, with an `add_note()` method to append notes. However, if richer functionality is needed later, a `Note` class could be created.

Commands can retrieve notes from model instances using a new `get_notes()` method.

## Token

Token needs to be retrieved from the client (`MregApiClient.get_token()`) and saved by the CLI. Token file management is not a part of the API client.

## HostName validation

HostName validation has been decoupled from MregCliConfig. Currently, a new class variable `HostName.domain` is used to provide a default domain when parsing hostnames. This can be set by the CLI after reading configuration.

It is not a very elegant solution, so we should consider other approaches.

# Mreg API

## Docstrings

Ruff's pydocstyle ruleset does not support sphinx style docstrings. Docstrings should be converted to google style.

https://docs.astral.sh/ruff/settings/#lint_pydocstyle_convention

## Force

References to "force" in error messages should be removed. The CLI should be the only place where "force" is relevant.

## Recording

We need to store requests in some internal client data structure that Mreg-CLI can read to generate its recording file.

IMPORTANT: mreg-cli needs to remember to clear this recording data structure between commands and store it in its OutputManager.
