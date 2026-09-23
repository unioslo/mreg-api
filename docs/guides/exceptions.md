---
icon: lucide/circle-alert
---

# Handling errors

mreg-api raises attempts to raise its own exceptions for most error conditions, including HTTP errors, validation failures, and entity-lookup issues.
This page explains the exception hierarchy, how to catch and inspect exceptions, and how to react to specific error conditions.

## Exception hierarchy

All mreg-api exceptions inherit from a common [`MregApiBaseError`][mreg_api.exceptions.MregApiBaseError] base class.

The exception hierarchy is as follows:

{{ exception_tree() }}

To catch all exceptions raised by mreg-api, one can use [`MregApiBaseError`][mreg_api.exceptions.MregApiBaseError],
while [`ResponseError`][mreg_api.exceptions.ResponseError] narrows to any error backed by an HTTP response,
and [`HTTPStatusError`][mreg_api.exceptions.HTTPStatusError] narrows further to 4xx/5xx responses.

## Catching all exceptions

``` python
from mreg_api.exceptions import MregApiBaseError

try:
    client.host.get("host.example.com")
except MregApiBaseError as exc:
    # Anything the library raises lands here.
    ...
```

## Catching HTTP status errors

HTTP operations against the server raise a subclass of [`HTTPStatusError`][mreg_api.exceptions.HTTPStatusError] depending on the HTTP method used.

These are:

- [`GetError`][mreg_api.exceptions.GetError] for GET requests
- [`PostError`][mreg_api.exceptions.PostError] for POST requests
- [`PatchError`][mreg_api.exceptions.PatchError] for PATCH requests
- [`DeleteError`][mreg_api.exceptions.DeleteError] for DELETE requests

For example, to catch a failed GET request specifically, one can do:

``` python
from mreg_api import MregClient
from mreg_api.exceptions import HTTPStatusError, GetError

client = MregClient(url="https://mreg.example.com")

try:
    host = client.host.get("some-host-that-maybe-exists.example.com")
    client.host.update(host, name="newname")
    client.host.delete(host)
except GetError as exc:
    # Only failed GET requests.
    ...
except HTTPStatusError as exc:
    # Any other 4xx/5xx response.
    ...
```

## Reading the error message

Raised [`ResponseError`][mreg_api.exceptions.ResponseError] exceptions contain the HTTP response and parsed error details.

The `detail` attribute contains the main error message from the server without additional context,
while the `formatted_message()` method returns a more detailed message, optionally formatted as JSON.

``` python
try:
    client.host.create(...)
except ResponseError as exc:
    print(str(exc)) # or print(exc.formatted_message())
    # 400 Bad Request: POST https://mreg.example.com/api/v1/hosts/
    # 1 error:
    #   name: This field is required.  (required)

    print(exc.detail)
    # This field is required.

    print(exc.formatted_message(json=True))
    # 400 Bad Request: POST https://mreg.example.com/api/v1/hosts/
    # {
    #   "type": "validation_error",
    #   "errors": [ ... ]
    # }
```

!!! note

    `str(ResponseError(...))` is equivalent to `ResponseError(...).formatted_message()` with no arguments.

## Inspecting structured errors

Each [`ResponseError`][mreg_api.exceptions.ResponseError] instance's `.errors` contains the parsed error response from the server as a [`MREGErrorResponse`][mreg_api.exceptions.MREGErrorResponse] object.
If the response cannot be parsed, a default `MREGErrorResponse` with type `"unknown"` is returned.

Each `MREGError` object in the `.errors` list has a standardized `code`, the offending
field in `attr`, and a (generally) more verbose and human-readable `detail`.

``` python
try:
    client.host.create(...)
except ResponseError as exc:
    print(exc.errors.type)          # e.g. "validation_error"
    for err in exc.errors.errors:
        print(err.code, err.attr, err.detail)
        # required name This field is required.
```

## Using the response object

[`ResponseError`][mreg_api.exceptions.ResponseError] exceptions expose the underlying HTTP response object as `.response`, which can be used to inspect headers, status codes, and other details.

``` python
try:
    client.host.update(host, name="newname")
except ResponseError as exc:
    if exc.response.status_code == 404:
        # some contrived example here :)
        ...
```

## Client-side validation errors

When internal validation fails (e.g. a model is missing a required field), the library raises [`MregValidationError`][mreg_api.exceptions.MregValidationError].
The exception wraps the underlying Pydantic validation error, which is available on `.pydantic_error`.

``` python
from mreg_api.exceptions import MregValidationError

try:
    client.host.list()
except MregValidationError as exc:
    print(exc)                 # includes the field, reason, and offending input
    print(exc.pydantic_error)  # the underlying pydantic.ValidationError, or None
```

## Other exceptions

These exception are less common, and typically indicate some user error or failsafe that is triggered, that must be handled carefully.

### `ForceMissing` for destructive or unusual operations

[`PreconditionError`][mreg_api.exceptions.PreconditionError] (and more often, its subclass [`ForceMissing`][mreg_api.exceptions.ForceMissing]) is raised when an operation's precondition is not met. MREG allows destructive, potentially cascading, operations on resources with many-to-many relations, as well as other modifications that should be explicitly acknowledged by the user before proceeding.

``` python
from mreg_api.exceptions import PreconditionError # or ForceMissing

try:
    client.zone.delete("somezone.example.com") # force missing
except PreconditionError as exc:
    # Zone still has registered entries or subzones
    # use some heuristic to determine if we should force deletion:
    if len(client.zone.list_subzones("somezone.example.com")) <= 2:
        client.zone.delete("somezone.example.com", force=True)
```

Currently, as of 0.6.0, these methods can raise `ForceMissing` if the precondition for the operation is not met:

- [`MregClient.atom.delete`][mreg_api.managers.AtomManager.delete]
- [`MregClient.role.delete`][mreg_api.managers.RoleManager.delete]
- [`MregClient.ipaddress.associate_mac`][mreg_api.managers.IPAddressManager.associate_mac]
- [`MregClient.delegation.create`][mreg_api.managers.DelegationManager.create]
- [`MregClient.zone.verify_nameservers`][mreg_api.managers.ZoneManager.verify_nameservers]
- [`MregClient.zone.create`][mreg_api.managers.ZoneManager.create]
- [`MregClient.zone.set_nameservers`][mreg_api.managers.ZoneManager.set_nameservers]
- [`MregClient.zone.delete`][mreg_api.managers.ZoneManager.delete]

## Reference

A complete API reference for all exceptions is available in the [Exceptions reference](../reference/exceptions.md) section.
