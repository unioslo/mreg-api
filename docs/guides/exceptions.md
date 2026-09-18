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
while [`APIError`][mreg_api.exceptions.APIError] narrows to server-side failures.

``` python
from mreg_api.exceptions import MregApiBaseError

try:
    client.host.get("host.example.com")
except MregApiBaseError as exc:
    # Anything the library raises lands here.
    ...
```

## Catching API errors

API operations raise a subclass of [`APIError`][mreg_api.exceptions.APIError] depending on the HTTP method used.

These are:

- [`GetError`][mreg_api.exceptions.GetError] for GET requests
- [`PostError`][mreg_api.exceptions.PostError] for POST requests
- [`PatchError`][mreg_api.exceptions.PatchError] for PATCH requests
- [`DeleteError`][mreg_api.exceptions.DeleteError] for DELETE requests

For example, to catch a failed GET request specifically, one can do:

``` python
from mreg_api import MregClient
from mreg_api.exceptions import APIError, GetError

client = MregClient(url="https://mreg.example.com")

try:
    host = client.host.get("some-host-that-maybe-exists.example.com")
    client.host.update(host, name="newname")
except GetError as exc:
    # Only failed GET requests.
    ...
except APIError as exc:
    # Any other request failure.
    ...
```

## Reading the error message

Raised [`APIError`][mreg_api.exceptions.APIError] exceptions contain the HTTP response and parsed error details.

The `detail` attribute contains the main error message from the server without additional context,
while the `formatted_message()` method returns a more detailed message, optionally formatted as JSON.

``` python
try:
    client.host.create(...)
except APIError as exc:
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

    `str(APIError(...))` is equivalent to `APIError(...).formatted_message()` with no arguments.

## Inspecting structured errors

Each[`APIError`][mreg_api.exceptions.APIError]instance's `.errors` contains the parsed error response from the server as a [`MREGErrorResponse`][mreg_api.exceptions.MREGErrorResponse] object.
If the response cannot be parsed, a default `MREGErrorResponse` with type `"unknown"` is returned.

Each `MREGError` object in the `.errors` list has a standardized `code`, the offending
field in `attr`, and a (generally) more verbose and human-readable `detail`.

``` python
try:
    client.host.create(...)
except APIError as exc:
    print(exc.errors.type)          # e.g. "validation_error"
    for err in exc.errors.errors:
        print(err.code, err.attr, err.detail)
        # required name This field is required.
```

## Using the response object

`APIError` exceptions also expose the underlying HTTP response object as `.response`, which can be used to inspect headers, status codes, and other details.

``` python
try:
    client.host.update(host, name="newname")
except APIError as exc:
    if exc.response and exc.response.status_code == 404:
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

Other exceptions are remnants of mreg-cli that are slated to be removed from the library in the future.
These will be rolled into existing exception classes, such as[`APIError`][mreg_api.exceptions.APIError]and `MregValidationError`.

- [`EntityNotFound`][mreg_api.exceptions.EntityNotFound]
- [`MultipleEntitiesFound`][mreg_api.exceptions.MultipleEntitiesFound]
- [`EntityAlreadyExists`][mreg_api.exceptions.EntityAlreadyExists]
- [`EntityRelationMissing`][mreg_api.exceptions.EntityRelationMissing]
- [`InputFailure`][mreg_api.exceptions.InputFailure]
- [`IPNetworkError`][mreg_api.exceptions.IPNetworkError]

## Reference

A complete API reference for all exceptions is available in the [Exceptions reference](../reference/exceptions.md) section.
