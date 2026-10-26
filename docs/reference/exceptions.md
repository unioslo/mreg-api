---
icon: lucide/triangle-alert
---

# Exceptions

All library exceptions derive from
[`MregApiBaseError`][mreg_api.exceptions.MregApiBaseError].

HTTP response status errors ([`ResponseError`][mreg_api.exceptions.ResponseError] and subclasses) compose the original HTTP response and expose parsed error details for errors that derive from the server's `drf-standardized-errors` errors.

::: mreg_api.exceptions
