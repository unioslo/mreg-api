---
icon: lucide/triangle-alert
---

# Exceptions

All library exceptions derive from
[`MregApiBaseError`][mreg_api.exceptions.MregApiBaseError]. API errors
([`APIError`][mreg_api.exceptions.APIError] and subclasses) carry the originating HTTP
response and expose parsed error details.

::: mreg_api.exceptions
