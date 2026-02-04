"""Exception classes for MREG API errors."""

from __future__ import annotations

import logging
from functools import cached_property

import httpx
from httpx import Request
from httpx import Response
from pydantic import BaseModel
from pydantic import ValidationError

from mreg_api.types import HTTPMethod

logger = logging.getLogger(__name__)


class MregApiBaseError(Exception):
    """Base exception class for MREG API exceptions."""


class InternalError(MregApiBaseError):
    """Error class for internal errors."""


class APIError(MregApiBaseError):
    """Exception class for API errors.

    Parses drf-standardized-errors errors from the MREG API if present in response.
    """

    def __init__(self, message: str, response: Response | None = None):
        """Initialize an APIError exception.

        :param message: The exception message.
        :param response: The response object that triggered the exception.
        """
        super().__init__(message)
        self._response: Response | None = response

    @cached_property
    def errors(self) -> MREGErrorResponse | None:
        """Get the parsed MREG errors from the response.

        :returns: The MREGErrorResponse object or None if not available.
        """
        if self.response:
            return parse_mreg_error(self.response)
        return None

    @cached_property
    def details(self) -> str:
        """Error details from response as plain text."""
        if self.errors and (msg := self.errors.as_str()):
            return msg
        if self.response and self.response.text:
            return self.response.text
        return ""

    @cached_property
    def details_json(self) -> str | None:
        """The error details as JSON string."""
        if self.errors and (msg := self.errors.as_json_str()):
            return msg
        return None

    @property
    def request(self) -> Request | None:
        """Get the request that triggered the exception.

        Uses the request from the cause if not set directly.
        :returns: The request object or None if not set.
        """
        if self._response:
            return self._response.request
        # NOTE: do we want to recurse deeper here? Can we try to access the request
        #       attribute as long as it exists, or do we risk infinite recursion?
        if self.__cause__ and isinstance(self.__cause__, httpx.HTTPError):
            try:
                return self.__cause__.request
            except RuntimeError:
                # Request object not set on the cause
                pass
        return None

    @property
    def response(self) -> Response | None:
        """Get the response object associated with the error.

        Uses the response from the cause if not set directly.

        :returns: The response object or None if not set.
        """
        if self._response:
            return self._response
        if self.__cause__ and isinstance(self.__cause__, APIError):
            return self.__cause__.response
        return None

    def _request_info_str(self) -> str | None:
        """Prefix message with request info if available."""
        if self.response:
            parts = [
                self.response.request.method,
                f'"{self.response.request.url}":',
                f"{self.response.status_code}:",
                f"{self.response.reason_phrase}",
            ]
            return " ".join(parts)
        return None

    def formatted_message(self, *, json: bool = False) -> str:
        """Get a formatted error message including error details.

        Args:
            json (bool, optional): Whether to include error details as JSON. Defaults to False.

        Returns:
            str: The formatted error message.
        """
        parts: list[str] = []

        # Request info prefix
        if request_info := self._request_info_str():
            parts.append(request_info)

        # Error details (JSON or plain text), falling back to exception message
        if json and (details := self.details_json):
            parts.append(details)
        elif details := self.details:
            parts.append(details)
        elif self.args:
            parts.append(str(self.args[0]))

        return "\n".join(p.strip() for p in parts if p.strip())


class PostError(APIError):
    """Error class for failed creation."""


class PatchError(APIError):
    """Error class for failed patching."""


class DeleteError(APIError):
    """Error class for failed deletion."""


class GetError(APIError):
    """Error class for failed retrieval."""


class UnexpectedDataError(APIError):
    """Error class for unexpected API data."""


class MregValidationError(MregApiBaseError):
    """Validation error class for MREG API.

    Stems from Pydantic ValidationError but adds context about
    the API request that caused the validation to fail.
    """

    def __init__(self, message: str, pydantic_error: ValidationError | None = None):
        """Initialize an MregValidationError.

        :param message: The error message.
        """
        super().__init__(message)
        self.pydantic_error = pydantic_error

    @classmethod
    def from_pydantic(cls, e: ValidationError, context: str | None = None) -> MregValidationError:
        """Create an MregValidationError from a Pydantic MregValidationError.

        :param e: The Pydantic MregValidationError.
        :param context: What was being validated (e.g., "JSON", "string", "object").
        :returns: The created MregValidationError.
        """
        from mreg_api.client import last_request_method  # noqa: PLC0415
        from mreg_api.client import last_request_url  # noqa: PLC0415

        # Display a title containing the HTTP method and URL if available
        method = last_request_method.get()
        url = last_request_url.get()

        ctx = context or e.title
        msg = f"Failed to validate {ctx}"
        if url and method:
            msg += f" response from {method.upper()} {url}"

        exc_errors = e.errors()

        # Show the input used to instantiate the model if available
        inp = exc_errors[0]["input"] if exc_errors else ""

        # Show field and reason for each error
        errors: list[str] = []
        for err in exc_errors:
            errlines: list[str] = [
                f"Field: {', '.join(str(l) for l in err['loc'])}",  # noqa: E741
                f"Reason: {err['msg']}",
            ]
            errors.append("\n".join(f"    {line}" for line in errlines))

        err_msg = f"{msg}\n  Input: {inp}\n  Errors:\n" + "\n\n".join(errors)
        return cls(err_msg, e)


class TooManyResults(MregApiBaseError):
    """API returned too many results."""


class EntityNotFound(MregApiBaseError):
    """No entity found when at least one was expected."""


class EntityAlreadyExists(MregApiBaseError):
    """Entity already exists when none was expected."""


class MultipleEntitiesFound(MregApiBaseError):
    """Multiple entities found when only one was expected."""


class EntityOwnershipMismatch(MregApiBaseError):
    """Entity already exists but is owned by someone else."""


class InputFailure(MregApiBaseError, ValueError):
    """Error class for input failure."""


class ForceMissing(MregApiBaseError):
    """Error class for missing force flag."""


class IPNetworkError(ValueError, MregApiBaseError):
    """Error class for IP network/address errors."""


class InvalidIPAddress(IPNetworkError):
    """Entity is not a valid IP address."""


class InvalidIPv4Address(IPNetworkError):
    """Entity is not a valid IPv4 address."""


class InvalidIPv6Address(IPNetworkError):
    """Entity is not a valid IPv6 address."""


class InvalidNetwork(IPNetworkError):
    """Entity is not a valid network."""


class LoginFailedError(APIError):
    """Login failed."""


class InvalidAuthTokenError(LoginFailedError):
    """Invalid authentication token."""


class CacheError(MregApiBaseError):
    """Error related to caching."""


class CacheMiss(CacheError):
    """Cache miss error."""


def fmt_error_code(code: str) -> str:
    """Format the error code.

    :param code: The error code to format.
    :returns: The formatted error code.
    """
    return code.replace("_", " ").title()


# NOTE: BASE CLASS FOR MREGError:
# We cannot use `api.abstracts.FrozenModel` as the base class here because
# it causes an import cycle when attempting to import it from `utilities.api`,
# as `api.abstracts` already imports `utilities.api`.
# This could be solved by any of the following:
#
# 1. Inline imports of `utilities.api.get`, `utilities.api.post`, etc. in
#    model methods in `api.abstracts`
# 2. Move `FrozenModel` to a separate module, free of other imports,
#    where it can be imported from anywhere.
# 3. Accept that this is an internal model where it's fine that it's not immutable,
#    and just use `BaseModel` as the base class.
#
# We choose option 3 for now, as it's the least invasive approach.
class MREGError(BaseModel):
    """Details of an MREG error."""

    code: str
    detail: str
    attr: str | None

    def fmt_error(self) -> str:
        """Format the error message.

        :param field: The field name.
        :param messages: The list of error messages.
        :returns: A formatted error message.
        """
        msg = f"{fmt_error_code(self.code)} - {self.detail}"
        if self.attr:
            msg += f": {self.attr}"
        return msg


class MREGErrorResponse(BaseModel):
    """MREG error response."""

    type: str
    errors: list[MREGError] = []

    def as_str(self) -> str:
        """Convert the error response to a string.

        :returns: A string representation of the error response.
        """
        errors = "; ".join([error.fmt_error() for error in self.errors])
        # NOTE: could result in colon followed by no errors, but it's unlikely
        return f"{fmt_error_code(self.type)}: {errors}"

    def as_json_str(self, indent: int = 2) -> str:
        """Convert the error response to a JSON string.

        :param indent: The indentation level for the JSON string.
        :returns: A JSON string representation of the error response.
        """
        return self.model_dump_json(indent=indent)


def parse_mreg_error(resp: Response) -> MREGErrorResponse | None:
    """Parse an MREG error response.

    :param resp: The response object to parse.

    :returns: A MREGErrorResponse object or None if it cannot be parsed.
    """
    try:
        return MREGErrorResponse.model_validate_json(resp.text)
    except ValidationError:
        logger.error("Failed to parse response text '%s' from %s", resp.text, resp.url)
    return None


ERROR_MAPPING: dict[HTTPMethod, type[APIError]] = {
    "GET": GetError,
    "POST": PostError,
    "PATCH": PatchError,
    "DELETE": DeleteError,
}


def determine_http_error_class(method: HTTPMethod) -> type[APIError]:
    """Get the appropriate exception class for a given HTTP method.

    :param method: The HTTP method.
    :returns: The exception class corresponding to the HTTP method.
    """
    if t := ERROR_MAPPING.get(method):
        return t
    # NOTE: should be unreachable
    logger.warning("No specific exception class for HTTP method '%s', using generic APIError", method)
    return APIError
