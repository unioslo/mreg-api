"""Exception classes for MREG API errors."""

from __future__ import annotations

import logging
from functools import cached_property

from httpx import Request
from httpx import Response
from pydantic import BaseModel
from pydantic import ValidationError
from typing_extensions import deprecated
from typing_extensions import override

from mreg_api.__about__ import __version__
from mreg_api.types import HTTPMethod

logger = logging.getLogger(__name__)


class MregApiBaseError(Exception):
    """Base exception class for MREG API exceptions."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message


class InternalError(MregApiBaseError):
    """Error class for internal errors."""


class PreconditionError(MregApiBaseError):
    """Client-side guard: an operation's precondition was not met.

    Raised before or independently of an HTTP request when the library refuses to
    perform an operation (e.g. deleting a resource that is still in use).
    """


def _indent(text: str, spaces: int = 2) -> str:
    """Indent every line of `text` by `spaces` spaces."""
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" for line in text.split("\n"))


def _labeled(label: str, value: str, *, width: int = 10) -> str:
    """Render `label` + `value`, aligning continuation lines under the value."""
    pad = " " * (2 + width + 1)
    lines = value.split("\n")
    first = f"  {label:<{width}} {lines[0]}"
    return "\n".join([first, *(f"{pad}{line}" for line in lines[1:])])


class ResponseError(MregApiBaseError):
    """Base class for errors that always carry an HTTP response."""

    def __init__(self, message: str = "", *, response: Response):
        """Initialize a ResponseError.

        Args:
            message: An optional message override. When empty, the message is
                derived from the response via `formatted_message`.
            response: The response object that triggered the exception.
        """
        super().__init__(message)
        self._response: Response = response

    @property
    def response(self) -> Response:
        """The response object associated with the error."""
        return self._response

    @property
    def request(self) -> Request:
        """The request that triggered the exception."""
        return self._response.request

    @property
    def status_code(self) -> int:
        """The HTTP status code of the response."""
        return self._response.status_code


class APIError(ResponseError):
    """Exception class for API errors.

    Parses drf-standardized-errors errors from the MREG API if present in response.
    """

    @override
    def __str__(self) -> str:
        """Return the full formatted error message (request info + details)."""
        return self.formatted_message()

    @cached_property
    def errors(self) -> MREGErrorResponse:
        """Get the parsed MREG errors from the response.

        Returns:
            The MREGErrorResponse object, or a default MREGErrorResponse with type
            "unknown" if the response could not be parsed.
        """
        if errors := parse_mreg_error(self.response):
            return errors
        return MREGErrorResponse(type="unknown")

    @cached_property
    def detail(self) -> str:
        """Clean human-readable error message from the response.

        Joins the detail of each parsed error with "; ". Empty if the response
        has no parseable errors.
        """
        return self.errors.detail

    @cached_property
    def error_message(self) -> str:
        """Verbose error text (with codes), falling back to raw response text."""
        if msg := self.errors.as_str():
            return msg
        if self.response.text:
            return self.response.text
        return ""

    @cached_property
    def error_message_json(self) -> str:
        """Verbose error text (with codes), falling back to raw response text."""
        return self.errors.as_json_str()

    @property
    @deprecated('Use "error_message" instead.')
    def details(self) -> str:
        """Get the error details from the response."""
        return self.error_message

    @property
    @deprecated('Use "error_message_json" instead.')
    def details_json(self) -> str:
        """Get the error details from the response."""
        return self.errors.as_json_str()

    def _not_found_hint(self) -> str | None:
        """Helpful hint for 404s on endpoints that don't exist on the server."""
        resp = self.response
        if resp.status_code != 404 or "The requested resource was not found on this server." not in resp.text:
            return None
        url = str(resp.request.url)
        endpoint = url.split("/api/v1/")[-1] if "/api/v1/" in url else url
        return (
            f"Endpoint not found: '{endpoint}'\n"
            f"This may be because your library version ({__version__}) is:\n"
            f"  - Too old: The endpoint has been removed from the server\n"
            f"  - Too new: You're using a beta feature not yet available on the server"
        )

    def _detail_block(self) -> str:
        """The error detail to show, honoring explicit-message precedence.

        An explicit message (enrichment override) wins over the parsed response
        detail; otherwise fall back to the parsed errors / raw response text.
        """
        if self.message:
            return self.message
        return self.error_message

    def formatted_message(self, *, verbose: bool = False, json: bool = False) -> str:
        """Get a formatted error message including error details.

        Args:
            verbose: Render the labeled multi-field format instead of the compact
                default.
            json: Render the parsed errors as JSON.

        Returns:
            The formatted error message.
        """
        status = f"{self.status_code} {self.response.reason_phrase}".strip()
        request = f"{self.request.method} {self.request.url}"
        parts: list[str] = [status]

        if json:
            parts.append(f"  {request}")
            parts.append(_indent(self.errors.as_json_str()))
            return "\n".join(parts)

        hint = self._not_found_hint()

        if verbose:
            parts.append(_labeled("Request:", request))
            if hint:
                parts.append(_labeled("Hint:", hint))
            elif self.message:
                parts.append(_labeled("Detail:", self.message))
            elif errs := self.errors.errors:
                if len(errs) == 1:
                    err = errs[0]
                    if err.attr:
                        parts.append(_labeled("Attr:", err.attr))
                    parts.append(_labeled("Detail:", err.detail))
                    parts.append(_labeled("Code:", err.code))
                else:
                    parts.append(f"  Errors ({len(errs)}):")
                    for i, err in enumerate(errs, 1):
                        prefix = f"{err.attr}: " if err.attr else ""
                        parts.append(f"    [{i}] {prefix}{err.detail}  ({err.code})")
            elif self.response.text:
                parts.append(_labeled("Detail:", self.response.text))
            return "\n".join(parts)

        # Format 1 (compact default)
        parts.append(f"  {request}")
        detail = hint or self._detail_block()
        if detail:
            parts.append(_indent(detail))
        return "\n".join(parts)


class PostError(APIError):
    """Error class for failed creation."""


class PatchError(APIError):
    """Error class for failed patching."""


class DeleteError(APIError):
    """Error class for failed deletion."""


class GetError(APIError):
    """Error class for failed retrieval."""


class UnexpectedResponseError(APIError):
    """Server returned a success status but a body/content we could not use."""


class MregValidationError(MregApiBaseError):
    """Validation error class for MREG API.

    Stems from Pydantic ValidationError but adds context about
    the API request that caused the validation to fail.
    """

    def __init__(self, message: str, pydantic_error: ValidationError | None = None):
        super().__init__(message)
        self.pydantic_error = pydantic_error

    @classmethod
    def from_pydantic(cls, e: ValidationError, context: str | None = None) -> MregValidationError:
        """Create an MregValidationError from a Pydantic MregValidationError.

        Args:
            e: The Pydantic MregValidationError.
            context: What was being validated (e.g., "JSON", "string", "object").

        Returns:
            The created MregValidationError.
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


class ForceMissing(PreconditionError):
    """A precondition failed that `force=True` would override."""


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

    Args:
        code: The error code to format.

    Returns:
        The formatted error code.
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
    """The error code identifying the type of error."""

    detail: str
    """Human-readable representation of the error."""

    attr: str | None
    """The attribute (field) associated with the error, if any."""

    def fmt_error(self) -> str:
        """Format the error message.

        Returns:
            A formatted error message.
        """
        detail = self.detail.rstrip(".")  # remove trailing period
        code = fmt_error_code(self.code)
        if self.attr:
            return f"{self.attr}: {code} - {detail}"
        return f"{code} - {detail}"


class MREGErrorResponse(BaseModel):
    """MREG error response."""

    type: str
    errors: list[MREGError] = []

    def __bool__(self) -> bool:
        """Response contains MREGError objects if True, else False."""
        return bool(self.errors)

    @cached_property
    def detail(self) -> str:
        """Get the detail field of the error(s).

        Most MREG error responses only contain a single error object.
        """
        return "; ".join([error.detail for error in self.errors])

    def as_str(self) -> str:
        """Convert the error response to a string.

        Returns:
            A string representation of the error response.
        """
        return "\n".join(error.fmt_error() for error in self.errors)

    def as_json_str(self, indent: int = 2) -> str:
        """Convert the error response to a JSON string.

        Args:
            indent: The indentation level for the JSON string.

        Returns:
            A JSON string representation of the error response.
        """
        return self.model_dump_json(indent=indent)


def parse_mreg_error(resp: Response) -> MREGErrorResponse | None:
    """Parse an MREG error response.

    Args:
        resp: The response object to parse.

    Returns:
        A MREGErrorResponse object or None if it cannot be parsed.
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


def determine_http_error_class(method: str) -> type[APIError]:
    """Get the appropriate exception class for a given HTTP method.

    Args:
        method: The HTTP method.

    Returns:
        The exception class corresponding to the HTTP method.
    """
    if t := ERROR_MAPPING.get(method):  # pyright: ignore[reportArgumentType]
        return t
    # NOTE: should be unreachable
    logger.warning("No specific exception class for HTTP method '%s', using generic APIError", method)
    return APIError
