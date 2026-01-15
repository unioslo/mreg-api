"""Exception classes for MREG API errors and warnings."""

from __future__ import annotations

import logging

from httpx import Response
from pydantic import BaseModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class MregApiBaseError(Exception):
    """Base exception class for MREG API exceptions."""

    pass


class InternalError(MregApiBaseError):
    """Error class for internal errors."""

    pass


# TODO: rename or consolidate to show that this error type requires
# a request object to instantiate.
class APIError(MregApiBaseError):
    """Warning class for API errors."""

    message: str

    def __init__(self, message: str, response: Response | None = None):
        """Initialize an APIError warning.

        :param message: The warning message.
        :param response: The response object that triggered the exception.
        """
        super().__init__(message)
        self.message = message
        self._response = response

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


class PostError(APIError):
    """Error class for failed creation."""

    pass


class PatchError(APIError):
    """Error class for failed patching."""

    pass


class DeleteError(APIError):
    """Error class for failed deletion."""

    pass


class GetError(APIError):
    """Error class for failed retrieval."""

    pass


class UnexpectedDataError(APIError):
    """Error class for unexpected API data."""

    pass


class MregValidationError(MregApiBaseError):
    """Error class for validation failures."""

    def __init__(self, message: str, pydantic_error: ValidationError | None = None):
        """Initialize an MregValidationError.

        :param message: The error message.
        """
        super().__init__(message)
        self.pydantic_error = pydantic_error

    @classmethod
    def from_pydantic(cls, e: ValidationError) -> MregValidationError:
        """Create an MregValidationError from a Pydantic MregValidationError.

        :param e: The Pydantic MregValidationError.
        :returns: The created MregValidationError.
        """
        from mreg_api.client import last_request_method  # noqa: PLC0415
        from mreg_api.client import last_request_url  # noqa: PLC0415

        # Display a title containing the HTTP method and URL if available
        method = last_request_method.get()
        url = last_request_url.get()
        msg = f"Failed to validate {e.title}"
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


class FileError(MregApiBaseError):
    """Error class for file errors."""

    pass


class TooManyResults(MregApiBaseError):
    """Warning class for too many results."""

    pass


class NoHistoryFound(MregApiBaseError):
    """Warning class for no history found."""

    pass


class EntityNotFound(MregApiBaseError):
    """Warning class for an entity that was not found."""

    pass


class EntityAlreadyExists(MregApiBaseError):
    """Warning class for an entity that already exists."""

    pass


class MultipleEntitiesFound(MregApiBaseError):
    """Warning class for multiple entities found."""

    pass


class EntityOwnershipMismatch(MregApiBaseError):
    """Warning class for an entity that already exists but owned by someone else."""

    pass


class InputFailure(MregApiBaseError, ValueError):
    """Warning class for input failure."""

    pass


class ForceMissing(MregApiBaseError):
    """Warning class for missing force flag."""

    pass


class IPNetworkWarning(ValueError, MregApiBaseError):
    """Warning class for IP network/address warnings."""

    pass


class InvalidIPAddress(IPNetworkWarning):
    """Warning class for an entity that is not an IP address."""

    pass


class InvalidIPv4Address(IPNetworkWarning):
    """Warning class for an entity that is not an IPv4 address."""

    pass


class InvalidIPv6Address(IPNetworkWarning):
    """Warning class for an entity that is not an IPv6 address."""

    pass


class InvalidNetwork(IPNetworkWarning):
    """Warning class for an entity that is not a network."""

    pass


class NetworkOverlap(IPNetworkWarning):
    """Warning class for a networkthat overlaps with another network."""


class LoginFailedError(APIError):
    """Error class for login failure."""


class InvalidAuthTokenError(LoginFailedError):
    """Error class for invalid authentication token."""


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
