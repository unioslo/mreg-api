"""Exception classes for MREG API errors and warnings."""

from __future__ import annotations

import logging

from pydantic import ValidationError as PydanticValidationError
from requests import Response

logger = logging.getLogger(__name__)


class MregApiBaseError(Exception):
    """Base exception class for MREG API exceptions."""

    pass


class PostError(MregApiBaseError):
    """Error class for failed creation."""

    pass


class PatchError(MregApiBaseError):
    """Error class for failed patching."""

    pass


class DeleteError(MregApiBaseError):
    """Error class for failed deletion."""

    pass


class GetError(MregApiBaseError):
    """Error class for failed retrieval."""

    pass


class InternalError(MregApiBaseError):
    """Error class for internal errors."""

    pass


# TODO: rename or consolidate to show that this error type requires
# a request object to instantiate.
class APIError(MregApiBaseError):
    """Warning class for API errors."""

    response: Response

    def __init__(self, message: str, response: Response):
        """Initialize an APIError warning.

        :param message: The warning message.
        :param response: The response object that triggered the exception.
        """
        super().__init__(message)
        self.response = response


class UnexpectedDataError(APIError):
    """Error class for unexpected API data."""

    pass


class MregValidationError(MregApiBaseError):
    """Error class for validation failures."""

    def __init__(self, message: str, pydantic_error: PydanticValidationError | None = None):
        """Initialize an MregValidationError.

        :param message: The error message.
        """
        super().__init__(message)
        self.pydantic_error = pydantic_error

    @classmethod
    def from_pydantic(cls, e: PydanticValidationError) -> MregValidationError:
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

    pass


class LoginFailedError(MregApiBaseError):
    """Error class for login failure."""
