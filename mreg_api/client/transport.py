"""Transport-layer response models and validators."""

from __future__ import annotations

from httpx import Response
from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import ValidationError
from pydantic import field_validator

from mreg_api.exceptions import MregValidationError
from mreg_api.types import Json


class PaginatedResponse(BaseModel):
    """Paginated response data from the API."""

    count: int
    next: str | None  # noqa: A003
    previous: str | None
    results: list[Json]

    @field_validator("count", mode="before")
    @classmethod
    def _none_count_is_0(cls, v: object) -> object:
        """Ensure `count` is never `None`."""
        # Django count doesn't seem to be guaranteed to be an integer.
        # https://github.com/django/django/blob/bcbc4b9b8a4a47c8e045b060a9860a5c038192de/django/core/paginator.py#L105-L111
        # Theoretically any callable can be passed to the "count" attribute of the paginator.
        # Ensures here that None (and any falsey value) is treated as 0.
        return v if v else 0

    @classmethod
    def from_response(cls, response: Response) -> PaginatedResponse:
        """Create a PaginatedResponse from a Response."""
        return cls.model_validate_json(response.text)


ListResponse = TypeAdapter(list[Json])
"""JSON list (array) response adapter."""


# TODO: Provide better validation error introspection
def validate_list_response(response: Response) -> list[Json]:
    """Parse and validate that a response contains a JSON array.

    :param response: The response to validate.
    :raises MregValidationError: If the response does not contain a valid JSON array.
    :returns: Parsed response data as a list of Python objects.
    """
    try:
        return ListResponse.validate_json(response.text)
    # NOTE: ValueError catches custom Pydantic errors too
    except ValidationError as e:
        raise MregValidationError.from_pydantic(e, "JSON list") from e


def validate_paginated_response(response: Response) -> PaginatedResponse:
    """Validate and parse that a response contains paginated JSON data.

    :param response: The response to validate.
    :raises MregValidationError: If the response does not contain valid paginated JSON.
    :returns: Parsed response data as a PaginatedResponse object.
    """
    try:
        return PaginatedResponse.from_response(response)
    # NOTE: ValueError catches custom Pydantic errors too
    except ValidationError as e:
        raise MregValidationError.from_pydantic(e, "paginated JSON") from e
