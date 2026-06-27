"""History abstractions for mreg-cli."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

from mreg_api.types import JsonMapping
from mreg_api.types import parse_json_mapping_string


class HistoryResource(str, Enum):
    """History resources for the API.

    Names represent resource names.
    Values represent resource relations.

    Access resource names and relation with the `resource()` and `relation()` methods.
    """

    Host = "hosts"
    Group = "groups"
    HostPolicy_Role = "roles"
    HostPolicy_Atom = "atoms"

    @classmethod
    def _missing_(cls, value: Any) -> HistoryResource:
        v = str(value).lower()
        for resource in cls:
            if resource.value == v:
                return resource
            elif resource.name.lower() == v:
                return resource
        raise ValueError(f"Unknown resource {value}")

    def relation(self) -> str:
        """Get the resource relation."""
        return self.value

    def resource(self) -> str:
        """Get the resource name."""
        return self.name.lower()


class HistoryItem(BaseModel):
    """Represents a history item."""

    id: int  # noqa: A003
    timestamp: datetime.datetime
    user: str
    resource: HistoryResource
    name: str
    mid: int = Field(alias="model_id")  # model_ is an internal pydantic namespace.
    model: str
    action: str
    data: JsonMapping

    @field_validator("data", mode="before")
    def parse_json_data(cls, v: Any) -> Any:
        """Parse the data field as JSON if it's a string."""
        if isinstance(v, dict):
            return v  # pyright: ignore[reportUnknownVariableType]
        try:
            return parse_json_mapping_string(v)
        except ValidationError as e:
            raise ValueError("Failed to parse history data as JSON") from e
