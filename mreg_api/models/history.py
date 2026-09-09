"""History abstractions for mreg-cli."""

from __future__ import annotations

import datetime
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

from mreg_api.types import JsonMapping
from mreg_api.types import parse_json_mapping_string

logger = logging.getLogger(__name__)


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


class HistoryUpdate(BaseModel):
    """Data for history of updates to an object."""

    current_data: JsonMapping
    update: JsonMapping

    def render(self, model: str) -> str:
        """Render the history data as a human-readable string."""
        changes: list[str] = []
        if model == "Ipaddress" and (ipaddr := self.current_data.get("ipaddress")):
            changes.append(str(ipaddr))
        for k, v in self.update.items():
            changes.append(f"{k}: {self.current_data.get(k) or 'not set'} -> {v or 'not set'}")
        return ", ".join(changes)


class HistoryRelation(BaseModel):
    """Data for history of addition/removal of relations."""

    relation: str
    id: int | str  # this path stores id as a string
    name: str

    def render(self, action: str, resource: str, target: str) -> str:
        """Render the history data as a human-readable string."""
        direction = "to" if action == "add" else "from"
        cls = resource.rsplit(".", 1)[-1].replace("HostPolicy_", "").lower()
        return f"{self.relation[:-1]} {self.name} {direction} {cls} {target}"


def _render_fields(data: JsonMapping) -> str:
    return ", ".join(f"{k} = '{v}'" for k, v in data.items())


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

    @property
    def message(self) -> str:
        """Human-readable description of this history entry."""
        match self.action:
            case "add" | "remove":
                return HistoryRelation.model_validate(self.data).render(
                    self.action, str(self.resource), self.name
                )
            case "update":
                return HistoryUpdate.model_validate(self.data).render(self.model)
            case "create":
                return _render_fields(self.data)
            case "destroy":
                if self.model == "Host":
                    return f"deleted {self.name}"
                return _render_fields(self.data)
            case _:
                logger.error("Unhandled history action: %s", self.action)
                try:
                    return _render_fields(self.data)
                except Exception:
                    return str(self.data)
