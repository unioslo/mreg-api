"""History abstractions for mreg-cli."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from enum import Enum
from typing import Self
from typing import cast

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator
from typing_extensions import override

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.abstracts import manager_only
from mreg_api.types import ClientProtocol
from mreg_api.types import QueryParams
from mreg_api.types import parse_json_mapping_string


class HistoryResource(str, Enum):
    """History resources for the API.

    Names represent resource names.
    Values represent resource relations.

    Access resource names and relation with the `resource()` and `relation()` methods.
    """

    Host = "hosts"
    Permissions = "permissions"
    Group = "groups"
    HostPolicy_Role = "roles"
    HostPolicy_Atom = "atoms"

    @override
    @classmethod
    def _missing_(cls, value: object) -> HistoryResource:
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
    data: dict[str, object]

    @field_validator("data", mode="before")
    def parse_json_data(cls, v: object) -> dict[str, object]:
        """Parse the data field as JSON if it's a string."""
        if isinstance(v, Mapping):
            mapping_value = cast(Mapping[str, object], v)
            return dict(mapping_value)
        if not isinstance(v, str):
            raise ValueError("Failed to parse history data as JSON")
        try:
            parsed = parse_json_mapping_string(v)
            return cast(dict[str, object], dict(parsed))
        except ValidationError as e:
            raise ValueError("Failed to parse history data as JSON") from e

    @classmethod
    @manager_only
    def get(cls, client: "ClientProtocol", name: str, resource: HistoryResource) -> list[Self]:
        """Get history items for a resource."""
        params: QueryParams = {"resource": resource.resource(), "name": name}
        ret = client.get_typed(Endpoint.History, list[cls], params=params)
        if len(ret) == 0:
            raise EntityNotFound(f"No history found for {name}")

        model_ids = ",".join({str(i.mid) for i in ret})
        params = {
            "resource": resource.resource(),
            "model_id__in": model_ids,
        }
        ret = client.get_typed(Endpoint.History, list[cls], params=params)

        params = {
            "data__relation": resource.relation(),
            "data__id__in": model_ids,
        }
        ret.extend(client.get_typed(Endpoint.History, list[cls], params=params))

        return ret
