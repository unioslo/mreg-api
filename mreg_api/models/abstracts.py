"""Abstract models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict


class MregModel(BaseModel):
    """Model for an immutable object."""

    def __setattr__(self, name: str, value: Any):
        """Raise an exception when trying to set an attribute."""
        raise AttributeError("Cannot set attribute on a frozen object")

    def __delattr__(self, name: str):
        """Raise an exception when trying to delete an attribute."""
        raise AttributeError("Cannot delete attribute on a frozen object")

    model_config = ConfigDict(
        # Freeze model to make it immutable and thus hashable.
        frozen=True,
    )


class MregModelWithTimestamps(MregModel):
    """Model with created_at and updated_at fields."""

    created_at: datetime
    updated_at: datetime
