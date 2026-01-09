"""Choice enums for models."""

from __future__ import annotations

from enum import Enum


class CommunitySortOrder(str, Enum):
    """Sort order for communities."""

    NAME = "name"
    VLAN = "vlan"
    GLOBAL_NAME = "global_name"
