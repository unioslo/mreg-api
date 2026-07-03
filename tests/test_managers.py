from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pytest

from mreg_api.managers import ResourceManager
from mreg_api.models.abstracts import MregModel


def get_resource_managers() -> Sequence[tuple[str, type[ResourceManager[Any]]]]:
    from mreg_api import managers

    managers_list: list[tuple[str, type[ResourceManager[Any]]]] = []
    for name in dir(managers):
        if name.endswith("Manager"):
            manager_class = getattr(managers, name)
            if (
                inspect.isabstract(manager_class)
                # Must inherit from ResourceManager.
                # As of 2026-07-03, some managers are implemented as
                # standalone classes that do not inherit from ResourceManager,
                # e.g. CommunityManager, ZoneManager, DelegationManager,
                # due to their use of scoped resources that do not play well
                # with the abstractions in ResourceManager.
                or ResourceManager not in manager_class.__mro__
                # Specialized /dhcphosts endpoints (ipv4, ipv6, ipv6byipv4)
                # do not have a path field, so skip them for this test.
                or "Dhcp" in name
            ):
                continue
            if isinstance(manager_class, type):
                managers_list.append((manager_class.__name__, manager_class))
    return managers_list


@pytest.mark.parametrize(
    "manager_name, manager_t", get_resource_managers(), ids=[name for name, _ in get_resource_managers()]
)
def test_manager_path_field(
    manager_name: str,  # pyright: ignore[reportUnusedParameter]  # noqa: ARG001
    manager_t: type[ResourceManager[MregModel]],
    client: Any,
) -> None:
    """Test that the manager's _path_field exists in the model's fields."""
    manager = manager_t(client)

    model_class = manager.model
    path_field = manager._path_field  # pyright: ignore[reportPrivateUsage]

    # Check that model has a field with the same name as the manager's path_field
    assert path_field in model_class.model_fields
