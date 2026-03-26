from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from mreg_api.models import models
from mreg_api.models.models import ClassVarNotSet
from mreg_api.models.models import WithHistory


def get_with_history_subclasses() -> list[type[WithHistory]]:
    """Get all subclasses of WithHistory."""
    subclasses = set[type[WithHistory]]()
    for attr in dir(models):
        obj = getattr(models, attr)
        if isinstance(obj, type) and issubclass(obj, WithHistory) and obj is not WithHistory:
            subclasses.add(obj)
    # Ensure we also include any direct subclasses defined anywhere else
    # (shouldn't happen, but doesn't hurt to be careful)
    subclasses.update(WithHistory.__subclasses__())
    return list(subclasses)


def test_with_history_subclasses_snapshot() -> None:
    """Snapshot test for list of defined WithHistory subclasses."""
    subclasses = get_with_history_subclasses()
    assert len(subclasses) == snapshot(4)
    assert sorted((cls.__name__ for cls in subclasses)) == snapshot(
        ["Atom", "Host", "HostGroup", "Role"]
    )


@pytest.mark.parametrize("subclass", get_with_history_subclasses())
def test_with_history_subclass_requirements(subclass: type[WithHistory]) -> None:
    """Test basic assumptions/requirements for WithHistory subclasses."""
    # Class var must be set
    assert subclass.history_resource != ClassVarNotSet
    # Must inherit from pydantic.BaseModel
    assert issubclass(subclass, BaseModel)
