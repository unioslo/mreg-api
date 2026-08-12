from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import Label

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(integration_client: MregClient) -> MregClient:
    return integration_client


@pytest.fixture(scope="module")
def label(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> Label:
    name = f"{test_prefix}lbl"
    integration_client.label.create(
        name=name,
        description="integration test label",
    )
    lbl = integration_client.label.get_by_name(name)
    assert lbl is not None
    resource_tracker.add(lambda: integration_client.label.delete(lbl))
    return lbl


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    name = f"{test_prefix}lc"
    integration_client.label.create(
        name=name,
        description="create test",
    )
    lbl = integration_client.label.get_by_name(name)
    assert lbl is not None
    resource_tracker.add(lambda: integration_client.label.delete(lbl))
    assert lbl.name == f"{test_prefix}lc"


def test_get_by_id(integration_client: MregClient, label: Label) -> None:
    result = integration_client.label.get(label.id)
    assert result is not None
    assert result.id == label.id


def test_get_by_name(integration_client: MregClient, label: Label) -> None:
    result = integration_client.label.get(label.name)
    assert result is not None
    assert result.id == label.id


def test_get_by_object(integration_client: MregClient, label: Label) -> None:
    result = integration_client.label.get(label)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == label.id


def test_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.label.get(99999999, required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.label.get(99999999)


def test_get_by_name_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.label.get_by_name("nope", required=False)
    assert result is None


def test_delete_by_id(integration_client: MregClient, test_prefix: str) -> None:
    name = f"{test_prefix}ldid"
    integration_client.label.create(
        name=name,
        description="delete by id test",
    )
    lbl = integration_client.label.get_by_name(name)
    assert lbl is not None
    integration_client.label.delete(lbl.id)
    assert integration_client.label.get(lbl.id, required=False) is None


def test_delete_by_name(integration_client: MregClient, test_prefix: str) -> None:
    name = f"{test_prefix}ldn"
    integration_client.label.create(
        name=name,
        description="delete by name test",
    )
    lbl = integration_client.label.get_by_name(name)
    assert lbl is not None
    integration_client.label.delete(lbl.name)
    assert integration_client.label.get(lbl.id, required=False) is None


def test_delete_by_object(integration_client: MregClient, test_prefix: str) -> None:
    name = f"{test_prefix}ldo"
    integration_client.label.create(
        name=name,
        description="delete by object test",
    )
    lbl = integration_client.label.get_by_name(name)
    assert lbl is not None
    integration_client.label.delete(lbl)
    assert integration_client.label.get(lbl.id, required=False) is None


def test_list(integration_client: MregClient, label: Label) -> None:
    results = integration_client.label.list()
    assert label.id in {r.id for r in results}


def test_list_by_name_regex(
    integration_client: MregClient,
    label: Label,
    test_prefix: str,
) -> None:
    results = integration_client.label.list_by_name_regex(test_prefix)
    assert label in results


def test_count(integration_client: MregClient) -> None:
    result = integration_client.label.count()
    assert isinstance(result, int)
    assert result >= 0


def test_first(integration_client: MregClient) -> None:
    result = integration_client.label.first()
    assert result is None or isinstance(result, Label)


def test_rename(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    old_name = f"{test_prefix}lold"
    new_name = f"{test_prefix}lnew"
    integration_client.label.create(
        name=old_name,
        description="rename test",
    )
    lbl = integration_client.label.get_by_name(old_name)
    assert lbl is not None
    try:
        integration_client.label.rename(lbl, new_name)
        assert integration_client.label.get_by_name(new_name, required=False) is not None
        assert integration_client.label.get_by_name(old_name, required=False) is None
    finally:
        for name in [new_name, old_name]:
            try:
                integration_client.label.delete(name)
            except Exception:
                pass


def test_ensure_absent_nonexistent(integration_client: MregClient) -> None:
    integration_client.label.ensure_absent(99999999)


def test_ensure_absent_existing(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    name = f"{test_prefix}lea"
    integration_client.label.create(
        name=name,
        description="ensure absent test",
    )
    lbl = integration_client.label.get_by_name(name)
    assert lbl is not None
    resource_tracker.add(lambda: integration_client.label.delete(lbl))
    with pytest.raises(EntityAlreadyExists):
        integration_client.label.ensure_absent(lbl.id)
