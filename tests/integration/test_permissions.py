from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import Permission

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def permission(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> Permission:
    group = f"{test_prefix}grp"
    range_ = "10.0.0.0/8"
    regex = f".*{test_prefix}.*"
    integration_client.permission.create(
        group=group,
        range=range_,
        regex=regex,
    )
    perm = integration_client.permission.get_by_triplet(group, range_, regex)
    assert perm is not None
    resource_tracker.add(lambda: integration_client.permission.delete(perm))
    return perm


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    group = f"{test_prefix}grpc"
    range_ = "10.0.0.0/8"
    regex = f".*{test_prefix}c.*"
    integration_client.permission.create(
        group=group,
        range=range_,
        regex=regex,
    )
    perm = integration_client.permission.get_by_triplet(group, range_, regex)
    assert perm is not None
    resource_tracker.add(lambda: integration_client.permission.delete(perm))
    assert perm.group == f"{test_prefix}grpc"


def test_get_by_id(integration_client: MregClient, permission: Permission) -> None:
    result = integration_client.permission.get(permission.id)
    assert result is not None
    assert result.id == permission.id


def test_get_by_object(integration_client: MregClient, permission: Permission) -> None:
    result = integration_client.permission.get(permission)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == permission.id


def test_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.permission.get(99999999, required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.permission.get(99999999)


def test_get_by_triplet(
    integration_client: MregClient,
    permission: Permission,
    test_prefix: str,
) -> None:
    result = integration_client.permission.get_by_triplet(
        f"{test_prefix}grp",
        "10.0.0.0/8",
        f".*{test_prefix}.*",
    )
    assert result is not None
    assert result.id == permission.id


def test_get_by_triplet_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.permission.get_by_triplet(
        "nogroup",
        "0.0.0.0/32",
        ".*",
        required=False,
    )
    assert result is None


def test_delete_by_id(integration_client: MregClient, test_prefix: str) -> None:
    group = f"{test_prefix}grpdid"
    range_ = "10.0.0.0/8"
    regex = f".*{test_prefix}did.*"
    integration_client.permission.create(
        group=group,
        range=range_,
        regex=regex,
    )
    perm = integration_client.permission.get_by_triplet(group, range_, regex)
    assert perm is not None
    integration_client.permission.delete(perm.id)
    assert integration_client.permission.get(perm.id, required=False) is None


def test_delete_by_object(integration_client: MregClient, test_prefix: str) -> None:
    group = f"{test_prefix}grpdo"
    range_ = "10.0.0.0/8"
    regex = f".*{test_prefix}do.*"
    integration_client.permission.create(
        group=group,
        range=range_,
        regex=regex,
    )
    perm = integration_client.permission.get_by_triplet(group, range_, regex)
    assert perm is not None
    integration_client.permission.delete(perm)
    assert integration_client.permission.get(perm.id, required=False) is None


def test_list(integration_client: MregClient, permission: Permission) -> None:
    results = integration_client.permission.list()
    assert permission.id in {r.id for r in results}


def test_count(integration_client: MregClient) -> None:
    result = integration_client.permission.count()
    assert isinstance(result, int)
    assert result >= 0


def test_first(integration_client: MregClient) -> None:
    result = integration_client.permission.first(required=False)
    assert result is None or isinstance(result, Permission)


def test_ensure_absent_nonexistent(integration_client: MregClient) -> None:
    integration_client.permission.ensure_absent(99999999)


def test_ensure_absent_existing(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    group = f"{test_prefix}grpea"
    range_ = "10.0.0.0/8"
    regex = f".*{test_prefix}ea.*"
    integration_client.permission.create(
        group=group,
        range=range_,
        regex=regex,
    )
    perm = integration_client.permission.get_by_triplet(group, range_, regex)
    assert perm is not None
    resource_tracker.add(lambda: integration_client.permission.delete(perm))
    with pytest.raises(EntityAlreadyExists):
        integration_client.permission.ensure_absent(perm.id)


def test_add_remove_label(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    group = f"{test_prefix}grplbl"
    range_ = "10.0.0.0/8"
    regex = f".*{test_prefix}lbl.*"
    integration_client.permission.create(
        group=group,
        range=range_,
        regex=regex,
    )
    perm = integration_client.permission.get_by_triplet(group, range_, regex)
    assert perm is not None
    resource_tracker.add(lambda: integration_client.permission.delete(perm))

    integration_client.label.create(
        name=f"{test_prefix}plbl",
        description="permission label test",
    )
    lbl = integration_client.label.get_by_name(f"{test_prefix}plbl")
    assert lbl is not None
    resource_tracker.add(lambda: integration_client.label.delete(lbl))

    updated = integration_client.permission.add_label(perm, lbl)
    assert lbl.id in updated.labels

    removed = integration_client.permission.remove_label(updated, lbl)
    assert lbl.id not in removed.labels
