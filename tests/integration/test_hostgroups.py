from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = pytest.mark.integration


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-c"
    group = client.hostgroup.create(name=name, description="integration test group")
    assert group is not None
    assert group.name == name
    resource_tracker.add(lambda: client.hostgroup.delete(name))


def test_get_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-gbid"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    fetched = client.hostgroup.get(group.id)
    assert fetched is not None
    assert fetched.id == group.id
    assert fetched.name == name


def test_get_by_name(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-gbn"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    fetched = client.hostgroup.get(name)
    assert fetched is not None
    assert fetched.name == name


def test_get_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-gbo"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    fetched = client.hostgroup.get_by_name(group.name)
    assert fetched is not None
    assert fetched.id == group.id


def test_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    client = integration_client
    result = client.hostgroup.get("zzz-no-such-hg-xyzzy", required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    client = integration_client
    with pytest.raises(EntityNotFound):
        client.hostgroup.get("zzz-no-such-hg-xyzzy")


def test_get_by_name_nonexistent_returns_none(integration_client: MregClient) -> None:
    client = integration_client
    result = client.hostgroup.get_by_name("zzz-no-such-hg-xyzzy", required=False)
    assert result is None


def test_delete_by_id(integration_client: MregClient, test_prefix: str) -> None:
    client = integration_client
    name = f"{test_prefix}hg-dbi"
    group = client.hostgroup.create(name=name)
    assert group is not None
    client.hostgroup.delete(group.id)
    assert client.hostgroup.get(name, required=False) is None


def test_delete_by_name(integration_client: MregClient, test_prefix: str) -> None:
    client = integration_client
    name = f"{test_prefix}hg-dbn"
    group = client.hostgroup.create(name=name)
    assert group is not None
    client.hostgroup.delete(name)
    assert client.hostgroup.get(name, required=False) is None


def test_delete_by_object(integration_client: MregClient, test_prefix: str) -> None:
    client = integration_client
    name = f"{test_prefix}hg-dbo"
    group = client.hostgroup.create(name=name)
    assert group is not None
    client.hostgroup.delete(group)
    assert client.hostgroup.get(name, required=False) is None


def test_list(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-list"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    groups = client.hostgroup.list()
    assert any(g.name == name for g in groups)


def test_list_by_name_regex(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-rgx"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    results = client.hostgroup.list_by_name_regex(f"{test_prefix}hg-rgx")
    assert any(g.name == name for g in results)


def test_count(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-cnt"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    count = client.hostgroup.count()
    assert count >= 1


def test_first(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-fst"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    result = client.hostgroup.first(required=False)
    assert result is not None


def test_rename(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    old_name = f"{test_prefix}hg-rn-old"
    new_name = f"{test_prefix}hg-rn-new"
    group = client.hostgroup.create(name=old_name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(new_name))
    renamed = client.hostgroup.rename(group, new_name)
    assert renamed.name == new_name
    assert client.hostgroup.get(old_name, required=False) is None
    assert client.hostgroup.get(new_name, required=False) is not None


def test_ensure_absent_nonexistent(integration_client: MregClient) -> None:
    client = integration_client
    client.hostgroup.ensure_absent("zzz-no-such-hg-xyzzy")


def test_ensure_absent_existing(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    name = f"{test_prefix}hg-ea"
    group = client.hostgroup.create(name=name)
    assert group is not None
    resource_tracker.add(lambda: client.hostgroup.delete(name))
    with pytest.raises(EntityAlreadyExists):
        client.hostgroup.ensure_absent(name)


def test_add_remove_subgroup(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    parent_name = f"{test_prefix}hg-par"
    child_name = f"{test_prefix}hg-chd"

    parent = client.hostgroup.create(name=parent_name)
    child = client.hostgroup.create(name=child_name)
    assert parent is not None
    assert child is not None

    # Cleanup: child first so parent has no subgroup at deletion time
    resource_tracker.add(lambda: client.hostgroup.delete(child_name))
    resource_tracker.add(lambda: client.hostgroup.delete(parent_name))

    updated_parent = client.hostgroup.add_group(parent, child)
    assert child.name in updated_parent.groups

    updated_parent = client.hostgroup.remove_group(updated_parent, child)
    assert child.name not in updated_parent.groups


def test_list_parents(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    parent_name = f"{test_prefix}hg-lp-par"
    child_name = f"{test_prefix}hg-lp-chd"

    parent = client.hostgroup.create(name=parent_name)
    child = client.hostgroup.create(name=child_name)
    assert parent is not None
    assert child is not None

    resource_tracker.add(lambda: client.hostgroup.delete(child_name))
    resource_tracker.add(lambda: client.hostgroup.delete(parent_name))

    client.hostgroup.add_group(parent, child)
    parents = client.hostgroup.list_parents(child_name)
    assert any(p.name == parent_name for p in parents)


def test_add_remove_host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    client = integration_client
    group_name = f"{test_prefix}hg-rh"
    host_name = f"{test_prefix}hh.example.com"

    group = client.hostgroup.create(name=group_name)
    host = client.host.create(name=host_name)
    assert group is not None
    assert host is not None

    resource_tracker.add(lambda: client.host.delete(host_name))
    resource_tracker.add(lambda: client.hostgroup.delete(group_name))
    resource_tracker.add(lambda: client.hostgroup.remove_host(group_name, host_name))

    updated = client.hostgroup.add_host(group, host)
    assert host.name in updated.hosts

    updated = client.hostgroup.remove_host(updated, host)
    assert host.name not in updated.hosts
