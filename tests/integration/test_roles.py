from __future__ import annotations

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound

pytestmark = pytest.mark.integration


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-c"
    role = client.role.create(name=name, description="integration test role")
    assert role is not None
    assert role.name == name
    assert role.description == "integration test role"
    resource_tracker.append(lambda: client.role.delete(name))


def test_get_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-gbid"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    fetched = client.role.get(role.id)
    assert fetched is not None
    assert fetched.id == role.id
    assert fetched.name == name


def test_get_by_name(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-gbn"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    fetched = client.role.get(name)
    assert fetched is not None
    assert fetched.name == name


def test_get_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-gbo"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    fetched = client.role.get_by_name(role.name)
    assert fetched is not None
    assert fetched.id == role.id


def test_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    client = integration_client
    result = client.role.get("zzz-no-such-role-xyzzy", required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    client = integration_client
    with pytest.raises(EntityNotFound):
        client.role.get("zzz-no-such-role-xyzzy")


def test_get_by_name_nonexistent_returns_none(integration_client: MregClient) -> None:
    client = integration_client
    result = client.role.get_by_name("zzz-no-such-role-xyzzy", required=False)
    assert result is None


def test_delete_by_id(integration_client: MregClient, test_prefix: str) -> None:
    client = integration_client
    name = f"{test_prefix}role-dbi"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    client.role.delete(role.id)
    assert client.role.get(name, required=False) is None


def test_delete_by_name(integration_client: MregClient, test_prefix: str) -> None:
    client = integration_client
    name = f"{test_prefix}role-dbn"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    client.role.delete(name)
    assert client.role.get(name, required=False) is None


def test_delete_by_object(integration_client: MregClient, test_prefix: str) -> None:
    client = integration_client
    name = f"{test_prefix}role-dbo"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    client.role.delete(role)
    assert client.role.get(name, required=False) is None


def test_list(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-list"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    roles = client.role.list()
    assert any(r.name == name for r in roles)


def test_list_by_name_regex(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-rgx"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    results = client.role.list_by_name_regex(f"{test_prefix}role-rgx")
    assert any(r.name == name for r in results)


def test_count(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-cnt"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    count = client.role.count()
    assert count >= 1


def test_first(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-fst"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    result = client.role.first(required=False)
    assert result is not None


def test_rename(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    old_name = f"{test_prefix}role-rn-old"
    new_name = f"{test_prefix}role-rn-new"
    role = client.role.create(name=old_name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(new_name))
    renamed = client.role.rename(role, new_name)
    assert renamed.name == new_name
    assert client.role.get(old_name, required=False) is None
    assert client.role.get(new_name, required=False) is not None


def test_ensure_absent_nonexistent(integration_client: MregClient) -> None:
    client = integration_client
    client.role.ensure_absent("zzz-no-such-role-xyzzy")


def test_ensure_absent_existing(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    name = f"{test_prefix}role-ea"
    role = client.role.create(name=name, description="test role")
    assert role is not None
    resource_tracker.append(lambda: client.role.delete(name))
    with pytest.raises(EntityAlreadyExists):
        client.role.ensure_absent(name)


def test_add_remove_atom(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    atom_name = f"{test_prefix}atom-ar"
    role_name = f"{test_prefix}role-ar"

    client.atom.create(name=atom_name, description="test atom")
    atom = client.atom.get_by_name(atom_name)
    role = client.role.create(name=role_name, description="test role")
    assert atom is not None
    assert role is not None

    # Cleanup: role first (atoms don't block role deletion), then atom
    resource_tracker.append(lambda: client.atom.delete(atom_name))
    resource_tracker.append(lambda: client.role.delete(role_name))

    client.role.add_atom(role, atom)
    fresh = client.role.get_by_name(role_name)
    assert atom.name in fresh.atoms

    client.role.remove_atom(fresh, atom)
    after_remove = client.role.get_by_name(role_name)
    assert atom.name not in after_remove.atoms


@pytest.mark.skip(reason="requires zone 'example.com' to exist — ensure seed data is loaded")
def test_add_remove_host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    host_name = f"{test_prefix}rh.example.com"
    role_name = f"{test_prefix}role-rh"

    host = client.host.create(name=host_name)
    role = client.role.create(name=role_name, description="test role")
    assert host is not None
    assert role is not None

    resource_tracker.append(lambda: client.host.delete(host_name))
    resource_tracker.append(lambda: client.role.delete(role_name))
    resource_tracker.append(lambda: client.role.remove_host(role.id, host_name))

    client.role.add_host(role, host)
    fresh = client.role.get_by_name(role_name)
    assert host.name in fresh.hosts

    client.role.remove_host(role, host)
    fresh = client.role.get_by_name(role_name)
    assert host.name not in fresh.hosts


def test_add_remove_label(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    label_name = f"{test_prefix}lbl-ar"
    role_name = f"{test_prefix}role-lr"

    client.label.create(name=label_name, description="test label")
    label = client.label.get_by_name(label_name)
    role = client.role.create(name=role_name, description="test role")
    assert label is not None
    assert role is not None

    resource_tracker.append(lambda: client.label.delete(label_name))
    resource_tracker.append(lambda: client.role.delete(role_name))

    updated_role = client.role.add_label(role, label)
    lbs = client.role.list_labels(updated_role)
    assert any(lb.name == label.name for lb in lbs)

    updated_role = client.role.remove_label(updated_role, label)
    lbs = client.role.list_labels(updated_role)
    assert not any(lb.name == label.name for lb in lbs)


def test_delete_role_with_hosts_raises(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list,
) -> None:
    client = integration_client
    zone = client.zone.get("example.com", required=False)
    if zone is None:
        pytest.skip("requires zone 'example.com' to exist — ensure seed data is loaded")

    host_name = f"{test_prefix}dhr.example.com"
    role_name = f"{test_prefix}role-dhr"

    host = client.host.create(name=host_name)
    role = client.role.create(name=role_name, description="test role")
    assert host is not None
    assert role is not None

    client.role.add_host(role, host)

    # Cleanup: remove membership first, then role, then host
    resource_tracker.append(lambda: client.host.delete(host_name))
    resource_tracker.append(lambda: client.role.delete(role_name))
    resource_tracker.append(lambda: client.role.remove_host(role.id, host_name))

    with pytest.raises(DeleteError):
        client.role.delete(role_name)
