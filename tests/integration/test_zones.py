from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import ForwardZone
from mreg_api.models.models import ForwardZoneDelegation
from mreg_api.models.models import NameServer
from mreg_api.models.models import Zone

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def seed_ns(zone: Zone) -> str:
    nameservers = zone.nameservers
    if not nameservers:
        pytest.skip("No nameservers on seed zone")
    return nameservers[0].name


@pytest.fixture(scope="module")
def test_zone(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    seed_ns: str,
    main_zone: Zone,
) -> ForwardZone:
    zone_name = f"{test_prefix}.{main_zone.name}"
    zone = integration_client.zone.create(
        name=zone_name,
        email=f"hostmaster@{zone_name}",
        primary_ns=[seed_ns],
        force=True,
    )
    assert zone is not None
    resource_tracker.add(lambda: integration_client.zone.delete(zone_name, force=True))
    assert isinstance(zone, ForwardZone)
    return zone


@pytest.fixture(scope="module")
def delegation(
    integration_client: MregClient,
    test_zone: ForwardZone,
    seed_ns: str,
    resource_tracker: ResourceTracker,
) -> ForwardZoneDelegation:
    deleg_name = f"deleg.{test_zone.name}"
    d = integration_client.delegation.create(
        test_zone,
        name=deleg_name,
        nameservers=[seed_ns],
        force=True,
    )
    assert d is not None
    resource_tracker.add(lambda: integration_client.delegation.delete(test_zone, deleg_name))
    assert isinstance(d, ForwardZoneDelegation)
    return d


# --- Forward zone tests ---


def test_create_forward_zone(main_zone: Zone, test_zone: ForwardZone) -> None:
    """Test the forward zone fixture."""
    assert isinstance(test_zone, ForwardZone)
    assert test_zone.name.endswith(f".{main_zone.name}")  # redundant?


def test_get_by_name(integration_client: MregClient, test_zone: ForwardZone) -> None:
    result = integration_client.zone.get(test_zone.name)
    assert result is not None
    assert result.name == test_zone.name


def test_get_nonexistent_returns_none(integration_client: MregClient, main_zone: Zone) -> None:
    result = integration_client.zone.get(f"nonexistent-zzz.{main_zone.name}", required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient, main_zone: Zone) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.zone.get(f"nonexistent-zzz.{main_zone.name}")


def test_list_forward(integration_client: MregClient, test_zone: ForwardZone) -> None:
    results = integration_client.zone.list_forward()
    assert any(z.name == test_zone.name for z in results)


def test_ensure_absent_nonexistent(integration_client: MregClient, main_zone: Zone) -> None:
    integration_client.zone.ensure_absent(f"nonexistent-zzz.{main_zone.name}")


def test_ensure_absent_existing(
    integration_client: MregClient,
    test_zone: ForwardZone,
) -> None:
    with pytest.raises(EntityAlreadyExists):
        integration_client.zone.ensure_absent(test_zone.name)


def test_delete_by_name(
    integration_client: MregClient,
    test_prefix: str,
    seed_ns: str,
    main_zone: Zone,
) -> None:
    zone_name = f"{test_prefix}zdn.{main_zone.name}"
    zone = integration_client.zone.create(
        name=zone_name,
        email=f"hostmaster@{zone_name}",
        primary_ns=[seed_ns],
        force=True,
    )
    assert zone is not None
    integration_client.zone.delete(zone_name, force=True)
    assert integration_client.zone.get(zone_name, required=False) is None


def test_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    seed_ns: str,
    main_zone: Zone,
) -> None:
    zone_name = f"{test_prefix}zdo.{main_zone.name}"
    zone = integration_client.zone.create(
        name=zone_name,
        email=f"hostmaster@{zone_name}",
        primary_ns=[seed_ns],
        force=True,
    )
    assert zone is not None
    integration_client.zone.delete(zone, force=True)
    assert integration_client.zone.get(zone_name, required=False) is None


def test_get_from_host_in_zone(
    integration_client: MregClient,
    test_zone: ForwardZone,
) -> None:
    result = integration_client.zone.get_from_host(f"host.{test_zone.name}")
    assert result is not None
    assert result.name == test_zone.name


def test_get_from_host_nonexistent(integration_client: MregClient) -> None:
    result = integration_client.zone.get_from_host("nonexistent.nonexistent.invalid")
    assert result is None


# --- Delegation tests ---


def test_delegation_create(delegation: ForwardZoneDelegation) -> None:
    assert isinstance(delegation, ForwardZoneDelegation)


def test_delegation_get(
    integration_client: MregClient,
    test_zone: ForwardZone,
    delegation: ForwardZoneDelegation,
) -> None:
    result = integration_client.delegation.get(test_zone, delegation.name)
    assert result is not None
    assert result.name == delegation.name


def test_delegation_get_nonexistent_returns_none(
    integration_client: MregClient,
    test_zone: ForwardZone,
) -> None:
    result = integration_client.delegation.get(
        test_zone,
        f"nope.{test_zone.name}",
        required=False,
    )
    assert result is None


def test_delegation_get_nonexistent_raises(
    integration_client: MregClient,
    test_zone: ForwardZone,
) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.delegation.get(test_zone, f"nope.{test_zone.name}")


def test_delegation_list_by_zone(
    integration_client: MregClient,
    test_zone: ForwardZone,
    delegation: ForwardZoneDelegation,
) -> None:
    results = integration_client.delegation.list_by_zone(test_zone)
    assert any(d.name == delegation.name for d in results)


def test_delegation_delete(
    integration_client: MregClient,
    test_zone: ForwardZone,
    seed_ns: str,
) -> None:
    deleg_name = f"tmp.{test_zone.name}"
    integration_client.delegation.create(
        test_zone,
        name=deleg_name,
        nameservers=[seed_ns],
        force=True,
    )
    integration_client.delegation.delete(test_zone, deleg_name)
    assert integration_client.delegation.get(test_zone, deleg_name, required=False) is None


# --- NameServer tests ---


def test_nameserver_list(integration_client: MregClient) -> None:
    results = integration_client.nameserver.list()
    assert isinstance(results, list)


def test_nameserver_count(integration_client: MregClient) -> None:
    result = integration_client.nameserver.count()
    assert isinstance(result, int)
    assert result >= 0


def test_nameserver_get_by_name(integration_client: MregClient, seed_ns: str) -> None:
    result = integration_client.nameserver.get(seed_ns)
    assert result is not None
    assert isinstance(result, NameServer)
    assert result.name == seed_ns
