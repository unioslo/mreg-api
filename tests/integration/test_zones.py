from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import ForwardZone
from mreg_api.models.models import ForwardZoneDelegation
from mreg_api.models.models import NameServer
from mreg_api.models.models import ReverseZone
from mreg_api.models.models import ReverseZoneDelegation
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


def test_assert_absent_nonexistent(integration_client: MregClient, main_zone: Zone) -> None:
    integration_client.zone.assert_absent(f"nonexistent-zzz.{main_zone.name}")


def test_assert_absent_existing(
    integration_client: MregClient,
    test_zone: ForwardZone,
) -> None:
    with pytest.raises(EntityAlreadyExists):
        integration_client.zone.assert_absent(test_zone.name)


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


def test_update_soa(integration_client: MregClient, test_zone: ForwardZone) -> None:
    integration_client.zone.update_soa(test_zone, retry=7200)
    refreshed = integration_client.zone.refresh(test_zone)
    assert refreshed.retry == 7200


def test_set_default_ttl(integration_client: MregClient, test_zone: ForwardZone) -> None:
    integration_client.zone.set_default_ttl(test_zone, 3600)
    refreshed = integration_client.zone.refresh(test_zone)
    assert refreshed.default_ttl == 3600


def test_set_nameservers(integration_client: MregClient, test_zone: ForwardZone, seed_ns: str) -> None:
    integration_client.zone.set_nameservers(test_zone, [seed_ns], force=True)
    refreshed = integration_client.zone.refresh(test_zone)
    assert seed_ns in [ns.name for ns in refreshed.nameservers]


# --- Reverse zone fixtures ---
#
# Reverse zone names are octet-based (`b.a.in-addr.arpa` == a.b.0.0/16); the server
# rejects arbitrary prefixes and out-of-range/wrong-depth octets, so we can't reuse
# the alphanumeric `test_prefix`. We rotate the module zone's second octet by the
# current minute (0-59) to reduce collisions across runs, and use fixed high octets
# (>59) for the throwaway zones so they never clash with the module zone.


@pytest.fixture(scope="module")
def reverse_zone_name() -> str:
    """A unique-ish /16 reverse zone under the 10.0.0.0/8 seed network."""
    return f"{datetime.now().minute}.10.in-addr.arpa"


@pytest.fixture(scope="module")
def test_reverse_zone(
    integration_client: MregClient,
    resource_tracker: ResourceTracker,
    seed_ns: str,
    reverse_zone_name: str,
) -> ReverseZone:
    zone = integration_client.zone.create(
        name=reverse_zone_name,
        email=f"hostmaster@{reverse_zone_name}",
        primary_ns=[seed_ns],
        force=True,
    )
    assert zone is not None
    resource_tracker.add(lambda: integration_client.zone.delete(reverse_zone_name, force=True))
    assert isinstance(zone, ReverseZone)
    return zone


@pytest.fixture(scope="module")
def reverse_delegation(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
    seed_ns: str,
    resource_tracker: ResourceTracker,
) -> ReverseZoneDelegation:
    deleg_name = f"1.{test_reverse_zone.name}"
    d = integration_client.delegation.create(
        test_reverse_zone,
        name=deleg_name,
        nameservers=[seed_ns],
        force=True,
    )
    assert d is not None
    resource_tracker.add(lambda: integration_client.delegation.delete(test_reverse_zone, deleg_name))
    assert isinstance(d, ReverseZoneDelegation)
    return d


# --- Reverse zone tests ---


def test_create_reverse_zone(test_reverse_zone: ReverseZone) -> None:
    """Test the reverse zone fixture."""
    assert isinstance(test_reverse_zone, ReverseZone)
    assert test_reverse_zone.name.endswith(".in-addr.arpa")


def test_reverse_get_by_name(integration_client: MregClient, test_reverse_zone: ReverseZone) -> None:
    result = integration_client.zone.get(test_reverse_zone.name)
    assert result is not None
    assert isinstance(result, ReverseZone)
    assert result.name == test_reverse_zone.name


def test_reverse_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.zone.get("252.10.in-addr.arpa", required=False)
    assert result is None


def test_reverse_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.zone.get("252.10.in-addr.arpa")


def test_list_reverse(integration_client: MregClient, test_reverse_zone: ReverseZone) -> None:
    results = integration_client.zone.list_reverse()
    assert any(z.name == test_reverse_zone.name for z in results)


def test_reverse_assert_absent_nonexistent(integration_client: MregClient) -> None:
    integration_client.zone.assert_absent("252.10.in-addr.arpa")


def test_reverse_assert_absent_existing(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
) -> None:
    with pytest.raises(EntityAlreadyExists):
        integration_client.zone.assert_absent(test_reverse_zone.name)


def test_reverse_delete_by_name(
    integration_client: MregClient,
    seed_ns: str,
) -> None:
    zone_name = "254.10.in-addr.arpa"
    zone = integration_client.zone.create(
        name=zone_name,
        email=f"hostmaster@{zone_name}",
        primary_ns=[seed_ns],
        force=True,
    )
    assert zone is not None
    integration_client.zone.delete(zone_name, force=True)
    assert integration_client.zone.get(zone_name, required=False) is None


def test_reverse_delete_by_object(
    integration_client: MregClient,
    seed_ns: str,
) -> None:
    zone_name = "253.10.in-addr.arpa"
    zone = integration_client.zone.create(
        name=zone_name,
        email=f"hostmaster@{zone_name}",
        primary_ns=[seed_ns],
        force=True,
    )
    assert zone is not None
    integration_client.zone.delete(zone, force=True)
    assert integration_client.zone.get(zone_name, required=False) is None


# --- Reverse delegation tests ---


def test_reverse_delegation_create(reverse_delegation: ReverseZoneDelegation) -> None:
    assert isinstance(reverse_delegation, ReverseZoneDelegation)


def test_reverse_delegation_get(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
    reverse_delegation: ReverseZoneDelegation,
) -> None:
    result = integration_client.delegation.get(test_reverse_zone, reverse_delegation.name)
    assert result is not None
    assert result.name == reverse_delegation.name


def test_reverse_delegation_get_nonexistent_returns_none(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
) -> None:
    result = integration_client.delegation.get(
        test_reverse_zone,
        f"3.{test_reverse_zone.name}",
        required=False,
    )
    assert result is None


def test_reverse_delegation_get_nonexistent_raises(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.delegation.get(test_reverse_zone, f"3.{test_reverse_zone.name}")


def test_reverse_delegation_list_by_zone(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
    reverse_delegation: ReverseZoneDelegation,
) -> None:
    results = integration_client.delegation.list_by_zone(test_reverse_zone)
    assert any(d.name == reverse_delegation.name for d in results)


def test_reverse_delegation_delete(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
    seed_ns: str,
) -> None:
    deleg_name = f"2.{test_reverse_zone.name}"
    integration_client.delegation.create(
        test_reverse_zone,
        name=deleg_name,
        nameservers=[seed_ns],
        force=True,
    )
    integration_client.delegation.delete(test_reverse_zone, deleg_name)
    assert integration_client.delegation.get(test_reverse_zone, deleg_name, required=False) is None


# --- Reverse zone SOA / TTL / nameserver tests ---


def test_reverse_update_soa(integration_client: MregClient, test_reverse_zone: ReverseZone) -> None:
    integration_client.zone.update_soa(test_reverse_zone, retry=7200)
    refreshed = integration_client.zone.refresh(test_reverse_zone)
    assert refreshed.retry == 7200


def test_reverse_set_default_ttl(integration_client: MregClient, test_reverse_zone: ReverseZone) -> None:
    integration_client.zone.set_default_ttl(test_reverse_zone, 3600)
    refreshed = integration_client.zone.refresh(test_reverse_zone)
    assert refreshed.default_ttl == 3600


def test_reverse_set_nameservers(
    integration_client: MregClient, test_reverse_zone: ReverseZone, seed_ns: str
) -> None:
    integration_client.zone.set_nameservers(test_reverse_zone, [seed_ns], force=True)
    refreshed = integration_client.zone.refresh(test_reverse_zone)
    assert seed_ns in [ns.name for ns in refreshed.nameservers]


# --- list_subzones tests ---


def test_list_subzones_forward(
    integration_client: MregClient,
    test_zone: ForwardZone,
    seed_ns: str,
) -> None:
    sub_name = f"sub.{test_zone.name}"
    integration_client.zone.create(
        name=sub_name, email=f"hostmaster@{sub_name}", primary_ns=[seed_ns], force=True
    )
    try:
        results = integration_client.zone.list_subzones(test_zone)
        assert all(isinstance(z, ForwardZone) for z in results)
        assert any(z.name == sub_name for z in results)
    finally:
        integration_client.zone.delete(sub_name, force=True)


def test_list_subzones_reverse(
    integration_client: MregClient,
    test_reverse_zone: ReverseZone,
    seed_ns: str,
) -> None:
    sub_name = f"99.{test_reverse_zone.name}"
    integration_client.zone.create(
        name=sub_name, email=f"hostmaster@{sub_name}", primary_ns=[seed_ns], force=True
    )
    try:
        results = integration_client.zone.list_subzones(test_reverse_zone)
        assert all(isinstance(z, ReverseZone) for z in results)
        assert any(z.name == sub_name for z in results)
    finally:
        integration_client.zone.delete(sub_name, force=True)


def test_list_subzones_by_name(
    integration_client: MregClient,
    test_zone: ForwardZone,
    seed_ns: str,
) -> None:
    sub_name = f"sub2.{test_zone.name}"
    integration_client.zone.create(
        name=sub_name, email=f"hostmaster@{sub_name}", primary_ns=[seed_ns], force=True
    )
    try:
        results = integration_client.zone.list_subzones(test_zone.name)
        assert any(z.name == sub_name for z in results)
    finally:
        integration_client.zone.delete(sub_name, force=True)
