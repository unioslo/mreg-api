from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import ForwardZone
from mreg_api.models.models import NameServer
from mreg_api.models.models import Zone

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]

# NOTE: Similar tests are also defined in test_zones.py, however, these
# tests define nameservers independently of zones, whereas those in test_zones.py
# are defined implicitly via zone creation.


@pytest.fixture(scope="module")
def seed_ns(
    integration_client: MregClient, test_prefix: str, resource_tracker: ResourceTracker, main_zone: Zone
) -> list[NameServer]:
    """Ensure the zone we will define name servers in exists before proceeding."""
    ns_names = {f"{test_prefix}{name}.{main_zone.name}" for name in ["ns1", "ns2", "ns3"]}
    nameservers: list[NameServer] = []
    for name in ns_names:
        try:
            ns = integration_client.nameserver.create(name=name)
        except EntityAlreadyExists:
            ns = integration_client.nameserver.get(name)
        resource_tracker.add(lambda name=name: integration_client.nameserver.delete(name))
        if ns:
            nameservers.append(ns)
    return nameservers


def test_nameserver_create(
    test_prefix: str,
    integration_client: MregClient,
    resource_tracker: ResourceTracker,
    main_zone: ForwardZone,
    seed_ns: list[NameServer],
) -> None:
    ns_name = f"{test_prefix}ns001.{main_zone.name}"
    ns = integration_client.nameserver.create(name=ns_name)
    resource_tracker.add(lambda: integration_client.nameserver.delete(ns_name))
    assert ns is not None
    assert isinstance(ns, NameServer)
    assert ns.name == ns_name


def test_nameserver_list(
    integration_client: MregClient, seed_ns: list[NameServer], resource_tracker: ResourceTracker
) -> None:
    results = integration_client.nameserver.list()
    assert len(results) >= 3

    # All seeded nameservers should be present in the list of results
    # NOTE: prefer explicit comparison 1-by-1 instead of any() for better
    # test failure diagnostics (i.e., which nameserver is missing)
    seed_ns_names = {ns.name for ns in seed_ns}
    ns_names = {ns.name for ns in results}
    for seed_ns_name in seed_ns_names:
        assert seed_ns_name in ns_names


def test_nameserver_count(integration_client: MregClient, seed_ns: list[NameServer]) -> None:
    result = integration_client.nameserver.count()
    assert isinstance(result, int)
    assert result >= len(seed_ns)


def test_nameserver_get_by_name(integration_client: MregClient, seed_ns: list[NameServer]) -> None:
    for ns in seed_ns:
        result = integration_client.nameserver.get_by_name(ns.name)
        assert result is not None
        assert isinstance(result, NameServer)
        assert result.name == ns.name


def test_nameserver_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    ns_name = f"{test_prefix}nsdid"
    ns = integration_client.nameserver.create(name=ns_name)
    assert ns is not None
    integration_client.nameserver.delete(ns.id)
    assert integration_client.nameserver.get_by_name(ns_name, required=False) is None


def test_nameserver_delete_by_name(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    ns_name = f"{test_prefix}nsdn"
    ns = integration_client.nameserver.create(name=ns_name)
    assert ns is not None
    integration_client.nameserver.delete(ns.name)
    assert integration_client.nameserver.get_by_name(ns_name, required=False) is None


@pytest.mark.xfail(
    strict=True, raises=EntityNotFound, reason="NameServerManager does not automatically expand names"
)
def test_nameserver_delete_by_name_non_fqdn(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    ns_name = f"{test_prefix}nsdnnonfqdn"
    ns = integration_client.nameserver.create(name=ns_name)
    assert ns is not None
    integration_client.nameserver.delete(ns_name)  # not fqdn-expanded, will fail
    assert integration_client.nameserver.get_by_name(ns_name, required=False) is None  # unreachable


def test_nameserver_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    ns_name = f"{test_prefix}nsdo"
    ns = integration_client.nameserver.create(name=ns_name)
    assert ns is not None
    integration_client.nameserver.delete(ns)
    assert integration_client.nameserver.get_by_name(ns_name, required=False) is None
