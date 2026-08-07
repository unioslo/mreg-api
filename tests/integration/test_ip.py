from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import Host
from mreg_api.models.models import IPAddress

DOMAIN = "example.uio.no"

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def host_zone(integration_client: MregClient) -> object:
    zone = integration_client.zone.get(DOMAIN, required=False)
    if zone is None:
        pytest.skip(f"Zone {DOMAIN!r} not found; run ci/seed.py first")
    return zone


@pytest.fixture(scope="module")
def host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    host_zone: object,
) -> Host:
    hostname = f"{test_prefix}iph.{DOMAIN}"
    h = integration_client.host.create(
        name=hostname,
        comment="integration test host for ip tests",
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    return h


@pytest.fixture(scope="module")
def ip(
    integration_client: MregClient,
    host: Host,
    resource_tracker: list[Callable[[], Any]],
) -> IPAddress:
    addr = "10.0.99.10"
    integration_client.ipaddress.create(ipaddress=addr, host=host)
    created = integration_client.ipaddress.get(addr)
    assert created is not None
    resource_tracker.append(lambda: integration_client.ipaddress.delete(created))
    return created


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    host_zone: object,
) -> None:
    hostname = f"{test_prefix}ipc.{DOMAIN}"
    h = integration_client.host.create(
        name=hostname,
        comment="ip create test host",
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    addr = "10.0.99.20"
    integration_client.ipaddress.create(ipaddress=addr, host=h)
    created = integration_client.ipaddress.get(addr)
    assert created is not None
    resource_tracker.append(lambda: integration_client.ipaddress.delete(created))
    assert str(created.ipaddress) == addr


def test_get_by_id(integration_client: MregClient, ip: IPAddress) -> None:
    result = integration_client.ipaddress.get(ip.id)
    assert result is not None
    assert result.id == ip.id


def test_get_by_ip_str(integration_client: MregClient, ip: IPAddress) -> None:
    result = integration_client.ipaddress.get(str(ip.ipaddress))
    assert result is not None
    assert result.id == ip.id


def test_get_by_object(integration_client: MregClient, ip: IPAddress) -> None:
    result = integration_client.ipaddress.get(ip)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == ip.id


def test_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.ipaddress.get(99999999, required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.ipaddress.get(99999999)


def test_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    host_zone: object,
) -> None:
    hostname = f"{test_prefix}ipdid.{DOMAIN}"
    h = integration_client.host.create(
        name=hostname,
        comment="ip delete by id test host",
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    addr = "10.0.99.11"
    integration_client.ipaddress.create(ipaddress=addr, host=h)
    created = integration_client.ipaddress.get(addr)
    assert created is not None
    integration_client.ipaddress.delete(created.id)
    assert integration_client.ipaddress.get(created.id, required=False) is None


def test_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    host_zone: object,
) -> None:
    hostname = f"{test_prefix}ipdobj.{DOMAIN}"
    h = integration_client.host.create(
        name=hostname,
        comment="ip delete by object test host",
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    addr = "10.0.99.12"
    integration_client.ipaddress.create(ipaddress=addr, host=h)
    created = integration_client.ipaddress.get(addr)
    assert created is not None
    integration_client.ipaddress.delete(created)
    assert integration_client.ipaddress.get(created.id, required=False) is None


def test_list(integration_client: MregClient, ip: IPAddress) -> None:
    results = integration_client.ipaddress.list()
    assert ip.id in {r.id for r in results}


def test_count(integration_client: MregClient) -> None:
    result = integration_client.ipaddress.count()
    assert isinstance(result, int)
    assert result >= 0


def test_first(integration_client: MregClient) -> None:
    result = integration_client.ipaddress.first(required=False)
    assert result is None or isinstance(result, IPAddress)
