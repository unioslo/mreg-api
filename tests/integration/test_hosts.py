from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import Host
from mreg_api.models.models import Network
from mreg_api.models.models import Zone

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    zone: Zone,
) -> Host:
    hostname = f"{test_prefix}h.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="integration test host",
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    return h


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    zone: Zone,
) -> None:
    hostname = f"{test_prefix}hc.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="create test",
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    assert str(h.name).endswith(zone.name)


def test_get_by_name(integration_client: MregClient, host: Host) -> None:
    result = integration_client.host.get(host.name)
    assert result is not None
    assert result.id == host.id


def test_get_by_id(integration_client: MregClient, host: Host) -> None:
    result = integration_client.host.get_by_id(host.id)
    assert result is not None
    assert result.id == host.id


def test_get_by_object(integration_client: MregClient, host: Host) -> None:
    result = integration_client.host.get(host)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == host.id


def test_get_nonexistent_returns_none(integration_client: MregClient, zone: Zone) -> None:
    result = integration_client.host.get(f"nope.{zone.name}", required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient, zone: Zone) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.host.get(f"nope.{zone.name}")


def test_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    zone: Zone,
) -> None:
    hostname = f"{test_prefix}hdid.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="delete by id test",
        fetch_after_create=True,
    )
    assert h is not None
    integration_client.host.delete(h.id)
    assert integration_client.host.get_by_id(h.id, required=False) is None


def test_delete_by_name(
    integration_client: MregClient,
    test_prefix: str,
    zone: Zone,
) -> None:
    hostname = f"{test_prefix}hdn.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="delete by name test",
        fetch_after_create=True,
    )
    assert h is not None
    integration_client.host.delete(hostname)
    assert integration_client.host.get_by_id(h.id, required=False) is None


def test_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    zone: Zone,
) -> None:
    hostname = f"{test_prefix}hdo.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="delete by object test",
        fetch_after_create=True,
    )
    assert h is not None
    integration_client.host.delete(h)
    assert integration_client.host.get_by_id(h.id, required=False) is None


def test_list(integration_client: MregClient, host: Host) -> None:
    results = integration_client.host.list()
    assert host.id in {r.id for r in results}


def test_list_by_name_regex(
    integration_client: MregClient,
    host: Host,
    test_prefix: str,
) -> None:
    results = integration_client.host.list_by_name_regex(test_prefix)
    assert host in results


def test_count(integration_client: MregClient) -> None:
    result = integration_client.host.count()
    assert isinstance(result, int)
    assert result >= 0


def test_first(integration_client: MregClient) -> None:
    result = integration_client.host.first(required=False)
    assert result is None or isinstance(result, Host)


def test_rename(
    integration_client: MregClient,
    test_prefix: str,
    zone: Zone,
) -> None:
    old_name = f"{test_prefix}hold.{zone.name}"
    new_name = f"{test_prefix}hnew.{zone.name}"
    h = integration_client.host.create(
        name=old_name,
        comment="rename test",
        fetch_after_create=True,
    )
    assert h is not None
    try:
        integration_client.host.rename(h, new_name)
        assert integration_client.host.get_by_name(new_name, required=False) is not None
        assert integration_client.host.get_by_name(old_name, required=False) is None
    finally:
        for name in [new_name, old_name]:
            try:
                integration_client.host.delete(name)
            except Exception:
                pass


def test_get_by_ip(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    zone: Zone,
) -> None:
    ip = "10.0.99.1"
    hostname = f"{test_prefix}hbip.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="get by ip test",
        ipaddress=ip,
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    result = integration_client.host.get_by_ip(ip)
    assert result is not None
    assert result.id == h.id


def test_get_networks(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    zone: Zone,
    test_network: str,
) -> None:
    ip = "10.0.99.2"
    hostname = f"{test_prefix}hgn.{zone.name}"
    h = integration_client.host.create(
        name=hostname,
        comment="get networks test",
        ipaddress=ip,
        fetch_after_create=True,
    )
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    networks = integration_client.host.networks(h)
    assert isinstance(networks, dict)
    assert len(networks) >= 1
    assert all(isinstance(k, Network) for k in networks)


def test_ensure_absent_nonexistent(integration_client: MregClient, zone: Zone) -> None:
    integration_client.host.ensure_absent(f"nope.{zone.name}")


def test_ensure_absent_existing(
    integration_client: MregClient,
    host: Host,
) -> None:
    with pytest.raises(EntityAlreadyExists):
        integration_client.host.ensure_absent(str(host.name))
