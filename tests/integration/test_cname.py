from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import CNAME
from mreg_api.models.models import Host
from mreg_api.models.models import Zone

pytestmark = [pytest.mark.integration]

DOMAIN = "subzone.example.com"


@pytest.fixture(scope="module")
def zone(integration_client: MregClient) -> Zone:
    z = integration_client.zone.get(DOMAIN, required=False)
    if z is None:
        pytest.skip(f"Zone {DOMAIN!r} not found; run ci/seed.py first")
    return z


@pytest.fixture(scope="module")
def host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
    zone: Zone,
) -> Host:
    name = f"{test_prefix}cnameh.{DOMAIN}"
    h = integration_client.host.create(name=name, fetch_after_create=True)
    assert h is not None
    resource_tracker.append(lambda: integration_client.host.delete(h))
    return h


@pytest.fixture(scope="module")
def alias_fqdn(test_prefix: str) -> str:
    return f"{test_prefix}alias.{DOMAIN}"


@pytest.fixture(scope="module")
def cname(
    integration_client: MregClient,
    host: Host,
    alias_fqdn: str,
    resource_tracker: list[Callable[[], Any]],
) -> CNAME:
    integration_client.cname.create(host=host, name=alias_fqdn)
    c = integration_client.cname.get_by_name(alias_fqdn)
    assert c is not None
    resource_tracker.append(lambda: integration_client.cname.delete(c))
    return c


def test_create(
    integration_client: MregClient,
    host: Host,
    test_prefix: str,
    resource_tracker: list[Callable[[], Any]],
) -> None:
    alias = f"{test_prefix}aliasc.{DOMAIN}"
    integration_client.cname.create(host=host, name=alias)
    c = integration_client.cname.get_by_name(alias)
    assert c is not None
    resource_tracker.append(lambda: integration_client.cname.delete(c))
    assert c.name
    assert c.host == host.id


def test_get_by_id(
    integration_client: MregClient,
    cname: CNAME,
) -> None:
    result = integration_client.cname.get(cname.id)
    assert result is not None
    assert result.id == cname.id


def test_get_by_name(
    integration_client: MregClient,
    cname: CNAME,
    alias_fqdn: str,
) -> None:
    result = integration_client.cname.get(alias_fqdn)
    assert result is not None
    assert result.id == cname.id


def test_get_by_object(
    integration_client: MregClient,
    cname: CNAME,
) -> None:
    result = integration_client.cname.get(cname)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == cname.id


def test_get_by_host_and_name(
    integration_client: MregClient,
    host: Host,
    cname: CNAME,
    alias_fqdn: str,
) -> None:
    result = integration_client.cname.get_by_host_and_name(host, alias_fqdn)
    assert result is not None
    assert result.id == cname.id


def test_get_nonexistent_returns_none(
    integration_client: MregClient,
) -> None:
    result = integration_client.cname.get(f"nope.{DOMAIN}", required=False)
    assert result is None


def test_get_nonexistent_raises(
    integration_client: MregClient,
) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.cname.get(f"nope.{DOMAIN}")


def test_delete_by_id(
    integration_client: MregClient,
    host: Host,
    test_prefix: str,
) -> None:
    alias = f"{test_prefix}aliasdi.{DOMAIN}"
    integration_client.cname.create(host=host, name=alias)
    c = integration_client.cname.get_by_name(alias)
    assert c is not None
    integration_client.cname.delete(c.id)
    assert integration_client.cname.get(c.id, required=False) is None


def test_delete_by_name(
    integration_client: MregClient,
    host: Host,
    test_prefix: str,
) -> None:
    alias = f"{test_prefix}aliasdn.{DOMAIN}"
    integration_client.cname.create(host=host, name=alias)
    c = integration_client.cname.get_by_name(alias)
    assert c is not None
    integration_client.cname.delete(alias)
    assert integration_client.cname.get_by_name(alias, required=False) is None


def test_delete_by_object(
    integration_client: MregClient,
    host: Host,
    test_prefix: str,
) -> None:
    alias = f"{test_prefix}aliasdo.{DOMAIN}"
    integration_client.cname.create(host=host, name=alias)
    c = integration_client.cname.get_by_name(alias)
    assert c is not None
    integration_client.cname.delete(c)
    assert integration_client.cname.get(c.id, required=False) is None


def test_list(
    integration_client: MregClient,
    cname: CNAME,
) -> None:
    results = integration_client.cname.list()
    assert cname.id in {r.id for r in results}


def test_count(
    integration_client: MregClient,
) -> None:
    result = integration_client.cname.count()
    assert isinstance(result, int)
    assert result >= 0
