from __future__ import annotations

from ipaddress import IPv4Address
from ipaddress import IPv6Address
from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import Community
from mreg_api.models.models import Network

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# NetworkManager
# ---------------------------------------------------------------------------


def test_create(
    integration_client: MregClient,
    resource_tracker: ResourceTracker,
) -> None:
    net = integration_client.network.create(
        network="198.51.100.0/29",
        description="integration test create",
        fetch_after_create=True,
    )
    assert net is not None
    resource_tracker.add(lambda: integration_client.network.delete(net))
    assert net.network == "198.51.100.0/29"


def test_get_by_cidr(
    integration_client: MregClient,
    test_network: str,
) -> None:
    result = integration_client.network.get(test_network)
    assert result is not None
    assert result.network == test_network


def test_get_by_id(integration_client: MregClient) -> None:
    net = integration_client.network.create(
        network="198.51.100.8/29",
        description="integration test get by id",
        fetch_after_create=True,
    )
    assert net is not None
    try:
        result = integration_client.network.get(net.id)
        assert result is not None
        assert result.id == net.id
    finally:
        integration_client.network.delete(net)


def test_get_by_object(
    integration_client: MregClient,
    test_network: str,
) -> None:
    net = integration_client.network.get(test_network)
    assert net is not None
    result = integration_client.network.get(net)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == net.id


def test_get_nonexistent_cidr_returns_none(integration_client: MregClient) -> None:
    result = integration_client.network.get("192.0.2.0/30", required=False)
    assert result is None


def test_get_nonexistent_id_returns_none(integration_client: MregClient) -> None:
    result = integration_client.network.get(99999999, required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.network.get("192.0.2.0/30")


def test_get_by_ip(
    integration_client: MregClient,
    resource_tracker: ResourceTracker,
) -> None:
    net = integration_client.network.create(
        network="198.51.100.16/29",
        description="integration test get_by_ip",
        fetch_after_create=True,
    )
    assert net is not None
    resource_tracker.add(lambda: integration_client.network.delete(net))
    result = integration_client.network.get_by_ip("198.51.100.17")
    assert result is not None
    assert result.network == "198.51.100.16/29"


def test_delete_by_id(integration_client: MregClient) -> None:
    net = integration_client.network.create(
        network="198.51.100.24/29",
        description="integration test delete by id",
        fetch_after_create=True,
    )
    assert net is not None
    integration_client.network.delete(net.id)
    assert integration_client.network.get("198.51.100.24/29", required=False) is None


def test_delete_by_object(integration_client: MregClient) -> None:
    net = integration_client.network.create(
        network="198.51.100.32/29",
        description="integration test delete by object",
        fetch_after_create=True,
    )
    assert net is not None
    integration_client.network.delete(net)
    assert integration_client.network.get("198.51.100.32/29", required=False) is None


def test_list(
    integration_client: MregClient,
    resource_tracker: ResourceTracker,
) -> None:
    net = integration_client.network.create(
        network="198.51.100.40/29",
        description="integration test list",
        fetch_after_create=True,
    )
    assert net is not None
    resource_tracker.add(lambda: integration_client.network.delete(net))
    results = integration_client.network.list()
    assert net.id in {r.id for r in results}


def test_count(integration_client: MregClient) -> None:
    result = integration_client.network.count()
    assert isinstance(result, int)
    assert result >= 0


def test_first(integration_client: MregClient) -> None:
    result = integration_client.network.first(required=False)
    assert result is None or isinstance(result, Network)


def test_ensure_absent_nonexistent(integration_client: MregClient) -> None:
    integration_client.network.ensure_absent("192.0.2.0/30")


def test_ensure_absent_existing(
    integration_client: MregClient,
    test_network: str,
) -> None:
    with pytest.raises(EntityAlreadyExists):
        integration_client.network.ensure_absent(test_network)


def test_get_first_available_ip(
    integration_client: MregClient,
    test_network: str,
) -> None:
    result = integration_client.network.get_first_available_ip(test_network)
    assert isinstance(result, (str, IPv4Address, IPv6Address))


def test_get_used_count(
    integration_client: MregClient,
    test_network: str,
) -> None:
    result = integration_client.network.get_used_count(test_network)
    assert isinstance(result, int)
    assert result >= 0


def test_get_unused_count(
    integration_client: MregClient,
    test_network: str,
) -> None:
    result = integration_client.network.get_unused_count(test_network)
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# CommunityManager
# ---------------------------------------------------------------------------


def test_community_create(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}comm"
    result = integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test community create",
    )
    assert isinstance(result, bool)
    assert result
    comm = integration_client.network.community.get_by_name(name, test_network, required=False)
    try:
        assert comm is not None
    finally:
        if comm is not None:
            integration_client.network.community.delete(comm.id, test_network)


def test_community_get_by_name(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}commgbn"
    integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test get_by_name",
    )
    comm = integration_client.network.community.get_by_name(name, test_network, required=False)
    try:
        assert comm is not None
        assert comm.name == name
    finally:
        if comm is not None:
            integration_client.network.community.delete(comm.id, test_network)


def test_community_get_by_id(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}commgbi"
    integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test get_by_id",
    )
    comm = integration_client.network.community.get_by_name(name, test_network)
    try:
        result = integration_client.network.community.get_by_id(comm.id, test_network)
        assert result is not None
        assert result.id == comm.id
    finally:
        integration_client.network.community.delete(comm.id, test_network)


def test_community_get_by_object(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}commgbo"
    integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test get by object",
    )
    comm = integration_client.network.community.get_by_name(name, test_network)
    try:
        result = integration_client.network.community.get(comm.id, test_network)
        assert result is not None
        assert result.id == comm.id
        assert isinstance(result, Community)
    finally:
        integration_client.network.community.delete(comm.id, test_network)


def test_community_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}commdid"
    integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test delete by id",
    )
    comm = integration_client.network.community.get_by_name(name, test_network)
    integration_client.network.community.delete(comm.id, test_network)
    assert integration_client.network.community.get_by_name(name, test_network, required=False) is None


def test_community_delete_by_name(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}commdname"
    integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test delete by name",
    )
    comm = integration_client.network.community.get_by_name(name, test_network)
    integration_client.network.community.delete(comm.name, test_network)
    assert integration_client.network.community.get_by_name(name, test_network, required=False) is None


def test_community_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    name = f"{test_prefix}commdobj"
    integration_client.network.community.create(
        test_network,
        name=name,
        description="integration test delete by object",
    )
    comm = integration_client.network.community.get_by_name(name, test_network)
    integration_client.network.community.delete(comm, test_network)
    assert integration_client.network.community.get_by_name(name, test_network, required=False) is None
