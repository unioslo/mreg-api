from __future__ import annotations

from ipaddress import IPv4Address
from ipaddress import IPv6Address
from typing import TYPE_CHECKING

import pytest
from inline_snapshot import snapshot

from mreg_api.client import MregClient
from mreg_api.exceptions import ConflictError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
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
    )
    assert net is not None
    integration_client.network.delete(net.id)
    assert integration_client.network.get("198.51.100.24/29", required=False) is None


def test_delete_by_object(integration_client: MregClient) -> None:
    net = integration_client.network.create(
        network="198.51.100.32/29",
        description="integration test delete by object",
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


def test_assert_absent_nonexistent(integration_client: MregClient) -> None:
    integration_client.network.assert_absent("192.0.2.0/30")


def test_assert_absent_existing(
    integration_client: MregClient,
    test_network: str,
) -> None:
    with pytest.raises(EntityAlreadyExists):
        integration_client.network.assert_absent(test_network)


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


def test_update_network_range_overlaps_existing(integration_client: MregClient) -> None:
    net1 = "172.16.0.0/24"
    net2 = "172.16.1.0/24"
    try:
        net1 = integration_client.network.create(
            net1,
            description="integration test range overlap resize net1",
        )
        net2 = integration_client.network.create(
            net2,
            description="integration test range overlap resize net2",
        )
        with pytest.raises(ConflictError) as excinfo:
            integration_client.network.update(net1, network="172.16.0.0/23")
        msg = excinfo.value.formatted_message().replace(integration_client.url, "<URL>")
        assert msg == snapshot("""\
409 Conflict: PATCH <URL>/api/v1/networks/172.16.0.0/24
1 error:
  Network overlaps with: 172.16.1.0/24  (conflict)\
""")
    finally:
        integration_client.network.delete(net1)
        integration_client.network.delete(net2)


# ---------------------------------------------------------------------------
# CommunityManager
# ---------------------------------------------------------------------------


def test_community_create(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    communities = integration_client.network.community
    comm = communities.create(f"{test_prefix}comm", "integration test create", network=test_network)
    try:
        assert comm.network_cidr == test_network
    finally:
        communities.delete(comm)


def test_community_create_normalizes_name(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    """Ensure server normalizes community names to lowercase on creation and retrieval."""
    communities = integration_client.network.community
    comm = communities.create(f"{test_prefix}COMMUPPER", "upper", network=test_network)
    try:
        assert comm.name == f"{test_prefix}commupper".lower()
        assert communities.get(f"{test_prefix}CommUpper", network=test_network) == comm
    finally:
        communities.delete(comm)


def test_community_get(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    communities = integration_client.network.community
    name = f"{test_prefix}commget"
    created = communities.create(name, "integration test get", network=test_network)
    try:
        assert communities.get(created.id, network=test_network) == created
        assert communities.get(name, network=test_network) == created
    finally:
        communities.delete(created)


@pytest.mark.parametrize("ref", ["zzz-no-such-community-xyzzy", 999999])
def test_community_get_nonexistent_raises(
    integration_client: MregClient,
    test_network: str,
    ref: str | int,
) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.network.community.get(ref, network=test_network)


@pytest.mark.parametrize("ref", ["zzz-no-such-community-xyzzy", 999999])
def test_community_get_nonexistent_returns_none(
    integration_client: MregClient,
    test_network: str,
    ref: str | int,
) -> None:
    assert integration_client.network.community.get(ref, network=test_network, required=False) is None


def test_community_list_and_first(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    communities = integration_client.network.community
    name = f"{test_prefix}commlist"
    comm = communities.create(name, "integration test list", network=test_network)
    try:
        assert comm in communities.list(network=test_network)
        assert communities.first(network=test_network, name=name) == comm
    finally:
        communities.delete(comm)


def test_community_object_operations(
    integration_client: MregClient,
    test_prefix: str,
    test_network: str,
) -> None:
    """Operations on a Community object take the network from `network_cidr`."""
    communities = integration_client.network.community
    comm = communities.create(f"{test_prefix}commobj", "before", network=test_network)
    try:
        assert communities.get_hosts(comm) == []
        updated = communities.update(comm, description="after")
        assert updated.description == "after"
        assert communities.refresh(comm).description == "after"
    finally:
        communities.delete(comm)
    assert communities.get(comm.id, network=test_network, required=False) is None
