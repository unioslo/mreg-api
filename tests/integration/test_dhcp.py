from __future__ import annotations

import pytest

from mreg_api.client import MregClient
from mreg_api.models.models import DhcpHostIPv4
from mreg_api.models.models import DhcpHostIPv6
from mreg_api.models.models import DhcpHostIPv6ByIPv4

pytestmark = [pytest.mark.integration, pytest.mark.readonly]


def test_list_ipv4(integration_client: MregClient) -> None:
    results = integration_client.dhcphostipv4.list()
    assert isinstance(results, list)


def test_count_ipv4(integration_client: MregClient) -> None:
    result = integration_client.dhcphostipv4.count(strict=False)
    assert isinstance(result, int)
    assert result >= 0


def test_first_ipv4(integration_client: MregClient) -> None:
    result = integration_client.dhcphostipv4.first(required=False)
    assert result is None or isinstance(result, DhcpHostIPv4)


def test_list_by_range_ipv4(integration_client: MregClient) -> None:
    results = integration_client.dhcphostipv4.list_by_range("10.0.0.0", "8")
    assert isinstance(results, list)


def test_list_ipv6(integration_client: MregClient) -> None:
    results = integration_client.dhcphostipv6.list()
    assert isinstance(results, list)


def test_count_ipv6(integration_client: MregClient) -> None:
    result = integration_client.dhcphostipv6.count(strict=False)
    assert isinstance(result, int)
    assert result >= 0


def test_first_ipv6(integration_client: MregClient) -> None:
    result = integration_client.dhcphostipv6.first(required=False)
    assert result is None or isinstance(result, DhcpHostIPv6)


def test_list_by_range_ipv6(integration_client: MregClient) -> None:
    results = integration_client.dhcphostipv6.list_by_range("2001:db8::", "32")
    assert isinstance(results, list)


def test_list_ipv6_by_ipv4(integration_client: MregClient) -> None:
    results = integration_client.dhcphostipv6byipv4.list()
    assert isinstance(results, list)


def test_count_ipv6_by_ipv4(integration_client: MregClient) -> None:
    result = integration_client.dhcphostipv6byipv4.count(strict=False)
    assert isinstance(result, int)
    assert result >= 0


def test_first_ipv6_by_ipv4(integration_client: MregClient) -> None:
    result = integration_client.dhcphostipv6byipv4.first(required=False)
    assert result is None or isinstance(result, DhcpHostIPv6ByIPv4)
