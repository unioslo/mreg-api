from __future__ import annotations

from mreg_api.client import MregApiClient


def test_client_singleton() -> None:
    client1 = MregApiClient(url="http://example.com", domain="example.com", timeout=30)
    client2 = MregApiClient()
    assert client1 is client2


def test_client_reset_instance() -> None:
    client1 = MregApiClient()
    MregApiClient.reset_instance()
    client2 = MregApiClient()
    assert client1 is not client2
