from __future__ import annotations

from mreg_api.client import MregClient


def test_client_singleton() -> None:
    client1 = MregClient(url="http://example.com", domain="example.com", timeout=30)
    client2 = MregClient()
    assert client1 is client2


def test_client_reset_instance() -> None:
    client1 = MregClient()
    MregClient.reset_instance()
    client2 = MregClient()
    assert client1 is not client2
