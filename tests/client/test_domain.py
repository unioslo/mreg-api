from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from mreg_api.client import MregClient
from mreg_api.models.fields import HostName
from mreg_api.models.fields import hostname_domain


def test_client_set_domain() -> None:
    """set_domain changes the hostname domain used for validation."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert hostname_domain.get() == "example.com"
    assert HostName.validate_hostname("myhost") == "myhost.example.com"

    _ = client.set_domain("other.org")
    assert client.get_domain() == "other.org"
    assert client.get_domain() == hostname_domain.get()
    assert HostName.validate_hostname("myhost") == "myhost.other.org"


def test_client_reset_domain() -> None:
    """reset_domain restores the hostname domain to the value from initialization."""
    assert hostname_domain.get() == snapshot("uio.no")  # default domain

    client = MregClient(url="http://example.com", domain="example.com")
    assert hostname_domain.get() == "example.com"

    # Set another domain
    _ = client.set_domain("other.org")
    assert hostname_domain.get() == "other.org"

    # Reset to original
    client.reset_domain()
    assert client.get_domain() == snapshot("example.com")

    # Can be called multiple times (also in destructor)
    client.reset_domain()
    client.reset_domain()
    assert client.get_domain() == snapshot("example.com")


def test_client_reset_domain_after_multiple_set_domain() -> None:
    """reset_domain always restores to the initialization value, not the previous value."""
    client = MregClient(url="http://example.com", domain="example.com")

    _ = client.set_domain("first.org")
    _ = client.set_domain("second.org")
    _ = client.set_domain("third.org")
    assert client.get_domain() == "third.org"
    assert client.get_domain() == hostname_domain.get()

    # reset_domain should go back to example.com, not third.org or second.org
    client.reset_domain()
    assert client.get_domain() == "example.com"
    assert client.get_domain() == hostname_domain.get()


def test_client_domain_override_context_manager() -> None:
    """domain_override temporarily changes the hostname domain within a context."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.get_domain() == "example.com"

    with client.domain_override("temp.net"):
        assert client.get_domain() == "temp.net"
        assert HostName.validate_hostname("myhost") == "myhost.temp.net"

    # After exiting context, domain should be restored
    assert client.get_domain() == "example.com"
    assert HostName.validate_hostname("myhost") == "myhost.example.com"


def test_client_domain_override_restores_on_exception() -> None:
    """domain_override restores the domain even when an exception occurs."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.get_domain() == "example.com"

    with pytest.raises(ValueError, match="test error"):
        with client.domain_override("temp.net"):
            assert client.get_domain() == "temp.net"
            raise ValueError("test error")

    # Domain should still be restored after exception
    assert client.get_domain() == "example.com"


def test_client_domain_override_nested() -> None:
    """Nested domain_override contexts work correctly."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.get_domain() == "example.com"

    with client.domain_override("outer.org"):
        assert client.get_domain() == "outer.org"

        with client.domain_override("inner.net"):
            assert client.get_domain() == "inner.net"

        # After inner context exits, should restore to outer value
        assert client.get_domain() == "outer.org"

    # After outer context exits, should restore to original
    assert client.get_domain() == "example.com"


def test_client_domain_override_after_set_domain() -> None:
    """domain_override works correctly after set_domain has been called."""
    client = MregClient(url="http://example.com", domain="example.com")

    _ = client.set_domain("changed.org")
    assert client.get_domain() == "changed.org"

    with client.domain_override("temp.net"):
        assert client.get_domain() == "temp.net"

    # After context exits, should restore to the value before the context (changed.org)
    assert client.get_domain() == "changed.org"

    # reset_domain should still restore to original initialization value
    client.reset_domain()
    assert client.get_domain() == "example.com"


