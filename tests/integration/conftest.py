"""Session fixtures and activation hook for integration tests.

Integration tests are skipped by default. To run them, set MREG_URL or pass
--mreg-url to pytest:

    MREG_URL=http://127.0.0.1:8000 MREG_USERNAME=test MREG_PASSWORD=test uv run pytest tests/integration/
    uv run pytest tests/integration/ --mreg-url http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Generator
from contextlib import suppress
from datetime import datetime
from typing import Any

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import PostError
from mreg_api.models import Zone


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--mreg-url", default=None, help="mreg server URL (overrides MREG_URL env var)")
    parser.addoption("--mreg-username", default=None, help="mreg username (overrides MREG_USERNAME)")
    parser.addoption("--mreg-password", default=None, help="mreg password (overrides MREG_PASSWORD)")
    parser.addoption("--mreg-domain", default=None, help="mreg domain (overrides MREG_DOMAIN)")
    parser.addoption("--test-network", default=None, help="test network CIDR (overrides MREG_TEST_NETWORK)")
    parser.addoption("--test-ip", default=None, help="test IP address (overrides MREG_TEST_IP)")
    parser.addoption(
        "--mreg-cache", default=None, help="enable or disable mreg client cache (overrides MREG_CACHE)"
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "integration" in item.keywords:
        url = os.environ.get("MREG_URL") or item.config.getoption("--mreg-url", default=None)
        if not url:
            pytest.skip("set MREG_URL env var or --mreg-url to run integration tests")


# --- Connection config fixtures ---


@pytest.fixture(scope="session")
def mreg_url(request: pytest.FixtureRequest) -> str:
    return (
        os.environ.get("MREG_URL")
        or request.config.getoption("--mreg-url", default=None)
        or "http://127.0.0.1:8000"
    )


@pytest.fixture(scope="session")
def mreg_username(request: pytest.FixtureRequest) -> str:
    return (
        os.environ.get("MREG_USERNAME") or request.config.getoption("--mreg-username", default=None) or "test"
    )


@pytest.fixture(scope="session")
def mreg_password(request: pytest.FixtureRequest) -> str:
    return (
        os.environ.get("MREG_PASSWORD") or request.config.getoption("--mreg-password", default=None) or "test"
    )


@pytest.fixture(scope="session")
def mreg_domain(request: pytest.FixtureRequest) -> str:
    return (
        os.environ.get("MREG_DOMAIN")
        or request.config.getoption("--mreg-domain", default=None)
        or "example.com"
    )


@pytest.fixture(scope="session")
def mreg_cache(request: pytest.FixtureRequest) -> bool:
    val = os.environ.get("MREG_CACHE") or request.config.getoption("--mreg-cache", default=None) or "false"
    return str(val).lower() in ("1", "true", "yes")


@pytest.fixture(scope="session")
def test_network(request: pytest.FixtureRequest) -> str:
    return (
        os.environ.get("MREG_TEST_NETWORK")
        or request.config.getoption("--test-network", default=None)
        or "10.0.0.0/8"
    )


@pytest.fixture(scope="session")
def test_ip(request: pytest.FixtureRequest) -> str:
    return os.environ.get("MREG_TEST_IP") or request.config.getoption("--test-ip", default=None) or "10.0.0.1"


# --- Core fixtures ---


@pytest.fixture(scope="session")
def integration_client(
    mreg_url: str,
    mreg_username: str,
    mreg_password: str,
    mreg_domain: str,
    mreg_cache: bool,
) -> MregClient:
    client = MregClient(url=mreg_url, domain=mreg_domain, cache=mreg_cache)
    client.login(username=mreg_username, password=mreg_password)
    return client


@pytest.fixture(scope="session")
def test_prefix() -> str:
    """Unique prefix for all resources created in this test session."""
    return f"mrgtest{datetime.now().strftime('%H%M%S')}"


class ResourceTracker:
    """Ordered registry of zero-arg cleanup callables."""

    def __init__(self) -> None:  # noqa: D107
        self._cleanups: list[Callable[[], Any]] = []

    def add(self, cleanup: Callable[[], Any]) -> None:
        """Register a zero-arg callable to run at teardown."""
        self._cleanups.append(cleanup)

    def _run_cleanups(self) -> None:
        for cleanup in reversed(self._cleanups):
            with suppress(Exception):
                cleanup()


@pytest.fixture(scope="session")
def resource_tracker() -> Generator[ResourceTracker, None, None]:
    """Session-scoped cleanup registry.

    Register cleanup for each created resource:
        resource_tracker.add(lambda: client.label.delete(label))

    Runs all cleanups in reverse order on session teardown, swallowing
    individual exceptions so one failure doesn't block the rest.
    """
    tracker = ResourceTracker()
    yield tracker
    tracker._run_cleanups()


def _create_zone(
    client: MregClient, name: str, resource_tracker: ResourceTracker, primary_ns: list[str] | None = None
) -> Zone:
    """Create a zone if it doesn't exist, and register cleanup."""
    if not primary_ns:
        primary_ns = [f"ns1.{client.domain}"]

    if (zone := client.zone.get(name, required=False)) is None:
        zone = client.zone.create(
            name=name,
            email=f"hostmaster@{name}",
            primary_ns=primary_ns,
            force=True,
        )

    if not zone:
        pytest.exit(f"Failed to create or fetch zone {name}; aborting integration tests")
    resource_tracker.add(lambda: client.zone.delete(name, force=True))

    return zone


@pytest.fixture(scope="session")
def main_zone(
    integration_client: MregClient, resource_tracker: ResourceTracker
) -> Generator[Zone, None, None]:
    """Create the main zone (based on client's configured domain via MREG_DOMAIN).

    Aborts integration tests if the integration client has no configured domain.
    """
    if not integration_client.domain:
        pytest.exit("MregClient has no domain configured; aborting integration tests")
    yield _create_zone(integration_client, integration_client.domain, resource_tracker)


@pytest.fixture(scope="module")
def zone(
    integration_client: MregClient, main_zone: Zone, resource_tracker: ResourceTracker
) -> Generator[Zone, None, None]:
    """Create test zone before the module runs and tear it down after."""
    yield _create_zone(integration_client, f"subzone.{main_zone.name}", resource_tracker)


@pytest.fixture(scope="session", autouse=True)
def seed_test_network(
    request: pytest.FixtureRequest,
    test_network: str,
    resource_tracker: ResourceTracker,
) -> Generator[None, None, None]:
    """Create the shared test network before integration tests run.

    No-op when no mreg server URL is configured (i.e., when integration tests
    are being skipped). Creates the network only if it doesn't already exist.
    """
    url = os.environ.get("MREG_URL") or request.config.getoption("--mreg-url", default=None)
    if not url:
        yield
        return

    client: MregClient = request.getfixturevalue("integration_client")
    assert isinstance(client, MregClient)

    existing = client.network.get(test_network, required=False)
    if existing is None:
        try:
            net = client.network.create(
                network=test_network,
                description="Integration test seed network",
            )
            if net is not None:
                resource_tracker.add(lambda: client.network.delete(net))
        except PostError:
            pass  # overlaps with existing subnets from a previous run; acceptable
    yield
