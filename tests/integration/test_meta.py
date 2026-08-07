from __future__ import annotations

import pytest

from mreg_api.client import MregClient
from mreg_api.models.models import HealthInfo
from mreg_api.models.models import HeartbeatHealth
from mreg_api.models.models import LDAPHealth
from mreg_api.models.models import ServerLibraries
from mreg_api.models.models import ServerVersion
from mreg_api.models.models import UserInfo

pytestmark = [pytest.mark.integration, pytest.mark.readonly]


def test_version(integration_client: MregClient) -> None:
    result = integration_client.meta.version.get()
    assert isinstance(result, ServerVersion)
    assert isinstance(result.version, str)


def test_libraries(integration_client: MregClient) -> None:
    result = integration_client.meta.libraries.get()
    assert isinstance(result, ServerLibraries)


def test_userinfo(integration_client: MregClient) -> None:
    result = integration_client.meta.userinfo.get()
    assert isinstance(result, UserInfo)


def test_userinfo_for_user(integration_client: MregClient) -> None:
    result = integration_client.meta.userinfo.get(user="test")
    assert isinstance(result, UserInfo)


def test_ldap_health(integration_client: MregClient) -> None:
    result = integration_client.meta.ldap.get()
    assert isinstance(result, LDAPHealth)


def test_heartbeat(integration_client: MregClient) -> None:
    result = integration_client.meta.heartbeat.get()
    assert isinstance(result, HeartbeatHealth)


def test_health(integration_client: MregClient) -> None:
    result = integration_client.meta.health.get()
    assert isinstance(result, HealthInfo)
