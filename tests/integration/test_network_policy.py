from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.models import NetworkPolicy
from mreg_api.models.models import NetworkPolicyAttribute

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def attr(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> NetworkPolicyAttribute:
    obj = integration_client.networkpolicyattribute.create(
        name=f"{test_prefix}attr",
        description="integration test attribute",
        fetch_after_create=True,
    )
    assert obj is not None
    resource_tracker.add(lambda: integration_client.networkpolicyattribute.delete(obj))
    return obj


@pytest.fixture(scope="module")
def policy(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> NetworkPolicy:
    obj = integration_client.networkpolicy.create(
        name=f"{test_prefix}pol",
        description="integration test policy",
        fetch_after_create=True,
    )
    assert obj is not None
    resource_tracker.add(lambda: integration_client.networkpolicy.delete(obj))
    return obj


# ---------------------------------------------------------------------------
# NetworkPolicyAttribute tests
# ---------------------------------------------------------------------------


def test_attr_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    obj = integration_client.networkpolicyattribute.create(
        name=f"{test_prefix}attrc",
        description="create test",
        fetch_after_create=True,
    )
    assert obj is not None
    resource_tracker.add(lambda: integration_client.networkpolicyattribute.delete(obj))
    assert obj.name == f"{test_prefix}attrc"


def test_attr_get_by_id(
    integration_client: MregClient,
    attr: NetworkPolicyAttribute,
) -> None:
    result = integration_client.networkpolicyattribute.get(attr.id)
    assert result is not None
    assert result.id == attr.id


def test_attr_get_by_name(
    integration_client: MregClient,
    attr: NetworkPolicyAttribute,
) -> None:
    result = integration_client.networkpolicyattribute.get_by_name(attr.name)
    assert result is not None
    assert result.id == attr.id


def test_attr_get_by_name_case_insensitive(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    mixed_name = f"{test_prefix}CiAttr"
    obj = integration_client.networkpolicyattribute.create(
        name=mixed_name,
        description="case insensitivity test",
        fetch_after_create=True,
    )
    assert obj is not None
    resource_tracker.add(lambda: integration_client.networkpolicyattribute.delete(obj))
    result = integration_client.networkpolicyattribute.get_by_name(mixed_name.upper())
    assert result is not None
    assert result.id == obj.id


def test_attr_get_by_object(
    integration_client: MregClient,
    attr: NetworkPolicyAttribute,
) -> None:
    result = integration_client.networkpolicyattribute.get(attr)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == attr.id


def test_attr_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.networkpolicyattribute.get(99999999, required=False)
    assert result is None


def test_attr_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.networkpolicyattribute.get(99999999)


def test_attr_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    obj = integration_client.networkpolicyattribute.create(
        name=f"{test_prefix}attrdid",
        description="delete by id test",
        fetch_after_create=True,
    )
    assert obj is not None
    integration_client.networkpolicyattribute.delete(obj.id)
    assert integration_client.networkpolicyattribute.get(obj.id, required=False) is None


def test_attr_delete_by_name(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    obj = integration_client.networkpolicyattribute.create(
        name=f"{test_prefix}attrdn",
        description="delete by name test",
        fetch_after_create=True,
    )
    assert obj is not None
    integration_client.networkpolicyattribute.delete(obj.name)
    assert integration_client.networkpolicyattribute.get(obj.id, required=False) is None


def test_attr_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    obj = integration_client.networkpolicyattribute.create(
        name=f"{test_prefix}attrdo",
        description="delete by object test",
        fetch_after_create=True,
    )
    assert obj is not None
    integration_client.networkpolicyattribute.delete(obj)
    assert integration_client.networkpolicyattribute.get(obj.id, required=False) is None


def test_attr_list(
    integration_client: MregClient,
    attr: NetworkPolicyAttribute,
) -> None:
    results = integration_client.networkpolicyattribute.list()
    assert attr.id in {r.id for r in results}


def test_attr_count(integration_client: MregClient) -> None:
    result = integration_client.networkpolicyattribute.count()
    assert isinstance(result, int)
    assert result >= 0


def test_attr_first(integration_client: MregClient) -> None:
    result = integration_client.networkpolicyattribute.first(required=False)
    assert result is None or isinstance(result, NetworkPolicyAttribute)


def test_attr_rename(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    old_name = f"{test_prefix}attrold"
    new_name = f"{test_prefix}attrnew"
    obj = integration_client.networkpolicyattribute.create(
        name=old_name,
        description="rename test",
        fetch_after_create=True,
    )
    assert obj is not None
    try:
        integration_client.networkpolicyattribute.rename(obj, new_name)
        assert integration_client.networkpolicyattribute.get_by_name(new_name, required=False) is not None
        assert integration_client.networkpolicyattribute.get_by_name(old_name, required=False) is None
    finally:
        for name in [new_name, old_name]:
            try:
                integration_client.networkpolicyattribute.delete(name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# NetworkPolicy tests
# ---------------------------------------------------------------------------


def test_policy_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    obj = integration_client.networkpolicy.create(
        name=f"{test_prefix}polc",
        description="create test",
        fetch_after_create=True,
    )
    assert obj is not None
    resource_tracker.add(lambda: integration_client.networkpolicy.delete(obj))
    assert obj.name == f"{test_prefix}polc"


def test_policy_get_by_id(
    integration_client: MregClient,
    policy: NetworkPolicy,
) -> None:
    result = integration_client.networkpolicy.get(policy.id)
    assert result is not None
    assert result.id == policy.id


def test_policy_get_by_name(
    integration_client: MregClient,
    policy: NetworkPolicy,
) -> None:
    result = integration_client.networkpolicy.get_by_name(policy.name)
    assert result is not None
    assert result.id == policy.id


def test_policy_get_by_object(
    integration_client: MregClient,
    policy: NetworkPolicy,
) -> None:
    result = integration_client.networkpolicy.get(policy)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == policy.id


def test_policy_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.networkpolicy.get(99999999, required=False)
    assert result is None


def test_policy_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.networkpolicy.get(99999999)


def test_policy_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    obj = integration_client.networkpolicy.create(
        name=f"{test_prefix}poldid",
        description="delete by id test",
        fetch_after_create=True,
    )
    assert obj is not None
    integration_client.networkpolicy.delete(obj.id)
    assert integration_client.networkpolicy.get(obj.id, required=False) is None


def test_policy_delete_by_name(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    obj = integration_client.networkpolicy.create(
        name=f"{test_prefix}poldn",
        description="delete by name test",
        fetch_after_create=True,
    )
    assert obj is not None
    integration_client.networkpolicy.delete(obj.name)
    assert integration_client.networkpolicy.get(obj.id, required=False) is None


def test_policy_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    obj = integration_client.networkpolicy.create(
        name=f"{test_prefix}poldo",
        description="delete by object test",
        fetch_after_create=True,
    )
    assert obj is not None
    integration_client.networkpolicy.delete(obj)
    assert integration_client.networkpolicy.get(obj.id, required=False) is None


def test_policy_list(
    integration_client: MregClient,
    policy: NetworkPolicy,
) -> None:
    results = integration_client.networkpolicy.list()
    assert policy.id in {r.id for r in results}


def test_policy_count(integration_client: MregClient) -> None:
    result = integration_client.networkpolicy.count()
    assert isinstance(result, int)
    assert result >= 0


def test_policy_first(integration_client: MregClient) -> None:
    result = integration_client.networkpolicy.first(required=False)
    assert result is None or isinstance(result, NetworkPolicy)


def test_policy_rename(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    old_name = f"{test_prefix}polold"
    new_name = f"{test_prefix}polnew"
    obj = integration_client.networkpolicy.create(
        name=old_name,
        description="rename test",
        fetch_after_create=True,
    )
    assert obj is not None
    try:
        integration_client.networkpolicy.rename(obj, new_name)
        assert integration_client.networkpolicy.get_by_name(new_name, required=False) is not None
        assert integration_client.networkpolicy.get_by_name(old_name, required=False) is None
    finally:
        for name in [new_name, old_name]:
            try:
                integration_client.networkpolicy.delete(name)
            except Exception:
                pass


def test_policy_add_remove_attribute(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    pol = integration_client.networkpolicy.create(
        name=f"{test_prefix}polattr",
        description="add remove attribute test",
        fetch_after_create=True,
    )
    assert pol is not None
    resource_tracker.add(lambda: integration_client.networkpolicy.delete(pol))

    test_attr = integration_client.networkpolicyattribute.create(
        name=f"{test_prefix}paattr",
        description="policy attribute test",
        fetch_after_create=True,
    )
    assert test_attr is not None
    resource_tracker.add(lambda: integration_client.networkpolicyattribute.delete(test_attr))

    updated = integration_client.networkpolicy.add_attribute(pol, test_attr, True)
    assert any(a.name == test_attr.name for a in updated.attributes)

    removed = integration_client.networkpolicy.remove_attribute(updated, test_attr)
    assert not any(a.name == test_attr.name for a in removed.attributes)
