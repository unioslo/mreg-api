from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

import pytest

from mreg_api.client import MregClient
from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import GetError
from mreg_api.models.models import Atom

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


@pytest.fixture
def client(integration_client: MregClient) -> MregClient:
    return integration_client


@pytest.fixture(scope="module")
def atom(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> Atom:
    atm = integration_client.atom.create(
        name=f"{test_prefix}atm",
        description="integration test atom",
        fetch_after_create=True,
    )
    assert atm is not None
    resource_tracker.add(lambda: integration_client.atom.delete(atm))
    return atm


def test_create(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    atm = integration_client.atom.create(
        name=f"{test_prefix}ac",
        description="create test",
        fetch_after_create=True,
    )
    assert atm is not None
    resource_tracker.add(lambda: integration_client.atom.delete(atm))
    assert atm.name == f"{test_prefix}ac"


@pytest.mark.parametrize(
    "name",
    [
        "name_with_underscores",
        "name-with-dashes",
        "name.with.dots",
        "name+with+pluses",
        "name9with9numbers",
        pytest.param(
            # NOTE: it is actually upon receiving the `Location` header
            # that we get a 400 Bad Request, not on the POST itself,
            # hence the `GetError` instead of `PostError` here.
            # The server doesn't quote the name in the Location header,
            # and neither does the client. So this is just a disaster all around.
            "name/with/slashes",
            marks=pytest.mark.xfail(raises=GetError),
        ),
    ],
)
def test_create_various_names(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    name: str,
) -> None:
    """Test atom creation with various valid name formats."""
    atom_name = f"{test_prefix}{name}"
    atm = integration_client.atom.create(
        name=atom_name,
        description="create test",
        fetch_after_create=True,
    )
    assert atm is not None
    resource_tracker.add(lambda: integration_client.atom.delete(atm))
    assert atm.name == atom_name
    # Check that we can fetch these atoms by name as well
    assert integration_client.atom.get_by_name(atom_name)


def test_create_name_with_slashes(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    """Test atom creation with a name containing slashes.

    This is a case where it is possible to create the Atom, but
    automatic retrieval fails, and the API library does not quote the name
    in `get()`, so it just becomes almost impossible to retrieve unless
    the name is manually quoted when passing the argument to `get()`.

    `Endpoint.with_id` does not quote `/` because it is a valid character
    in a path parameter for networks (i.e. `/networks/10.0.0.0/24`), so MREG
    clients have historically never quoted `/` in path parameters.
    """
    #
    atom_name_1 = f"{test_prefix}name/with/slashes1"
    atm = integration_client.atom.create(
        name=atom_name_1,
        description="create test",
        # We MUST not fetch after create, because the server returns
        # an invalid Location header for this atom.
        fetch_after_create=False,
    )
    assert atm is None  # nothing returned

    # Trying to fetch it (with or without quoting) will fail
    with pytest.raises(EntityNotFound):
        integration_client.atom.get_by_name(atom_name_1)
    with pytest.raises(EntityNotFound):
        integration_client.atom.get_by_name(quote(atom_name_1, safe=""))


def test_create_name_with_slashes_location_header_invalid(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    """Test atom creation with a name containing slashes, using MregClient directly.

    Verifies that the Location header is invalid, and that the atom
    is still created, even though it cannot be fetched via conventional means.
    """
    # Creating an atom with fetch_after_create=True will fail,
    # but the atom is still created!

    atom_name_2 = f"{test_prefix}name/with/slashes2"
    assert integration_client.atom.get_by_name(atom_name_2, required=False) is None

    resp = integration_client.post(
        Endpoint.HostPolicyAtoms,
        json={"name": atom_name_2, "description": "create test"},
    )

    assert resp.status_code == 201

    # Location header contains URL with slashes in atom name, which is invalid.
    assert resp.headers["Location"] == f"{Endpoint.HostPolicyAtoms}{atom_name_2}"

    # We can't fetch atom directly, but we can still list it via querying all atoms.
    # We should still be able to see it if we fetch _all_ atoms
    all_atoms = integration_client.atom.list()
    atom = next((a for a in all_atoms if a.name == atom_name_2), None)
    assert atom is not None

    # Try to fetch our atom by ID
    fetched = integration_client.atom.get(atom.id)
    assert fetched is not None
    assert fetched.name == atom_name_2


def test_get_by_id(integration_client: MregClient, atom: Atom) -> None:
    result = integration_client.atom.get(atom.id)
    assert result is not None
    assert result.id == atom.id


def test_get_by_name(integration_client: MregClient, atom: Atom) -> None:
    result = integration_client.atom.get(atom.name)
    assert result is not None
    assert result.id == atom.id


def test_get_by_object(integration_client: MregClient, atom: Atom) -> None:
    result = integration_client.atom.get(atom)  # type: ignore[arg-type]
    assert result is not None
    assert result.id == atom.id


def test_get_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.atom.get(99999999, required=False)
    assert result is None


def test_get_nonexistent_raises(integration_client: MregClient) -> None:
    with pytest.raises(EntityNotFound):
        integration_client.atom.get(99999999)


def test_get_by_name_nonexistent_returns_none(integration_client: MregClient) -> None:
    result = integration_client.atom.get_by_name("nope", required=False)
    assert result is None


def test_delete_by_id(integration_client: MregClient, test_prefix: str) -> None:
    atm = integration_client.atom.create(
        name=f"{test_prefix}adid",
        description="delete by id test",
        fetch_after_create=True,
    )
    assert atm is not None
    integration_client.atom.delete(atm.id)
    assert integration_client.atom.get(atm.id, required=False) is None


def test_delete_by_name(integration_client: MregClient, test_prefix: str) -> None:
    atm = integration_client.atom.create(
        name=f"{test_prefix}adn",
        description="delete by name test",
        fetch_after_create=True,
    )
    assert atm is not None
    integration_client.atom.delete(atm.name)
    assert integration_client.atom.get(atm.id, required=False) is None


def test_delete_by_object(integration_client: MregClient, test_prefix: str) -> None:
    atm = integration_client.atom.create(
        name=f"{test_prefix}ado",
        description="delete by object test",
        fetch_after_create=True,
    )
    assert atm is not None
    integration_client.atom.delete(atm)
    assert integration_client.atom.get(atm.id, required=False) is None


def test_set_description(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    atm = integration_client.atom.create(
        name=f"{test_prefix}asd",
        description="original description",
        fetch_after_create=True,
    )
    assert atm is not None
    resource_tracker.add(lambda: integration_client.atom.delete(atm))
    integration_client.atom.set_description(atm, "new desc")
    updated = integration_client.atom.get(atm.name)
    assert updated is not None
    assert updated.description == "new desc"


def test_list(integration_client: MregClient, atom: Atom) -> None:
    results = integration_client.atom.list()
    assert atom.id in {r.id for r in results}


def test_list_by_name_regex(
    integration_client: MregClient,
    atom: Atom,
    test_prefix: str,
) -> None:
    results = integration_client.atom.list_by_name_regex(test_prefix)
    assert atom in results


def test_count(integration_client: MregClient) -> None:
    result = integration_client.atom.count()
    assert isinstance(result, int)
    assert result >= 0


def test_first(integration_client: MregClient) -> None:
    result = integration_client.atom.first()
    assert result is None or isinstance(result, Atom)


def test_rename(
    integration_client: MregClient,
    test_prefix: str,
) -> None:
    old_name = f"{test_prefix}aold"
    new_name = f"{test_prefix}anew"
    atm = integration_client.atom.create(
        name=old_name,
        description="rename test",
        fetch_after_create=True,
    )
    assert atm is not None
    try:
        integration_client.atom.rename(atm, new_name)
        assert integration_client.atom.get_by_name(new_name, required=False) is not None
        assert integration_client.atom.get_by_name(old_name, required=False) is None
    finally:
        for name in [new_name, old_name]:
            try:
                integration_client.atom.delete(name)
            except Exception:
                pass


def test_assert_absent_nonexistent(integration_client: MregClient) -> None:
    integration_client.atom.assert_absent(99999999)


def test_assert_absent_existing(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
) -> None:
    atm = integration_client.atom.create(
        name=f"{test_prefix}aea",
        description="ensure absent test",
        fetch_after_create=True,
    )
    assert atm is not None
    resource_tracker.add(lambda: integration_client.atom.delete(atm))
    with pytest.raises(EntityAlreadyExists):
        integration_client.atom.assert_absent(atm.name)
