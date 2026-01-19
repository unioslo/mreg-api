from __future__ import annotations

import datetime

from inline_snapshot import snapshot

from mreg_api.models import Host
from mreg_api.models.fields import HostName
from mreg_api.models.models import ContactEmail


def test_host_contacts_compatibility() -> None:
    host = Host(
        id=1,
        name=HostName("test-host"),
        ipaddresses=[],
        created_at=datetime.datetime(2020, 1, 1, 0, 0, 0),
        updated_at=datetime.datetime(2021, 1, 1, 0, 0, 0),
        contact="test-contact@example.com",
        contacts=[],
        comment="",
    )

    # Contains the constructed ContactEmail object
    assert host.contacts == snapshot(
        [
            ContactEmail(
                created_at=datetime.datetime(1970, 1, 1, 1, 0),
                updated_at=datetime.datetime(1970, 1, 1, 1, 0),
                id=0,
                email="test-contact@example.com",
            )
        ]
    )

    # Property works as expected
    assert host.contact_emails == ["test-contact@example.com"]


def test_host_contacts_compatibility_does_not_override_contacts() -> None:
    host = Host(
        id=1,
        name=HostName("test-host"),
        ipaddresses=[],
        created_at=datetime.datetime(2020, 1, 1, 0, 0, 0),
        updated_at=datetime.datetime(2021, 1, 1, 0, 0, 0),
        contact="test-contact@example.com",
        contacts=[
            ContactEmail(
                created_at=datetime.datetime(2022, 1, 1, 0, 0, 0),
                updated_at=datetime.datetime(2022, 1, 1, 0, 0, 0),
                id=123,
                email="actual-contact@example.com",
            )
        ],
        comment="",
    )

    assert host.contacts == snapshot(
        [
            ContactEmail(
                created_at=datetime.datetime(2022, 1, 1, 0, 0),
                updated_at=datetime.datetime(2022, 1, 1, 0, 0),
                id=123,
                email="actual-contact@example.com",
            )
        ]
    )

    # Property only contains contact from `contacts` field
    assert host.contact_emails == ["actual-contact@example.com"]
