from __future__ import annotations

import datetime

import pytest

from mreg_api.events import ObjectRef
from mreg_api.models import CNAME
from mreg_api.models import MX
from mreg_api.models import HInfo
from mreg_api.models import Host
from mreg_api.models import Network
from mreg_api.models.abstracts import APIMixin
from mreg_api.models.fields import HostName


@pytest.mark.parametrize(
    "obj,id_",
    [
        pytest.param(
            Host(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                id=123,
                name=HostName("example.com"),
                comment="Test host",
                ipaddresses=[],
            ),
            "example.com",
            id="Host",
        ),
        pytest.param(
            Network(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                id=1,
                network="192.168.0.0/24",
                excluded_ranges=[],
                description="Test network",
                dns_delegated=False,
                category="Test category",
                location="Test location",
                frozen=False,
                reserved=0,
            ),
            "192.168.0.0/24",
            id="Network",
        ),
        pytest.param(
            HInfo(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                host=123,
                cpu="x86_64",
                os="Linux",
            ),
            "123",
            id="HInfo",
        ),
        pytest.param(
            CNAME(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                host=123,
                id=1,
                name=HostName("alias.example.com"),
            ),
            "alias.example.com",
            id="CNAME",
        ),
        pytest.param(
            MX(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                host=123,
                id=1,
                priority=10,
                mx="mail.example.com",
            ),
            "1",
            id="MX",
        ),
    ],
)
def test_objectref_new(obj: APIMixin, id_: str):
    ref = ObjectRef.new(obj)
    assert ref.type == obj.__class__.__name__

    # The inferred ID field is correct and the value matches the object's ID
    assert ref.id == str(getattr(obj, obj.endpoint().external_id_field()))
    assert ref.id == id_
