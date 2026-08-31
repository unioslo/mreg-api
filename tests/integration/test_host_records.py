from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mreg_api.client import MregClient
from mreg_api.models.models import Host
from mreg_api.models.models import Zone

if TYPE_CHECKING:
    from tests.integration.conftest import ResourceTracker

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> Host:
    name = f"{test_prefix}rech.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    return h


@pytest.fixture(scope="module")
def hinfo_host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> Host:
    name = f"{test_prefix}hinfo.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    return h


@pytest.fixture(scope="module")
def ptr_host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> Host:
    name = f"{test_prefix}ptr.{zone.name}"
    h = integration_client.host.create(name=name, ipaddress="10.0.1.50")
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    return h


@pytest.fixture(scope="module")
def bacnet_host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> Host:
    name = f"{test_prefix}bacnet.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    return h


@pytest.fixture(scope="module")
def loc_host(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> Host:
    name = f"{test_prefix}loc.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    return h


# ── HInfo tests ───────────────────────────────────────────────────────────────


def test_hinfo_create(
    integration_client: MregClient,
    hinfo_host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.hinfo.create(host=hinfo_host, cpu="x86_64", os="Linux")
    hi = integration_client.hinfo.get_by_host(hinfo_host)
    assert hi is not None
    resource_tracker.add(lambda: integration_client.hinfo.delete(hi))
    assert hi.cpu == "x86_64"
    assert hi.os == "Linux"
    assert hi.host == hinfo_host.id


def test_hinfo_get_by_host_id(
    integration_client: MregClient,
    hinfo_host: Host,
) -> None:
    result = integration_client.hinfo.get_by_host(hinfo_host.id)
    assert result is not None
    assert result.host == hinfo_host.id


def test_hinfo_get_by_host(
    integration_client: MregClient,
    hinfo_host: Host,
) -> None:
    result = integration_client.hinfo.get_by_host(hinfo_host)
    assert result is not None
    assert result.host == hinfo_host.id


def test_hinfo_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}hinfodid.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.hinfo.create(host=h, cpu="x86_64", os="Linux")
    fetched = integration_client.hinfo.get_by_host(h)
    assert fetched is not None
    integration_client.hinfo.delete(fetched)
    assert integration_client.hinfo.get_by_host(h, required=False) is None


def test_hinfo_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}hinfodo.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.hinfo.create(host=h, cpu="x86_64", os="Linux")
    hi = integration_client.hinfo.get_by_host(h)
    assert hi is not None
    integration_client.hinfo.delete(hi)
    assert integration_client.hinfo.get_by_host(h, required=False) is None


# ── TXT tests ─────────────────────────────────────────────────────────────────


def test_txt_create(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    txt_content = "v=spf1 -all"
    integration_client.txt.create(host=host, txt=txt_content)
    txts = integration_client.txt.list(host=host.id)
    txt = next(t for t in txts if t.txt == txt_content)
    assert txt is not None
    resource_tracker.add(lambda: integration_client.txt.delete(txt))
    assert txt.txt == txt_content
    assert txt.host == host.id


def test_txt_get_by_id(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    txt_content = "integration test txt"
    integration_client.txt.create(host=host, txt=txt_content)
    txts = integration_client.txt.list(host=host.id)
    txt = next(t for t in txts if t.txt == txt_content)
    assert txt is not None
    resource_tracker.add(lambda: integration_client.txt.delete(txt))
    result = integration_client.txt.get(txt.id)
    assert result is not None
    assert result.id == txt.id


def test_txt_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}txth.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    txt_content = "delete by id test"
    integration_client.txt.create(host=h, txt=txt_content)
    txts = integration_client.txt.list(host=h.id)
    txt = next(t for t in txts if t.txt == txt_content)
    integration_client.txt.delete(txt.id)
    assert integration_client.txt.get(txt.id, required=False) is None


def test_txt_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}txto.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    txt_content = "delete by object test"
    integration_client.txt.create(host=h, txt=txt_content)
    txts = integration_client.txt.list(host=h.id)
    txt = next(t for t in txts if t.txt == txt_content)
    integration_client.txt.delete(txt)
    assert integration_client.txt.get(txt.id, required=False) is None


# ── MX tests ──────────────────────────────────────────────────────────────────


@pytest.fixture(name="mx_host", scope="module")
def mx_host(zone: Zone) -> str:
    return f"mail.{zone.name}"


def test_mx_create(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
    mx_host: str,
) -> None:
    integration_client.mx.create(host=host, mx=mx_host, priority=10)
    mx = integration_client.mx.get_unique(host, mx_host, 10)
    assert mx is not None
    resource_tracker.add(lambda: integration_client.mx.delete(mx))
    assert mx.mx == mx_host
    assert mx.priority == 10
    assert mx.host == host.id


def test_mx_get_by_id(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
    mx_host: str,
) -> None:
    integration_client.mx.create(host=host, mx=mx_host, priority=20)
    mx = integration_client.mx.get_unique(host, mx_host, 20)
    assert mx is not None
    resource_tracker.add(lambda: integration_client.mx.delete(mx))
    result = integration_client.mx.get(mx.id)
    assert result is not None
    assert result.id == mx.id


def test_mx_get_unique(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
    mx_host: str,
) -> None:
    integration_client.mx.create(host=host, mx=mx_host, priority=30)
    mx = integration_client.mx.get_unique(host, mx_host, 30)
    assert mx is not None
    resource_tracker.add(lambda: integration_client.mx.delete(mx))
    result = integration_client.mx.get_unique(host, mx_host, 30)
    assert result is not None
    assert result.id == mx.id


def test_mx_delete_by_id(
    integration_client: MregClient,
    host: Host,
    mx_host: str,
) -> None:
    integration_client.mx.create(host=host, mx=mx_host, priority=40)
    mx = integration_client.mx.get_unique(host, mx_host, 40)
    assert mx is not None
    integration_client.mx.delete(mx.id)
    assert integration_client.mx.get(mx.id, required=False) is None


def test_mx_delete_by_object(
    integration_client: MregClient,
    host: Host,
    mx_host: str,
) -> None:
    integration_client.mx.create(host=host, mx=mx_host, priority=50)
    mx = integration_client.mx.get_unique(host, mx_host, 50)
    assert mx is not None
    integration_client.mx.delete(mx)
    assert integration_client.mx.get(mx.id, required=False) is None


# ── NAPTR tests ───────────────────────────────────────────────────────────────


def test_naptr_create(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.naptr.create(host=host, preference=10, order=10, replacement=".")
    naptr = integration_client.naptr.get_unique(host, preference=10, order=10, replacement=".")
    assert naptr is not None
    resource_tracker.add(lambda: integration_client.naptr.delete(naptr))
    assert naptr.preference == 10
    assert naptr.order == 10
    assert naptr.host == host.id


def test_naptr_get_by_id(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.naptr.create(host=host, preference=20, order=20, replacement=".")
    naptr = integration_client.naptr.get_unique(host, preference=20, order=20, replacement=".")
    assert naptr is not None
    resource_tracker.add(lambda: integration_client.naptr.delete(naptr))
    result = integration_client.naptr.get(naptr.id)
    assert result is not None
    assert result.id == naptr.id


def test_naptr_get_unique(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.naptr.create(host=host, preference=30, order=30, replacement=".")
    naptr = integration_client.naptr.get_unique(host, preference=30, order=30, replacement=".")
    assert naptr is not None
    resource_tracker.add(lambda: integration_client.naptr.delete(naptr))
    result = integration_client.naptr.get_unique(host, preference=30, order=30, replacement=".")
    assert result is not None
    assert result.id == naptr.id


def test_naptr_delete_by_id(
    integration_client: MregClient,
    host: Host,
) -> None:
    integration_client.naptr.create(host=host, preference=40, order=40, replacement=".")
    naptr = integration_client.naptr.get_unique(host, preference=40, order=40, replacement=".")
    assert naptr is not None
    integration_client.naptr.delete(naptr.id)
    assert integration_client.naptr.get(naptr.id, required=False) is None


def test_naptr_delete_by_object(
    integration_client: MregClient,
    host: Host,
) -> None:
    integration_client.naptr.create(host=host, preference=50, order=50, replacement=".")
    naptr = integration_client.naptr.get_unique(host, preference=50, order=50, replacement=".")
    assert naptr is not None
    integration_client.naptr.delete(naptr)
    assert integration_client.naptr.get(naptr.id, required=False) is None


# ── SRV tests ─────────────────────────────────────────────────────────────────


def test_srv_create(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    integration_client.srv.create(
        host=host,
        name=f"_sip._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=5060,
    )
    srv = integration_client.srv.get_unique(
        name=f"_sip._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=5060,
        host=host,
    )
    assert srv is not None
    resource_tracker.add(lambda: integration_client.srv.delete(srv))
    assert srv.port == 5060
    assert srv.host == host.id


def test_srv_get_by_id(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    integration_client.srv.create(
        host=host,
        name=f"_xmpp._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=5222,
    )
    srv = integration_client.srv.get_unique(
        name=f"_xmpp._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=5222,
        host=host,
    )
    assert srv is not None
    resource_tracker.add(lambda: integration_client.srv.delete(srv))
    result = integration_client.srv.get(srv.id)
    assert result is not None
    assert result.id == srv.id


def test_srv_get_unique(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    integration_client.srv.create(
        host=host,
        name=f"_ldap._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=389,
    )
    srv = integration_client.srv.get_unique(
        name=f"_ldap._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=389,
        host=host,
    )
    assert srv is not None
    resource_tracker.add(lambda: integration_client.srv.delete(srv))
    result = integration_client.srv.get_unique(
        name=f"_ldap._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=389,
        host=host,
    )
    assert result is not None
    assert result.id == srv.id


def test_srv_delete_by_id(
    integration_client: MregClient,
    host: Host,
    zone: Zone,
) -> None:
    integration_client.srv.create(
        host=host,
        name=f"_smtp._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=25,
    )
    srv = integration_client.srv.get_unique(
        name=f"_smtp._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=25,
        host=host,
    )
    assert srv is not None
    integration_client.srv.delete(srv.id)
    assert integration_client.srv.get(srv.id, required=False) is None


def test_srv_delete_by_object(
    integration_client: MregClient,
    host: Host,
    zone: Zone,
) -> None:
    integration_client.srv.create(
        host=host,
        name=f"_imaps._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=993,
    )
    srv = integration_client.srv.get_unique(
        name=f"_imaps._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=993,
        host=host,
    )
    assert srv is not None
    integration_client.srv.delete(srv)
    assert integration_client.srv.get(srv.id, required=False) is None


# ── PTR override tests ────────────────────────────────────────────────────────


def test_ptr_create(
    integration_client: MregClient,
    ptr_host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.ptroverride.create(host=ptr_host, ipaddress="10.0.1.51")
    ptrs = integration_client.ptroverride.list(host=ptr_host.id)
    ptr = next(p for p in ptrs if str(p.ipaddress) == "10.0.1.51")
    assert ptr is not None
    resource_tracker.add(lambda: integration_client.ptroverride.delete(ptr))
    assert ptr.host == ptr_host.id


def test_ptr_get_by_id(
    integration_client: MregClient,
    ptr_host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.ptroverride.create(host=ptr_host, ipaddress="10.0.1.52")
    ptrs = integration_client.ptroverride.list(host=ptr_host.id)
    ptr = next(p for p in ptrs if str(p.ipaddress) == "10.0.1.52")
    assert ptr is not None
    resource_tracker.add(lambda: integration_client.ptroverride.delete(ptr))
    result = integration_client.ptroverride.get(ptr.id)
    assert result is not None
    assert result.id == ptr.id


def test_ptr_delete_by_id(
    integration_client: MregClient,
    ptr_host: Host,
) -> None:
    integration_client.ptroverride.create(host=ptr_host, ipaddress="10.0.1.53")
    ptrs = integration_client.ptroverride.list(host=ptr_host.id)
    ptr = next(p for p in ptrs if str(p.ipaddress) == "10.0.1.53")
    assert ptr is not None
    integration_client.ptroverride.delete(ptr.id)
    assert integration_client.ptroverride.get(ptr.id, required=False) is None


def test_ptr_delete_by_object(
    integration_client: MregClient,
    ptr_host: Host,
) -> None:
    integration_client.ptroverride.create(host=ptr_host, ipaddress="10.0.1.54")
    ptrs = integration_client.ptroverride.list(host=ptr_host.id)
    ptr = next(p for p in ptrs if str(p.ipaddress) == "10.0.1.54")
    assert ptr is not None
    integration_client.ptroverride.delete(ptr)
    assert integration_client.ptroverride.get(ptr.id, required=False) is None


# ── SSHFP tests ───────────────────────────────────────────────────────────────


def test_sshfp_create(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.sshfp.create(
        host=host,
        algorithm=1,
        hash_type=1,
        fingerprint="aabbccddeeff0011",
    )
    sshfps = integration_client.sshfp.list(host=host.id)
    sshfp = next(s for s in sshfps if s.fingerprint == "aabbccddeeff0011")
    assert sshfp is not None
    resource_tracker.add(lambda: integration_client.sshfp.delete(sshfp))
    assert sshfp.algorithm == 1
    assert sshfp.hash_type == 1
    assert sshfp.host == host.id


def test_sshfp_get_by_id(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.sshfp.create(
        host=host,
        algorithm=1,
        hash_type=1,
        fingerprint="aabbccddeeff0022",
    )
    sshfps = integration_client.sshfp.list(host=host.id)
    sshfp = next(s for s in sshfps if s.fingerprint == "aabbccddeeff0022")
    assert sshfp is not None
    resource_tracker.add(lambda: integration_client.sshfp.delete(sshfp))
    result = integration_client.sshfp.get(sshfp.id)
    assert result is not None
    assert result.id == sshfp.id


def test_sshfp_delete_by_id(
    integration_client: MregClient,
    host: Host,
) -> None:
    integration_client.sshfp.create(
        host=host,
        algorithm=1,
        hash_type=1,
        fingerprint="aabbccddeeff0033",
    )
    sshfps = integration_client.sshfp.list(host=host.id)
    sshfp = next(s for s in sshfps if s.fingerprint == "aabbccddeeff0033")
    assert sshfp is not None
    integration_client.sshfp.delete(sshfp.id)
    assert integration_client.sshfp.get(sshfp.id, required=False) is None


def test_sshfp_delete_by_object(
    integration_client: MregClient,
    host: Host,
) -> None:
    integration_client.sshfp.create(
        host=host,
        algorithm=1,
        hash_type=1,
        fingerprint="aabbccddeeff0044",
    )
    sshfps = integration_client.sshfp.list(host=host.id)
    sshfp = next(s for s in sshfps if s.fingerprint == "aabbccddeeff0044")
    assert sshfp is not None
    integration_client.sshfp.delete(sshfp)
    assert integration_client.sshfp.get(sshfp.id, required=False) is None


# ── BacnetID tests ────────────────────────────────────────────────────────────


def test_bacnetid_create(
    integration_client: MregClient,
    bacnet_host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.bacnetid.create(host=bacnet_host, id=9001)
    bacnet = integration_client.bacnetid.get_by_host(bacnet_host)
    assert bacnet is not None
    resource_tracker.add(lambda: integration_client.bacnetid.delete(bacnet))
    assert bacnet.id == 9001


def test_bacnetid_get_by_id(
    integration_client: MregClient,
    bacnet_host: Host,
) -> None:
    result = integration_client.bacnetid.get(9001)
    assert result is not None
    assert result.id == 9001


def test_bacnetid_get_by_host(
    integration_client: MregClient,
    bacnet_host: Host,
) -> None:
    result = integration_client.bacnetid.get_by_host(bacnet_host)
    assert result is not None
    assert result.id == 9001


def test_bacnetid_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}bacnetdid.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.bacnetid.create(host=h, id=9002)
    bacnet = integration_client.bacnetid.get(9002)
    assert bacnet is not None
    integration_client.bacnetid.delete(bacnet.id)
    assert integration_client.bacnetid.get(9002, required=False) is None


def test_bacnetid_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}bacnetdo.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.bacnetid.create(host=h, id=9003)
    bacnet = integration_client.bacnetid.get(9003)
    assert bacnet is not None
    integration_client.bacnetid.delete(bacnet)
    assert integration_client.bacnetid.get(9003, required=False) is None


# ── Location tests ────────────────────────────────────────────────────────────

_LOC_VALUE = "59 56 00.0 N 10 41 00.0 E 100m"


def test_location_create(
    integration_client: MregClient,
    loc_host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.location.create(host=loc_host, loc=_LOC_VALUE)
    loc = integration_client.location.get_by_host(loc_host)
    assert loc is not None
    resource_tracker.add(lambda: integration_client.location.delete(loc))
    assert loc.loc == _LOC_VALUE
    assert loc.host == loc_host.id


def test_location_get_by_host_id(
    integration_client: MregClient,
    loc_host: Host,
) -> None:
    result = integration_client.location.get_by_host(loc_host.id)
    assert result is not None
    assert result.host == loc_host.id


def test_location_get_by_host(
    integration_client: MregClient,
    loc_host: Host,
) -> None:
    result = integration_client.location.get_by_host(loc_host)
    assert result is not None
    assert result.host == loc_host.id


def test_location_delete_by_id(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}locdid.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.location.create(host=h, loc=_LOC_VALUE)
    fetched = integration_client.location.get_by_host(h)
    assert fetched is not None
    integration_client.location.delete(fetched)
    assert integration_client.location.get_by_host(h, required=False) is None


def test_location_delete_by_object(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}locdo.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.location.create(host=h, loc=_LOC_VALUE)
    loc = integration_client.location.get_by_host(h)
    assert loc is not None
    integration_client.location.delete(loc)
    assert integration_client.location.get_by_host(h, required=False) is None


# ── Update tests ──────────────────────────────────────────────────────────────


def test_hinfo_update(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}hinfoupd.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.hinfo.create(host=h, cpu="x86_64", os="Linux")
    hi = integration_client.hinfo.get_by_host(h)
    assert hi is not None
    resource_tracker.add(lambda: integration_client.hinfo.delete(hi))
    integration_client.hinfo.update(hi, cpu="ARM", os="FreeBSD")
    refreshed = integration_client.hinfo.refresh(hi)
    assert refreshed.cpu == "ARM"
    assert refreshed.os == "FreeBSD"


def test_txt_update(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}txtupd.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.txt.create(host=h, txt="original txt")
    txts = integration_client.txt.list(host=h.id)
    txt = next(t for t in txts if t.txt == "original txt")
    resource_tracker.add(lambda: integration_client.txt.delete(txt))
    integration_client.txt.update(txt, txt="updated txt")
    refreshed = integration_client.txt.refresh(txt)
    assert refreshed.txt == "updated txt"


def test_mx_update(
    integration_client: MregClient,
    host: Host,
    mx_host: str,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.mx.create(host=host, mx=mx_host, priority=60)
    mx = integration_client.mx.get_unique(host, mx_host, 60)
    assert mx is not None
    resource_tracker.add(lambda: integration_client.mx.delete(mx))
    integration_client.mx.update(mx, priority=65)
    refreshed = integration_client.mx.refresh(mx)
    assert refreshed.priority == 65


def test_naptr_update(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.naptr.create(host=host, preference=60, order=60, replacement=".")
    naptr = integration_client.naptr.get_unique(host, preference=60, order=60, replacement=".")
    assert naptr is not None
    resource_tracker.add(lambda: integration_client.naptr.delete(naptr))
    integration_client.naptr.update(naptr, preference=65)
    refreshed = integration_client.naptr.refresh(naptr)
    assert refreshed.preference == 65


def test_srv_update(
    integration_client: MregClient,
    host: Host,
    zone: Zone,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.srv.create(
        host=host,
        name=f"_ftp._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=21,
    )
    srv = integration_client.srv.get_unique(
        name=f"_ftp._tcp.{zone.name}",
        priority=10,
        weight=10,
        port=21,
        host=host,
    )
    assert srv is not None
    resource_tracker.add(lambda: integration_client.srv.delete(srv))
    integration_client.srv.update(srv, port=2121)
    refreshed = integration_client.srv.refresh(srv)
    assert refreshed.port == 2121


def test_ptr_update(
    integration_client: MregClient,
    ptr_host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.ptroverride.create(host=ptr_host, ipaddress="10.0.1.55")
    ptrs = integration_client.ptroverride.list(host=ptr_host.id)
    ptr = next(p for p in ptrs if str(p.ipaddress) == "10.0.1.55")
    resource_tracker.add(lambda: integration_client.ptroverride.delete(ptr))
    integration_client.ptroverride.update(ptr, ipaddress="10.0.1.56")
    refreshed = integration_client.ptroverride.refresh(ptr)
    assert str(refreshed.ipaddress) == "10.0.1.56"


def test_sshfp_update(
    integration_client: MregClient,
    host: Host,
    resource_tracker: ResourceTracker,
) -> None:
    integration_client.sshfp.create(
        host=host,
        algorithm=1,
        hash_type=1,
        fingerprint="aabbccddeeff0055",
    )
    sshfps = integration_client.sshfp.list(host=host.id)
    sshfp = next(s for s in sshfps if s.fingerprint == "aabbccddeeff0055")
    resource_tracker.add(lambda: integration_client.sshfp.delete(sshfp))
    integration_client.sshfp.update(sshfp, fingerprint="112233445566aabb")
    refreshed = integration_client.sshfp.refresh(sshfp)
    assert refreshed.fingerprint == "112233445566aabb"


def test_location_update(
    integration_client: MregClient,
    test_prefix: str,
    resource_tracker: ResourceTracker,
    zone: Zone,
) -> None:
    name = f"{test_prefix}locupd.{zone.name}"
    h = integration_client.host.create(name=name)
    assert h is not None
    resource_tracker.add(lambda: integration_client.host.delete(h))
    integration_client.location.create(host=h, loc=_LOC_VALUE)
    loc = integration_client.location.get_by_host(h)
    assert loc is not None
    resource_tracker.add(lambda: integration_client.location.delete(loc))
    new_loc = "59 57 00.0 N 10 42 00.0 E 200m"
    integration_client.location.update(loc, loc=new_loc)
    refreshed = integration_client.location.refresh(loc)
    assert refreshed.loc == new_loc
