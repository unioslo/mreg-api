from __future__ import annotations

from typing import Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from pydantic import ValidationError

from mreg_api.exceptions import InputFailure
from mreg_api.models.fields import HostNameField
from mreg_api.models.fields import MacAddress
from mreg_api.models.fields import NameList
from mreg_api.models.fields import _normalize_mac_address  # pyright: ignore[reportPrivateUsage]
from mreg_api.models.fields import parse_hostname


@pytest.mark.parametrize(
    "hostname",
    [
        "example.com",
        "sub.domain.com",
        "localhost",
        "localhost.",
        "my-host-123.com",
        "123-start-with-number.com",
        "singlelabel",
        "sub.sub2.sub3.domain.com",
        "_underscore.hostname.com",
        "host-name-with-dashes.com",
        "multi.label.domain.co.uk",
        "*.example.com",
        "*.sub.domain.com",
        "*.localhost",
        "*.my-host-123.com",
        "*.123-start-with-number.com",
        "*.singlelabel",
        "*.sub.sub2.sub3.domain.com",
        "*.underscore.hostname.com",
        "*.host-name-with-dashes.com",
        "*.multi.label.domain.co.uk",
    ],
)
def test_valid_hostname(hostname: str) -> None:
    res = parse_hostname(hostname)
    assert res
    assert isinstance(res, str)

    # When used as a Pydantic field type, HostNameField validates to str:
    class TestModel(BaseModel):
        name: HostNameField

    m = TestModel(name=hostname)
    assert isinstance(m.name, str)


@pytest.mark.parametrize(
    "hostname",
    [
        "-example.com",
        "sub..domain.com",
        ".singlelabel",
        "multi..label.domain.co.uk",
        "*.sub.-domain.com",
        "localhost*",
        "host name with spaces.com",
        "example.com/net",
        "*.sub..domain.com",
        "host>name.com",
        "example.com#section",
        "123&456.com",
        # TODO: Make these invalid names fail validation:
        pytest.param(
            "example-.com",
            marks=pytest.mark.xfail(
                reason="ends with '-'",
                strict=True,
            ),
        ),
        pytest.param(
            "*.example-.com",
            marks=pytest.mark.xfail(
                reason="ends with '-'",
                strict=True,
            ),
        ),
        pytest.param(
            "host--name-with-dashes.com",
            marks=pytest.mark.xfail(
                reason="double '-'",
                strict=True,
            ),
        ),
        pytest.param(
            "_underscore_.hostname.com",
            marks=pytest.mark.xfail(
                reason="Ends with '_'",
                strict=True,
            ),
        ),
        pytest.param(
            "my_host_123.com",
            marks=pytest.mark.xfail(
                reason="Underscores between words",
                strict=True,
            ),
        ),
        pytest.param(
            "123.start-with-number.com",
            marks=pytest.mark.xfail(
                reason="Starts with number",
                strict=True,
            ),
        ),
    ],
)
def test_invalid_hostname(hostname: str) -> None:
    with pytest.raises(InputFailure):
        parse_hostname(hostname)


def test_parse_hostname_lowercases() -> None:
    assert parse_hostname("WEB.EXAMPLE.COM") == "web.example.com"


def test_parse_hostname_strips_trailing_dot() -> None:
    assert parse_hostname("example.com.") == "example.com"


def test_parse_hostname_expands_with_domain() -> None:
    assert parse_hostname("web", "example.com") == "web.example.com"


def test_parse_hostname_no_expand_when_dot_present() -> None:
    assert parse_hostname("web.other.com", "example.com") == "web.other.com"


def test_parse_hostname_no_expand_when_no_domain() -> None:
    assert parse_hostname("web") == "web"


def test_parse_hostname_trailing_dot_then_qualified() -> None:
    # trailing dot stripped → has dot → returned as-is
    assert parse_hostname("web.example.com.", "other.org") == "web.example.com"


def test_parse_hostname_returns_hostname_type() -> None:
    """Test that parse_hostname returns a str on runtime."""
    result = parse_hostname("example.com")
    assert isinstance(result, str)
    # HostName is a NewType of str — runtime identity is str
    assert type(result) is str


def test_parse_hostname_domain_none_is_equal_empty_str() -> None:
    """Test that parse_hostname returns the same result for None and empty string domains."""
    assert parse_hostname("web", None) == parse_hostname("web", "") == "web"


MacAddressValidationFailure = pytest.mark.xfail(raises=InputFailure, strict=True)


@pytest.mark.parametrize(
    "inp, expect",
    [
        # 6-octet colon-separated MAC addresses
        ("00:00:00:00:00:00", "00:00:00:00:00:00"),
        ("FF:FF:FF:FF:FF:FF", "ff:ff:ff:ff:ff:ff"),
        ("A1:B2:C3:D4:E5:F6", "a1:b2:c3:d4:e5:f6"),
        ("a1:b2:c3:d4:e5:f6", "a1:b2:c3:d4:e5:f6"),
        ("Ab:cD:eF:01:23:45", "ab:cd:ef:01:23:45"),
        # 6-octet hyphen-separated MAC addresses
        ("00-00-00-00-00-00", "00:00:00:00:00:00"),
        ("FF-FF-FF-FF-FF-FF", "ff:ff:ff:ff:ff:ff"),
        ("A1-B2-C3-D4-E5-F6", "a1:b2:c3:d4:e5:f6"),
        ("a1-b2-c3-d4-e5-f6", "a1:b2:c3:d4:e5:f6"),
        ("Ab-cD-eF-01-23-45", "ab:cd:ef:01:23:45"),
        # 6-octet dot-separated MAC addresses
        ("0000.0000.0000", "00:00:00:00:00:00"),
        ("FFFF.FFFF.FFFF", "ff:ff:ff:ff:ff:ff"),
        ("A1B2.C3D4.E5F6", "a1:b2:c3:d4:e5:f6"),
        ("a1b2.c3d4.e5f6", "a1:b2:c3:d4:e5:f6"),
        ("Ab12.cD34.eF56", "ab:12:cd:34:ef:56"),
        # 8-octet colon-separated MAC addresses
        ("00:00:00:00:00:00:00:00", "00:00:00:00:00:00:00:00"),
        ("FF:FF:FF:FF:FF:FF:FF:FF", "ff:ff:ff:ff:ff:ff:ff:ff"),
        ("A1:B2:C3:D4:E5:F6:07:08", "a1:b2:c3:d4:e5:f6:07:08"),
        ("a1:b2:c3:d4:e5:f6:07:08", "a1:b2:c3:d4:e5:f6:07:08"),
        ("Ab:cD:eF:01:23:45:67:89", "ab:cd:ef:01:23:45:67:89"),
        # 8-octet hyphen-separated MAC addresses
        ("00-00-00-00-00-00-00-00", "00:00:00:00:00:00:00:00"),
        ("FF-FF-FF-FF-FF-FF-FF-FF", "ff:ff:ff:ff:ff:ff:ff:ff"),
        ("A1-B2-C3-D4-E5-F6-07-08", "a1:b2:c3:d4:e5:f6:07:08"),
        ("a1-b2-c3-d4-e5-f6-07-08", "a1:b2:c3:d4:e5:f6:07:08"),
        ("Ab-cD-eF-01-23-45-67-89", "ab:cd:ef:01:23:45:67:89"),
        # 8-octet dot-separated MAC addresses
        ("0000.0000.0000.0000", "00:00:00:00:00:00:00:00"),
        ("FFFF.FFFF.FFFF.FFFF", "ff:ff:ff:ff:ff:ff:ff:ff"),
        ("A1B2.C3D4.E5F6.0708", "a1:b2:c3:d4:e5:f6:07:08"),
        ("a1b2.c3d4.e5f6.0708", "a1:b2:c3:d4:e5:f6:07:08"),
        ("Ab12.cD34.eF56.6789", "ab:12:cd:34:ef:56:67:89"),
        # 20-octet colon-separated MAC addresses
        (
            "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
            "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        ),
        (
            "FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF:FF",
            "ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff",
        ),
        (
            "A1:B2:C3:D4:E5:F6:07:08:09:0A:0B:0C:0D:0E:0F:10:11:12:13:14",
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
        ),
        (
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
        ),
        (
            "Ab:cD:eF:01:23:45:67:89:0A:0b:Cd:eF:12:34:56:78:90:Ab:Cd:Ef",
            "ab:cd:ef:01:23:45:67:89:0a:0b:cd:ef:12:34:56:78:90:ab:cd:ef",
        ),
        # 20-octet hyphen-separated MAC addresses
        (
            "00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00",
            "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        ),
        (
            "FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF-FF",
            "ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff",
        ),
        (
            "A1-B2-C3-D4-E5-F6-07-08-09-0A-0B-0C-0D-0E-0F-10-11-12-13-14",
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
        ),
        (
            "a1-b2-c3-d4-e5-f6-07-08-09-0a-0b-0c-0d-0e-0f-10-11-12-13-14",
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
        ),
        (
            "Ab-cD-eF-01-23-45-67-89-0A-0b-Cd-eF-12-34-56-78-90-Ab-Cd-Ef",
            "ab:cd:ef:01:23:45:67:89:0a:0b:cd:ef:12:34:56:78:90:ab:cd:ef",
        ),
        # 20-octet dot-separated MAC addresses (10 groups of 4)
        (
            "0000.0000.0000.0000.0000.0000.0000.0000.0000.0000",
            "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        ),
        (
            "FFFF.FFFF.FFFF.FFFF.FFFF.FFFF.FFFF.FFFF.FFFF.FFFF",
            "ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff",
        ),
        (
            "A1B2.C3D4.E5F6.0708.090A.0B0C.0D0E.0F10.1112.1314",
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
        ),
        (
            "a1b2.c3d4.e5f6.0708.090a.0b0c.0d0e.0f10.1112.1314",
            "a1:b2:c3:d4:e5:f6:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14",
        ),
        (
            "Ab12.cD34.eF56.6789.0AbC.dEf1.2345.6789.0aBc.DeF0",
            "ab:12:cd:34:ef:56:67:89:0a:bc:de:f1:23:45:67:89:0a:bc:de:f0",
        ),
        # Invalid mac addresses
        pytest.param("00:00:00:00:00:00:00", "", marks=MacAddressValidationFailure),
        pytest.param("00:00:00:00:00", "", marks=MacAddressValidationFailure),
        pytest.param("00:00:00:00:00:0", "", marks=MacAddressValidationFailure),
        pytest.param("00:00:00:00:00:0g", "", marks=MacAddressValidationFailure),
        pytest.param("00-00-00-00-00-00:00", "", marks=MacAddressValidationFailure),
        pytest.param("00-00-00-00-00", "", marks=MacAddressValidationFailure),
        pytest.param("00-00-00-00-00-0", "", marks=MacAddressValidationFailure),
        pytest.param("00-00-00-00-00-0g", "", marks=MacAddressValidationFailure),
        pytest.param("ab:cd:ef:12:34", "", marks=MacAddressValidationFailure),
        pytest.param("ab-cd-ef-12-34", "", marks=MacAddressValidationFailure),
        pytest.param("abcd.ef12.34", "", marks=MacAddressValidationFailure),
    ],
)
def test_mac_address_type(inp: str, expect: str) -> None:
    """Test the MAC address field."""
    res = MacAddress.parse_or_raise(inp)
    assert str(res) == expect
    # Narrow and broad type
    assert isinstance(res, MacAddress)
    assert isinstance(res, str)

    # When used as a Pydantic field type, the field validates to str
    class TestModel(BaseModel):
        mac: MacAddress

    m = TestModel(mac=inp)
    assert m.mac == expect
    assert isinstance(m.mac, str)
    assert not isinstance(m.mac, MacAddress)  # Core schema coerces this to str


@pytest.mark.parametrize(
    "macaddress,sep,expected",
    [
        # 6 octet
        ("001122334455", ":", "00:11:22:33:44:55"),
        ("001122334455", "-", "00-11-22-33-44-55"),
        ("001122334455", ".", "0011.2233.4455"),
        # 8 octet
        ("0011223344556677", ":", "00:11:22:33:44:55:66:77"),
        ("0011223344556677", "-", "00-11-22-33-44-55-66-77"),
        ("0011223344556677", ".", "0011.2233.4455.6677"),
        # 20 octet
        (
            "00112233445566778899aabbccddeeff00112233",
            ":",
            "00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff:00:11:22:33",
        ),
        (
            "00112233445566778899aabbccddeeff00112233",
            "-",
            "00-11-22-33-44-55-66-77-88-99-aa-bb-cc-dd-ee-ff-00-11-22-33",
        ),
        (
            "00112233445566778899aabbccddeeff00112233",
            ".",
            "0011.2233.4455.6677.8899.aabb.ccdd.eeff.0011.2233",
        ),
    ],
)
def test__normalize_mac_address(macaddress: str, sep: Literal[":", "-", "."], expected: str) -> None:
    """Test the _normalize_mac_address function with various MAC address formats and separators."""
    assert _normalize_mac_address(macaddress, sep) == expected


@pytest.mark.parametrize(
    "macaddress",
    [
        # 6 octet
        "00:11:22:33:44:55",
        "00-11-22-33-44-55",
        "0011.2233.4455",
        # 8 octet
        "0011:2233:4455:6677",
        "00-11-22-33-44-55-66-77",
        "0011.2233.4455.6677",
        # 20 octet
        "00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff:00:11:22:33",
        "00-11-22-33-44-55-66-77-88-99-aa-bb-cc-dd-ee-ff-00-11-22-33",
        "0011.2233.4455-66-77-88-99-aa-bb-cc-dd-ee-ff-00-11-22-33",
    ],
)
@pytest.mark.parametrize(
    "sep",
    [
        ":",
        "-",
        ".",
    ],
)
def test__normalize_mac_address_existing_sep(
    macaddress: str,
    sep: Literal[":", "-", "."],
) -> None:
    """Test _normalize_mac_address function with MAC addresses that already contain separators.

    The function should return the original MAC address unchanged regardless of the specified separator.
    """
    # unchanged
    assert _normalize_mac_address(macaddress, sep) == macaddress


def test_name_list_basic():
    """Test NameList field with basic input."""
    inp = {
        "hosts": [
            {"name": "test1", "value": 1},
            {"name": "test2"},
            {"name": "test3", "value": 3},
        ]
    }

    class TestModel(BaseModel):
        hosts: NameList

    m = TestModel.model_validate(inp)

    assert m.model_dump(mode="json") == snapshot({"hosts": ["test1", "test2", "test3"]})


def test_name_list_with_invalid_item(caplog: pytest.LogCaptureFixture):
    """Test NameList field with an item without a name.

    Should log an error and skip the item.
    """
    inp = {
        "hosts": [
            {"name": "test1", "value": 1},
            {"value": 2},
            {"name": "test3", "value": 3},
        ]
    }

    class TestModel(BaseModel):
        hosts: NameList

    m = TestModel.model_validate(inp)

    assert m.model_dump(mode="json") == snapshot({"hosts": ["test1", "test3"]})

    assert caplog.record_tuples == snapshot([("mreg_api.models.fields", 40, "No 'name' key in {'value': 2}")])


def test_name_list_invalid_type():
    """Test NameList field with the wrong type (dict instead of list of dicts)."""
    # hosts is not a list
    inp = {"hosts": {"name": "test1", "value": 1}}

    class TestModel(BaseModel):
        hosts: NameList

    with pytest.raises(ValidationError) as exc_info:
        TestModel.model_validate(inp)

    assert exc_info.value.errors(include_url=False) == snapshot(
        [
            {
                "type": "list_type",
                "loc": ("hosts",),
                "msg": "Input should be a valid list",
                "input": {"name": "test1", "value": 1},
            }
        ]
    )


def test_name_list_with_list() -> None:
    """Test NameList field with a list of strings. Should return the same list."""
    inp = {"hosts": ["test1", "test2", "test3"]}

    class TestModel(BaseModel):
        hosts: NameList

    m = TestModel.model_validate(inp)

    assert m.model_dump(mode="json") == snapshot({"hosts": ["test1", "test2", "test3"]})


def test_name_list_with_empty_name() -> None:
    """Test NameList field with a list of strings, where one name is an empty string."""
    inp = {"hosts": ["test1", "test2", "", "test3"]}

    class TestModel(BaseModel):
        hosts: NameList

    m = TestModel.model_validate(inp)

    # NOTE: this is a special case where the empty string is removed,
    # just like with the list of dictionaries. Whether or not this is
    # desirable is up for debate.
    # This test ensures that any change to that behavior is caught.
    assert m.model_dump(mode="json") == snapshot({"hosts": ["test1", "test2", "test3"]})
