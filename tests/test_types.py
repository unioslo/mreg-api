from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mreg_api.types import Json
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ({"key": "value"}, {"key": "value"}),
        (["item1", "item2"], ["item1", "item2"]),
        ("string", "string"),
        (42, 42),
        (3.14, 3.14),
        (True, True),
        (None, None),
    ],
)
def test_json_type_python(input_value: Any, expected: Any) -> None:
    # Use the convenience function to get the type adapter
    adapter = get_type_adapter(Json)
    assert adapter.validate_python(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ('{"key": "value"}', {"key": "value"}),
        ('["item1", "item2"]', ["item1", "item2"]),
        ('"string"', "string"),
        ("42", 42),
        ("3.14", 3.14),
        ("true", True),
        ("null", None),
    ],
)
def test_json_type_jsonstr(input_value: str, expected: Any) -> None:
    # Use the convenience function to get the type adapter
    adapter = get_type_adapter(Json)
    assert adapter.validate_json(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ({"key": "value"}, {"key": "value"}),
        ({"item1": 1, "item2": 2}, {"item1": 1, "item2": 2}),
        ({}, {}),
        (
            {"nested": {"a": 1, "b": [1, "foo", None, {"extra_nesting": {"c": 3, "d": 4}}]}},
            {"nested": {"a": 1, "b": [1, "foo", None, {"extra_nesting": {"c": 3, "d": 4}}]}},
        ),
    ],
)
def test_jsonmapping_type_python(input_value: Any, expected: Any) -> None:
    # Use the convenience function to get the type adapter
    adapter = get_type_adapter(JsonMapping)
    assert adapter.validate_python(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        ('{"key": "value"}', {"key": "value"}),
        ('{"item1": 1, "item2": 2}', {"item1": 1, "item2": 2}),
        ("{}", {}),
        (
            '{"nested": {"a": 1, "b": [1, "foo", null, {"extra_nesting": {"c": 3, "d": 4}}]}}',
            {"nested": {"a": 1, "b": [1, "foo", None, {"extra_nesting": {"c": 3, "d": 4}}]}},
        ),
    ],
)
def test_jsonmapping_type_jsonstr(input_value: str, expected: Any) -> None:
    # Use the convenience function to get the type adapter
    adapter = get_type_adapter(JsonMapping)
    assert adapter.validate_json(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        # Strings only
        ({"search": "query", "page": "2"}, {"search": "query", "page": "2"}),
        # Mixed types (all allowed)
        (
            {"search": "query", "page": 1, "pi": 3.14, "valid": True, "filter": None},
            {"search": "query", "page": 1, "pi": 3.14, "valid": True, "filter": None},
        ),
        ({}, {}),
        # Invalid case: input is not a mapping
        pytest.param(
            {"param1": [1, 2, 3]},
            None,
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
        ),
        # Invalid case: Keys are not strings
        pytest.param(
            {1: "value1", 2: "value2"},
            None,
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
        ),
        # Invalid case: Value with invalid type
        pytest.param(
            {"param1": object()},
            None,
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
        ),
        # Invalid case: Nesting NYI (may allow in future)
        pytest.param(
            {"nested": {"key": "value"}},
            {"nested": {"key": "value"}},
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
            id="Nested dicts not allowed",
        ),
        # Invalid case: Lists as values NYI (may allow in future)
        pytest.param(
            {"array": [1, 2, 3]},
            {"array": [1, 2, 3]},
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
            id="Lists not allowed",
        ),
    ],
)
def test_queryparams_type_python(input_value: Any, expected: Any) -> None:
    # Use the convenience function to get the type adapter
    adapter = get_type_adapter(QueryParams)
    assert adapter.validate_python(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        # Strings only
        ('{"search": "query", "page": "2"}', {"search": "query", "page": "2"}),
        # Mixed types (all allowed)
        (
            '{"search": "query", "page": 1, "pi": 3.14, "valid": true, "filter": null}',
            {"search": "query", "page": 1, "pi": 3.14, "valid": True, "filter": None},
        ),
        ("{}", {}),
        # Invalid case: input is not a mapping
        pytest.param(
            '{"param1": [1, 2, 3]}',
            None,
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
        ),
        # Invalid case: Nesting NYI (may allow in future)
        pytest.param(
            '{"nested": {"key": "value"}}',
            {"nested": {"key": "value"}},
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
            id="Nested dicts not allowed",
        ),
        # Invalid case: Lists as values NYI (may allow in future)
        pytest.param(
            '{"array": [1, 2, 3]}',
            {"array": [1, 2, 3]},
            marks=pytest.mark.xfail(raises=ValidationError, strict=True),
            id="Lists not allowed",
        ),
    ],
)
def test_queryparams_type_jsonstr(input_value: str, expected: Any) -> None:
    # Use the convenience function to get the type adapter
    adapter = get_type_adapter(QueryParams)
    assert adapter.validate_json(input_value) == expected
