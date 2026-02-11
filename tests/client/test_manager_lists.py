from __future__ import annotations

from typing import Any

import pytest
from pytest_httpserver import HTTPServer

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityNotFound
from mreg_api.models.manager import ModelList
from mreg_api.models.manager import ModelManager


def test_manager_get_list_returns_model_list(httpserver: HTTPServer, client: MregClient) -> None:
    httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
        [
            {
                "id": 1,
                "name": "host1.example.com",
                "ipaddresses": [],
                "comment": "My comment",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        ]
    )

    hosts = client.host().get_list()
    assert isinstance(hosts, ModelList)
    assert [str(host.name) for host in hosts] == ["host1.example.com"]


def test_model_list_delete_bulk(httpserver: HTTPServer, client: MregClient) -> None:
    httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
        [
            {
                "id": 1,
                "name": "host1.example.com",
                "ipaddresses": [],
                "comment": "My comment",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 2,
                "name": "host2.example.com",
                "ipaddresses": [],
                "comment": "My other comment",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            },
        ]
    )
    httpserver.expect_oneshot_request("/api/v1/hosts/host1.example.com", method="DELETE").respond_with_json({})
    httpserver.expect_oneshot_request("/api/v1/hosts/host2.example.com", method="DELETE").respond_with_json({})

    hosts = client.host().get_list()
    hosts.delete()


def test_model_list_patch_dispatches() -> None:
    class Patchable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch(self, fields: dict[str, Any] | None = None, validate: bool = True, **field_kwargs: Any) -> Patchable:
            payload = fields if fields is not None else field_kwargs
            self.calls.append((payload, validate))
            return self

    one = Patchable()
    two = Patchable()
    objects = ModelList([one, two])

    patched = objects.patch(comment="updated", validate=False)

    assert isinstance(patched, ModelList)
    assert patched == [one, two]
    assert one.calls == [({"comment": "updated"}, False)]
    assert two.calls == [({"comment": "updated"}, False)]


def test_model_list_delete_raises_on_failure() -> None:
    class Deletable:
        def __init__(self, ok: bool) -> None:
            self.ok = ok

        def delete(self) -> bool:
            return self.ok

    objects = ModelList([Deletable(True), Deletable(False)])
    with pytest.raises(RuntimeError, match="Failed to delete"):
        _ = objects.delete()


def test_model_list_patch_raises_if_missing_method() -> None:
    objects = ModelList([object()])
    with pytest.raises(TypeError, match="Cannot bulk-patch"):
        _ = objects.patch({"comment": "updated"})


def test_model_list_patch_matches_manual_loop() -> None:
    class Patchable:
        def __init__(self, identifier: int) -> None:
            self.identifier = identifier
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch(self, fields: dict[str, Any] | None = None, validate: bool = True, **field_kwargs: Any) -> Patchable:
            payload = fields if fields is not None else field_kwargs
            self.calls.append((payload, validate))
            return self

    bulk_objects = ModelList([Patchable(1), Patchable(2)])
    loop_objects = ModelList([Patchable(1), Patchable(2)])

    bulk_result = bulk_objects.patch(comment="hi")

    loop_result = ModelList[Patchable]()
    for host in loop_objects:
        loop_result.append(host.patch(comment="hi"))

    assert isinstance(bulk_result, ModelList)
    assert bulk_result == bulk_objects
    assert [obj.identifier for obj in bulk_result] == [obj.identifier for obj in loop_result]
    assert [obj.calls for obj in bulk_objects] == [obj.calls for obj in loop_objects]


def test_model_list_patch_raw_dispatches() -> None:
    class Patchable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch_raw(self, fields: dict[str, Any], validate: bool = True) -> Patchable:
            self.calls.append((fields, validate))
            return self

    one = Patchable()
    two = Patchable()
    objects = ModelList([one, two])

    patched = objects.patch_raw({"comment": "updated"}, validate=False)

    assert isinstance(patched, ModelList)
    assert patched == [one, two]
    assert one.calls == [({"comment": "updated"}, False)]
    assert two.calls == [({"comment": "updated"}, False)]


def test_model_list_patch_typed_dispatches() -> None:
    class Patchable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch_typed(self, *, validate: bool = True, **field_kwargs: Any) -> Patchable:
            self.calls.append((field_kwargs, validate))
            return self

    one = Patchable()
    two = Patchable()
    objects = ModelList([one, two])

    patched = objects.patch_typed(comment="updated", validate=False)

    assert isinstance(patched, ModelList)
    assert patched == [one, two]
    assert one.calls == [({"comment": "updated"}, False)]
    assert two.calls == [({"comment": "updated"}, False)]


def test_model_list_patch_typed_raises_if_missing_method() -> None:
    objects = ModelList([object()])
    with pytest.raises(TypeError, match="Cannot bulk-patch_typed"):
        _ = objects.patch_typed(comment="updated")


def test_model_manager_get_by_id_or_raise_returns_object() -> None:
    class DummyModel:
        @classmethod
        def get_by_id(cls, client: object, _id: int, _manager: bool = False) -> object | None:
            _ = (client, _id, _manager)
            return {"id": 1}

    manager = ModelManager(MregClient(), DummyModel)
    assert manager.get_by_id_or_raise(1) == {"id": 1}


def test_model_manager_get_by_id_or_raise_raises_with_default_message() -> None:
    class DummyModel:
        @classmethod
        def get_by_id(cls, client: object, _id: int, _manager: bool = False) -> object | None:
            _ = (client, _id, _manager)
            return None

    manager = ModelManager(MregClient(), DummyModel)
    with pytest.raises(EntityNotFound, match="DummyModel with ID 1 not found"):
        _ = manager.get_by_id_or_raise(1)


