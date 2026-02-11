from __future__ import annotations

from typing import Any
from typing import cast

import pytest
from inline_snapshot import snapshot
from pytest_httpserver import HTTPServer

from mreg_api import models
from mreg_api.client import MregClient
from mreg_api.models.manager import ModelManager
from mreg_api.models.manager import to_snake_case
from mreg_api.models.models import Host


def test_client_model_composition(client: MregClient, httpserver: HTTPServer) -> None:
    """MregClient has models composed as attributes for easy access."""
    assert hasattr(client, "host")
    assert client.host().model is Host

    # Test that we can use the composed model to make requests
    httpserver.expect_oneshot_request("/api/v1/hosts/").respond_with_json(
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
    assert [(host.id, host.name, host.comment) for host in hosts] == snapshot([(1, "host1.example.com", "My comment")])


def _client_models() -> list[type]:
    client_models: list[type] = []
    for model in models.__all__:
        model_obj = getattr(models, model)
        # All models with a get or fetch method should be accessible via the client
        if any(hasattr(model_obj, attr) for attr in ["get", "fetch"]):
            client_models.append(model_obj)
    return client_models


@pytest.mark.parametrize("model", _client_models())
def test_client_model_composition_dynamic(model: type, client: MregClient) -> None:
    """Ensure all models with get/fetch are accessible via client attributes."""
    attr_name = to_snake_case(model.__name__)
    assert hasattr(client, attr_name)
    manager: Any = getattr(client, attr_name)
    if callable(manager):
        manager = manager()
    typed_manager = cast(ModelManager[Any], manager)
    assert typed_manager.model is model
