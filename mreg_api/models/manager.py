"""Client-bound model managers for MREG API resources."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any
from typing import Generic
from typing import TypeVar

from mreg_api.types import ClientProtocol

T = TypeVar("T")

_SPECIAL_SNAKE_CASE: dict[str, str] = {
    "BacnetID": "bacnet_id",
    "CNAME": "cname",
    "HInfo": "hinfo",
    "IPAddress": "ip_address",
    "LDAPHealth": "ldap_health",
    "MX": "mx",
    "NAPTR": "naptr",
    "PTR_override": "ptr_override",
    "SSHFP": "sshfp",
    "TXT": "txt",
}


def to_snake_case(name: str) -> str:
    """Convert a model class name to a client attribute name."""
    if name in _SPECIAL_SNAKE_CASE:
        return _SPECIAL_SNAKE_CASE[name]
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def client_model_map(models_module: ModuleType) -> dict[str, type]:
    """Build a mapping of client attribute name -> model class."""
    mapping: dict[str, type] = {}
    for model_name in getattr(models_module, "__all__", []):
        model_obj = getattr(models_module, model_name)
        if any(getattr(model_obj, attr, None) is not None for attr in ["get", "fetch"]):
            mapping[to_snake_case(model_obj.__name__)] = model_obj
    return mapping


class ModelManager(Generic[T]):
    """Bind a model's class API to a specific client instance."""

    def __init__(self, client: ClientProtocol, model: type[T]) -> None:
        """Create a manager bound to a client and model."""
        self._client = client
        self._model = model

    def __call__(self) -> "ModelManager[T]":
        """Return self to support fluent call chains."""
        return self

    @property
    def model(self) -> type[T]:
        """Return the model class this manager wraps."""
        return self._model

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access, binding client for compatible callables."""
        attr = getattr(self._model, name)
        if not callable(attr):
            return attr
        if inspect.ismethod(attr):
            try:
                sig = inspect.signature(attr)
            except (TypeError, ValueError):
                return attr
            params = list(sig.parameters.values())
            if params and params[0].name == "client":
                return functools.partial(attr, self._client, _manager=True)
        return attr
