from __future__ import annotations

import pytest

from mreg_api.client import MregClient


@pytest.fixture(autouse=True, scope="function")
def reset_instance_after_test() -> None:
    """Reset MregClient singleton instance after each test."""
    try:
        MregClient.reset_instance()
    except KeyError:
        pass
