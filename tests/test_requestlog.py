from __future__ import annotations

import httpx
import pytest

from mreg_api.requestlog import RequestLog


@pytest.mark.parametrize(
    "maxsize",
    [
        None,
        5,
        pytest.param(0, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
        pytest.param(-5, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    ],
)
def test_request_log_init(maxsize: int | None) -> None:
    log = RequestLog(maxsize=maxsize)
    assert len(log) == 0
    assert log.get() == []


def test_request_log_add_default_maxsize() -> None:
    """No maxsize is specified means the log is unbounded."""
    log = RequestLog()
    assert log.maxsize is None

    response1 = httpx.Response(200, request=httpx.Request("GET", "https://example.com"))
    response2 = httpx.Response(404, request=httpx.Request("POST", "https://example.com"))
    log.add(response1)
    log.add(response2)
    assert len(log) == 2


def test_request_log_add_overflow() -> None:
    log = RequestLog(maxsize=2)
    response1 = httpx.Response(200, request=httpx.Request("GET", "https://example.com"))
    response2 = httpx.Response(404, request=httpx.Request("POST", "https://example.com"))
    log.add(response1)
    log.add(response2)

    assert len(log) == 2
    assert log.get()[0].status == 200
    assert log.get()[1].status == 404

    # Adding a third record removes the oldest one
    response3 = httpx.Response(500, request=httpx.Request("PUT", "https://example.com"))
    log.add(response3)
    assert len(log) == 2
    assert log.get()[0].status == 404
    assert log.get()[1].status == 500


def test_history_size_kwarg_deprecated_alias() -> None:
    """`history_size` still works but warns and aliases `request_log_size`."""
    from mreg_api.client import MregClient

    with pytest.warns(DeprecationWarning, match="request_log_size"):
        client = MregClient("https://example.com", history_size=3)
    assert client.requests.maxsize == 3
