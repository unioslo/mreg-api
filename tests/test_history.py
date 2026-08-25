from __future__ import annotations

import httpx
import pytest

from mreg_api.history import RequestHistory


@pytest.mark.parametrize(
    "maxsize",
    [
        None,
        5,
        pytest.param(0, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
        pytest.param(-5, marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    ],
)
def test_request_history_init(maxsize: int | None) -> None:
    history = RequestHistory(maxsize=maxsize)
    assert len(history) == 0
    assert history.get() == []


def test_request_history_add_default_maxsize() -> None:
    """No maxsize is specified means the history is unbounded."""
    history = RequestHistory()
    assert history.maxsize is None

    response1 = httpx.Response(200, request=httpx.Request("GET", "https://example.com"))
    response2 = httpx.Response(404, request=httpx.Request("POST", "https://example.com"))
    history.add(response1)
    history.add(response2)
    assert len(history) == 2


def test_request_history_add_overflow() -> None:
    history = RequestHistory(maxsize=2)
    response1 = httpx.Response(200, request=httpx.Request("GET", "https://example.com"))
    response2 = httpx.Response(404, request=httpx.Request("POST", "https://example.com"))
    history.add(response1)
    history.add(response2)

    assert len(history) == 2
    assert history.get()[0].status == 200
    assert history.get()[1].status == 404

    # Adding a third record removes the oldest one
    response3 = httpx.Response(500, request=httpx.Request("PUT", "https://example.com"))
    history.add(response3)
    assert len(history) == 2
    assert history.get()[0].status == 404
    assert history.get()[1].status == 500
