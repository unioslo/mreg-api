"""Request history management for the MREG API client."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from typing import NamedTuple

from httpx import Request
from httpx import Response

from mreg_api.types import Json
from mreg_api.types import JsonMapping


class RequestRecord(NamedTuple):
    """A complete record of an HTTP request and its response.

    Captures all relevant details of an API call including the request
    parameters, response data, and metadata like status code and URL.
    """

    method: str
    request: Request
    response: Response
    status: int
    data: JsonMapping | None
    json: Json | None

    @property
    def path(self) -> str:
        """Get the request path (URL without base)."""
        # TODO: Make this less hacky! Can we get the path + query directly from httpx?
        parts = [f"{self.response.request.url.scheme}://", self.response.request.url.host]

        if (port_s := f":{self.response.request.url.port}") in self.url:
            parts.append(port_s)

        return self.url.removeprefix("".join(parts))

    @property
    def url(self) -> str:
        """Get the full request URL."""
        return str(self.request.url)


class RequestHistory:
    """A history of HTTP requests made by the client."""

    def __init__(self, maxsize: int | None = None):
        """Initialize the request history.

        Args:
            maxsize: Maximum number of records to keep in history. If None, history is unbounded.
        """
        if maxsize is not None and maxsize <= 0:
            raise ValueError("Request history max size must be >=0 or None")
        self._records: deque[RequestRecord] = deque(maxlen=maxsize)

    @property
    def maxsize(self) -> int | None:
        """Get the maximum number of records that can be stored in history."""
        return self._records.maxlen

    def add(
        self,
        response: Response,
        data: JsonMapping | None = None,
        json: Json | None = None,
    ) -> None:
        """Add a new request record to the history."""
        r = RequestRecord(
            method=response.request.method,
            request=response.request,
            response=response,
            status=response.status_code,
            data=data,
            json=json,
        )
        self._records.append(r)

    def _add_record(self, r: RequestRecord) -> None:
        """Add a new request record to the history."""
        self._records.append(r)

    def clear(self) -> None:
        """Clear the request history."""
        self._records.clear()

    def get(
        self,
        *,
        status: int | None = None,
        method: str | None = None,
        url: str | None = None,
        where: Callable[[RequestRecord], bool] | None = None,
    ) -> list[RequestRecord]:
        """Return all recorded requests, optionally filtered by status, method and/or URL.

        Multiple filters can be applied at once, and only requests matching
        all provided filters will be returned.

        More complex filtering can be done by providing a callable to the
        `where` parameter, which acts as a predicate function to determine
        if a request record should be included in the results.

        Args:
            status (int | None, optional): If provided, only requests with
                this status code are returned. Defaults to None.
            method (str | None, optional): If provided, only requests with
                this HTTP method are returned. Defaults to None.
            url (str | None, optional): If provided, only requests to this
                URL are returned. Defaults to None.
            where (Callable[[RequestRecord], bool] | None, optional):
                A predicate function that returns True if the request record
                should be included.

        Returns:
            list[RequestRecord]: _description_

        Args:
            status:
            method: If given, return only requests with this HTTP method.
            url: If given, return only requests to this URL.
            where: If given, return only records for which this callable returns True.

        Returns:
            A list of request records matching the specified filters.
        """
        where = where or (lambda _: True)  # coalesce to a no-op filter if none is provided
        return [
            r
            for r in self._records
            if (status is None or r.status == status)
            and (method is None or r.method == method)
            and (url is None or r.url == url)
            and where(r)
        ]

    def last(self) -> RequestRecord | None:
        """Get the most recent request record, or None if history is empty."""
        return self._records[-1] if self._records else None

    def __iter__(self) -> Iterator[RequestRecord]:
        """Iterate over the request records in the history."""
        return iter(self._records)

    def __len__(self) -> int:
        """Return the number of request records in the history."""
        return len(self._records)
