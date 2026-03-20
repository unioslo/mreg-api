"""Event log for tracking operations and resolutions in the MREG API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import override


class EventKind(str, Enum):
    """Classification of an event."""

    RESOLUTION = "resolution"
    """Object found via an alias or redirect (e.g. CNAME, PTR override)."""

    MUTATION = "mutation"
    """Object was created, patched, or deleted."""

    INFO = "info"
    """General informational event."""

    WARNING = "warning"
    """Something notable but not an error."""


@dataclass(frozen=True)
class ObjectRef:
    """Lightweight reference to an mreg API object, used for event correlation.

    ``type`` is the model class name (e.g. ``"Host"``, ``"MX"``).
    ``id`` is always a string: pass named objects directly and convert numeric
    IDs with ``str()``.
    """

    type: str
    id: str

    @override
    def __str__(self) -> str:
        """Return a human-readable representation of the reference."""
        return f"{self.type}({self.id!r})"


@dataclass(frozen=True)
class Event:
    """Structured record of something that happened to or around an mreg object."""

    kind: EventKind
    message: str
    """Human-readable description of the event."""
    subject: ObjectRef
    """Primary object this event concerns."""
    timestamp: datetime = field(default_factory=datetime.now)
    related: tuple[ObjectRef, ...] = ()
    """Other objects involved in the event."""
    correlation_id: str = ""
    """Client correlation ID active at the time the event was emitted."""


class EventLog:
    """Append-only event log with querying and optional subscriber callbacks.

    When *max_size* is set, the oldest event is evicted when the limit is
    reached so memory use stays bounded.
    """

    def __init__(self, max_size: int | None = 100) -> None:
        """Initialise the log with an optional maximum size."""
        self._events: list[Event] = []
        self._max_size: int | None = max_size
        self._handlers: list[Callable[[Event], None]] = []

    def emit(self, event: Event) -> None:
        """Record an event, evicting the oldest entry if *max_size* is reached."""
        if self._max_size is not None and len(self._events) >= self._max_size:
            _ = self._events.pop(0)
        self._events.append(event)
        for handler in self._handlers:
            handler(event)

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        """Register a callback invoked for every new event."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[Event], None]) -> None:
        """Remove a previously registered callback."""
        self._handlers.remove(handler)

    def get_all(self) -> list[Event]:
        """Return all recorded events."""
        return list(self._events)

    def get_for(self, ref: ObjectRef) -> list[Event]:
        """Return all events where *ref* is the subject or an involved object."""
        return [e for e in self._events if e.subject == ref or ref in e.related]

    def get_by_kind(self, kind: EventKind) -> list[Event]:
        """Return all events of the given kind."""
        return [e for e in self._events if e.kind == kind]

    def clear(self) -> None:
        """Remove all recorded events."""
        self._events.clear()

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)

    @override
    def __repr__(self) -> str:
        """Return a developer-readable representation."""
        return f"EventLog(len={len(self._events)}, max_size={self._max_size})"
