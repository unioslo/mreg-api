"""Event log for tracking operations and resolutions in the MREG API."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from typing import Self

from typing_extensions import override

if TYPE_CHECKING:
    from mreg_api.models.abstracts import APIMixin

logger = logging.getLogger(__name__)


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

    @classmethod
    def new(cls, obj: APIMixin) -> Self:
        """Create a reference to an object with an ID attribute.

        Cannot fail. Logs and defaults to str(obj) for id on failure.
        """
        try:
            id_field = obj.endpoint().external_id_field()
            id_val = str(getattr(obj, id_field))  # pyright: ignore[reportAny]
            return cls(type=obj.__class__.__name__, id=id_val)
        except Exception:
            logger.exception("Failed to instantiate ObjectRef from %s", obj)
            return cls(type=obj.__class__.__name__, id=str(obj))

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


EventHandler = Callable[[Event], None]
"""Callable invoked whenever an event is recorded to an EventLog."""


class EventLog:
    """Append-only event log with querying and optional subscriber callbacks.

    When *max_size* is set, the oldest event is evicted when the limit is
    reached so memory use stays bounded.
    """

    def __init__(self, max_size: int | None = 100) -> None:
        """Initialise the log with an optional maximum size."""
        self._events: deque[Event] = deque(maxlen=max_size)
        self._handlers: list[EventHandler] = []

    def record(self, event: Event) -> None:
        """Record an event, evicting the oldest entry if *max_size* is reached.

        Calls all registered handlers with the new event as the argument.
        """
        self._events.append(event)
        for handler in self._handlers:
            handler(event)

    def subscribe(self, handler: EventHandler) -> None:
        """Register a callback invoked for every new event."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
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
        return f"EventLog(len={len(self._events)}, max_size={self._events.maxlen})"
