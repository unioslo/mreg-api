"""Event log for tracking operations and resolutions in the MREG API."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import IntEnum
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Self

from typing_extensions import override

if TYPE_CHECKING:
    from mreg_api.models.abstracts import APIMixin

logger = logging.getLogger(__name__)


class EventKind(StrEnum):
    """Classification of an event."""

    RESOLUTION = "resolution"
    """Object found via an alias or redirect (e.g. CNAME, PTR override)."""

    MUTATION = "mutation"
    """Object was created, patched, or deleted."""

    NOTICE = "notice"
    """General informational event."""


class EventLevel(IntEnum):
    """Severity level of an event.

    Handlers can use this to filter which events to react to.
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass(frozen=True)
class ObjectRef:
    """Lightweight reference to an MREG API object, used for event correlation.

    `type` is the model class name (e.g. `"Host"`, `"MX"`).
    `value` is always a string. Field values are converted to strings.
    `field` is the specific field within the object that the value pertains to.
    """

    type: str
    value: str
    field: str = "id"

    @classmethod
    def new(cls, obj: APIMixin) -> Self:
        """Create a reference to an API object.

        Uses the object's endpoint to determine the ID field to use.

        Cannot fail. Logs and defaults to str(obj) for value on failure.
        """
        try:
            id_field = obj.endpoint().external_id_field()
            id_val = str(getattr(obj, id_field))  # pyright: ignore[reportAny]
            return cls(type=obj.__class__.__name__, value=id_val, field=id_field)
        except Exception:
            logger.exception("Failed to instantiate ObjectRef from %s", obj)
            return cls(type=obj.__class__.__name__, value=str(obj))

    @override
    def __str__(self) -> str:
        """Return a human-readable representation of the reference."""
        return f"{self.type}({self.field}={self.value!r})"


@dataclass(frozen=True)
class Event:
    """Structured record of something that happened to or around an MREG API object."""

    kind: EventKind
    message: str
    """Human-readable description of the event."""
    subject: ObjectRef
    level: EventLevel = EventLevel.INFO
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

    def get_by_level(self, level: EventLevel) -> list[Event]:
        """Return all events with exactly the given level."""
        return [e for e in self._events if e.level == level]

    def get_at_or_above(self, level: EventLevel) -> list[Event]:
        """Return all events at or above the given severity level."""
        return [e for e in self._events if e.level >= level]

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
