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

from typing_extensions import deprecated
from typing_extensions import override

if TYPE_CHECKING:
    from mreg_api.models.abstracts import MregModel


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
    def new(cls, obj: MregModel) -> Self:
        """Create a reference to an API object.

        Uses the object's endpoint to determine the ID field to use.

        Cannot fail. Logs and defaults to str(obj) for value on failure.
        """
        from mreg_api.models.models import CNAME
        from mreg_api.models.models import Atom
        from mreg_api.models.models import ForwardZone
        from mreg_api.models.models import ForwardZoneDelegation
        from mreg_api.models.models import HInfo
        from mreg_api.models.models import Host
        from mreg_api.models.models import HostGroup
        from mreg_api.models.models import Location
        from mreg_api.models.models import NameServer
        from mreg_api.models.models import Network
        from mreg_api.models.models import ReverseZone
        from mreg_api.models.models import ReverseZoneDelegation
        from mreg_api.models.models import Role

        # TODO: refactor when we have a better way to get the external ID
        # field from an object without needing the endpoint.
        MODEL_FIELDS = {
            "name": (
                Host,
                HostGroup,
                CNAME,
                ForwardZone,
                ReverseZone,
                ForwardZoneDelegation,
                ReverseZoneDelegation,
                Role,
                Atom,
                NameServer,
            ),
            "network": (Network,),
            "host": (HInfo, Location),
        }
        for field_name, models in MODEL_FIELDS.items():
            if isinstance(obj, models):
                id_value = getattr(obj, field_name)
                id_field = field_name
                break
        else:
            try:
                id_value = str(obj.id)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
                id_field = "id"
            except AttributeError:
                id_value = str(obj)
                id_field = "id"

        try:
            return cls(type=obj.__class__.__name__, value=str(id_value), field=id_field)
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

    def get(
        self,
        subject: ObjectRef | None = None,
        kind: EventKind | None = None,
        level: EventLevel | None = None,
        min_level: EventLevel | None = None,
        where: Callable[[Event], bool] | None = None,
    ) -> list[Event]:
        """Return all recorded events, optionally filtered by subject, kind, or level.

        Multiple filters can be applied at once, and only events matching
        all provided filters will be returned.

        More complex filtering can be done by providing a callable to the
        `where` parameter, which will be called with each event record,
        and should return True if it should be included in the results.

        Args:
            subject: If given, return only events where this object is the subject or involved.
            kind: If given, return only events of this kind.
            level: If given, return only events with exactly this level.
            min_level: If given, return only events with this level or higher.
            where: A predicate function that returns True if the event should be included.

        Returns:
            A list of events matching the specified filters.
        """
        where = where or (lambda _: True)
        return [
            e
            for e in self._events
            if (subject is None or e.subject == subject or subject in e.related)
            and (kind is None or e.kind == kind)
            and (level is None or e.level == level)
            and (min_level is None or e.level >= min_level)
            and where(e)
        ]

    @deprecated('Use "get()" instead')
    def get_all(self) -> list[Event]:
        """Return all recorded events."""
        return self.get()

    @deprecated('Use "get(subject=)" instead')
    def get_for(self, ref: ObjectRef) -> list[Event]:
        """Return all events where *ref* is the subject or an involved object."""
        return self.get(subject=ref)

    @deprecated('Use "get(kind=)" instead')
    def get_by_kind(self, kind: EventKind) -> list[Event]:
        """Return all events of the given kind."""
        return self.get(kind=kind)

    @deprecated('Use "get(level=)" instead')
    def get_by_level(self, level: EventLevel) -> list[Event]:
        """Return all events with exactly the given level."""
        return self.get(level=level)

    @deprecated('Use "get(min_level=)" instead')
    def get_at_or_above(self, level: EventLevel) -> list[Event]:
        """Return all events at or above the given severity level."""
        return self.get(min_level=level)

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
