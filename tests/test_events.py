from __future__ import annotations

import datetime

import pytest

from mreg_api.events import Event
from mreg_api.events import EventKind
from mreg_api.events import EventLog
from mreg_api.events import ObjectRef
from mreg_api.models import CNAME
from mreg_api.models import MX
from mreg_api.models import HInfo
from mreg_api.models import Host
from mreg_api.models import Network
from mreg_api.models.abstracts import APIMixin
from mreg_api.models.fields import HostName


def test_eventlog_subscription():
    """Test that event log subscription with a single handler."""
    log = EventLog()
    events_received: list[Event] = []

    def subscriber(event: Event):
        events_received.append(event)

    log.subscribe(subscriber)

    test_event = Event(
        kind=EventKind.INFO,
        message="Test event",
        subject=ObjectRef(type="TestModel", id="1"),
        related=(),
        correlation_id="test-correlation-id",
    )
    log.record(test_event)

    assert len(events_received) == 1
    assert events_received[0] == test_event


def test_eventlog_subscription_multiple_handlers():
    """Test event log subscription with multiple handlers."""
    log = EventLog()
    events_received: list[Event] = []

    def subscriber(event: Event) -> None:
        events_received.append(event)

    def another_subscriber(event: Event) -> None:
        events_received.append(event)

    log.subscribe(subscriber)
    log.subscribe(another_subscriber)

    test_event = Event(
        kind=EventKind.INFO,
        message="Test event",
        subject=ObjectRef(type="TestModel", id="1"),
        related=(),
        correlation_id="test-correlation-id",
    )
    log.record(test_event)

    assert len(events_received) == 2
    assert events_received[0] == test_event
    assert events_received[1] == test_event

    # Unsubscribe the first handler and record another event
    log.unsubscribe(subscriber)
    another_test_event = Event(
        kind=EventKind.INFO,
        message="Another test event",
        subject=ObjectRef(type="TestModel", id="1"),
        related=(),
        correlation_id="test-correlation-id-2",
    )
    log.record(another_test_event)

    assert len(events_received) == 3
    assert events_received[2] == another_test_event


def test_eventlog_clear():
    """Test clearing of the event log."""
    log = EventLog()

    test_event1 = Event(
        kind=EventKind.INFO,
        message="Test event 1",
        subject=ObjectRef(type="TestModel", id="1"),
        related=(),
        correlation_id="test-correlation-id-1",
    )
    test_event2 = Event(
        kind=EventKind.INFO,
        message="Test event 2",
        subject=ObjectRef(type="TestModel", id="2"),
        related=(),
        correlation_id="test-correlation-id-2",
    )
    log.record(test_event1)
    log.record(test_event2)

    assert len(log) == 2

    log.clear()
    assert len(log) == 0
    assert log.get_all() == []


def test_eventlog_max_size() -> None:
    """Test eviction of events when max size is exceeded."""
    log = EventLog(max_size=2)

    event1 = Event(kind=EventKind.INFO, message="Test event 1", subject=ObjectRef("TestModel", id="1"))
    event2 = Event(kind=EventKind.INFO, message="Test event 2", subject=ObjectRef("TestModel", id="2"))
    event3 = Event(kind=EventKind.INFO, message="Test event 3", subject=ObjectRef("TestModel", id="3"))
    for event in [event1, event2, event3]:
        log.record(event)

    all_events = log.get_all()
    assert len(all_events) == 2
    assert all_events[0] is event2
    assert all_events[1] is event3


def test_eventlog_get_for() -> None:
    """Test retrieval of events for a specific object reference."""
    log = EventLog()

    ref1 = ObjectRef(type="TestModel", id="1")
    ref2 = ObjectRef(type="TestModel", id="2")

    event1 = Event(kind=EventKind.INFO, message="Event for ref1", subject=ref1)
    event2 = Event(kind=EventKind.INFO, message="Event for ref2", subject=ref2)
    event3 = Event(
        kind=EventKind.INFO, message="Event involving ref1 and ref2", subject=ref1, related=(ref2,)
    )
    for event in [event1, event2, event3]:
        log.record(event)

    events_for_ref1 = log.get_for(ref1)
    assert len(events_for_ref1) == 2
    assert event1 in events_for_ref1
    assert event3 in events_for_ref1

    events_for_ref2 = log.get_for(ref2)
    assert len(events_for_ref2) == 2
    assert event2 in events_for_ref2
    assert event3 in events_for_ref2


def test_get_by_kind() -> None:
    """Test retrieval of events by kind."""
    log = EventLog()

    event1 = Event(kind=EventKind.INFO, message="Info event", subject=ObjectRef("TestModel", id="1"))
    event2 = Event(
        kind=EventKind.WARNING, message="Warning event", subject=ObjectRef("TestModel", id="1")
    )
    event3 = Event(
        kind=EventKind.INFO, message="Another info event", subject=ObjectRef("TestModel", id="1")
    )
    event4 = Event(
        kind=EventKind.RESOLUTION, message="A resolution event", subject=ObjectRef("TestModel", id="1")
    )
    event5 = Event(
        kind=EventKind.MUTATION, message="A mutation event", subject=ObjectRef("TestModel", id="1")
    )
    for event in [event1, event2, event3, event4, event5]:
        log.record(event)

    info_events = log.get_by_kind(EventKind.INFO)
    assert len(info_events) == 2
    assert event1 in info_events
    assert event3 in info_events

    warning_events = log.get_by_kind(EventKind.WARNING)
    assert len(warning_events) == 1
    assert event2 in warning_events

    resolution_events = log.get_by_kind(EventKind.RESOLUTION)
    assert len(resolution_events) == 1
    assert event4 in resolution_events

    mutation_events = log.get_by_kind(EventKind.MUTATION)
    assert len(mutation_events) == 1
    assert event5 in mutation_events


def test_objectref_equality():
    ref1 = ObjectRef(type="model", id="123")
    ref2 = ObjectRef(type="model", id="123")
    assert ref1 == ref2

    ref3 = ObjectRef(type="model", id="456")
    assert ref1 != ref3


@pytest.mark.parametrize(
    "obj,id_",
    [
        pytest.param(
            Host(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                id=123,
                name=HostName("example.com"),
                comment="Test host",
                ipaddresses=[],
            ),
            "example.com",
            id="Host",
        ),
        pytest.param(
            Network(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                id=1,
                network="192.168.0.0/24",
                excluded_ranges=[],
                description="Test network",
                dns_delegated=False,
                category="Test category",
                location="Test location",
                frozen=False,
                reserved=0,
            ),
            "192.168.0.0/24",
            id="Network",
        ),
        pytest.param(
            HInfo(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                host=123,
                cpu="x86_64",
                os="Linux",
            ),
            "123",
            id="HInfo",
        ),
        pytest.param(
            CNAME(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                host=123,
                id=1,
                name=HostName("alias.example.com"),
            ),
            "alias.example.com",
            id="CNAME",
        ),
        pytest.param(
            MX(
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                host=123,
                id=1,
                priority=10,
                mx="mail.example.com",
            ),
            "1",
            id="MX",
        ),
    ],
)
def test_objectref_new(obj: APIMixin, id_: str):
    ref = ObjectRef.new(obj)
    assert ref.type == obj.__class__.__name__

    # The inferred ID field is correct and the value matches the object's ID
    assert ref.id == str(getattr(obj, obj.endpoint().external_id_field()))
    assert ref.id == id_
