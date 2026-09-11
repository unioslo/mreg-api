from __future__ import annotations

import logging

import pytest
from inline_snapshot import snapshot

from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.models.history import HistoryUpdate


def test_history_item_message_failure_falback(caplog: pytest.LogCaptureFixture) -> None:
    """Test that HistoryItem.message falls back to a raw string representation of the data if rendering fails."""
    caplog.set_level(logging.WARNING)

    # Create a HistoryItem that is valid initially
    item = HistoryItem(
        id=1,
        mid=123,
        timestamp="2024-01-01T00:00:00Z",  # pyright: ignore[reportArgumentType] # validator handles this
        user="testuser",
        resource=HistoryResource.Host,
        name="testhost",
        model="Host",
        action="update",
        data=HistoryUpdate(
            current_data={"name": "oldname", "note": "this will be replaced"},
            update={"name": "newname"},
        ).model_dump(),
    )

    # Render the valid message
    assert item.message == snapshot("name: oldname -> newname")
    assert len(caplog.records) == snapshot(0)  # No warnings should be logged for valid data

    # Now make the data invalid to trigger the fallback
    item.data = ["i am not valid anymore!"]  # pyright: ignore[reportAttributeAccessIssue]
    assert item.message == snapshot("['i am not valid anymore!']")
    assert len(caplog.records) == snapshot(1)  # A warning should be logged for invalid data
    assert caplog.records[0].message == snapshot(
        "Failed to render history item id=1 action=update; falling back to raw data"
    )
    assert caplog.records[0].exc_info is not None  # we have exc info for consumers
