"""Input widget with tab-based autocomplete."""

from __future__ import annotations

from typing import Iterable

from textual.widgets import Input
from textual.suggester import SuggestFromList
from textual import events


class TabAutocompleteInput(Input):
    """Input widget that accepts suggestions with the Tab key."""

    def __init__(
        self, *args, suggestions: Iterable[str] | None = None, **kwargs
    ) -> None:
        suggester = SuggestFromList(suggestions) if suggestions else None
        super().__init__(*args, suggester=suggester, **kwargs)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "tab":
            # accept completion suggestion if present
            if self._suggestion:
                self.action_cursor_right()
            event.stop()
            return
        await super()._on_key(event)


__all__ = ["TabAutocompleteInput"]
