"""Input widget with tab-based autocomplete."""

from __future__ import annotations

from typing import Iterable, Callable

from textual.widgets import Input
from textual.suggester import SuggestFromList
from textual import events


class TabAutocompleteInput(Input):
    """Input widget that accepts suggestions with the Tab key and keeps history."""

    def __init__(
        self,
        *args,
        suggestions: Iterable[str] | Callable[[str], Iterable[str]] | None = None,
        **kwargs,
    ) -> None:
        if callable(suggestions):
            self._suggestions_fn: Callable[[str], Iterable[str]] | None = suggestions
            suggester = SuggestFromList([])
        else:
            self._suggestions_fn = None
            suggester = SuggestFromList(suggestions or [])

        super().__init__(*args, suggester=suggester, **kwargs)
        self._history: list[str] = []
        self._hist_idx: int = 0

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "tab":
            # accept completion suggestion if present
            if self._suggestion:
                self.action_cursor_right()
            event.stop()
            return
        if event.key == "up":
            if self._history:
                if self._hist_idx > 0:
                    self._hist_idx -= 1
                self.value = self._history[self._hist_idx]
                self.cursor_position = len(self.value)
            event.stop()
            return
        if event.key == "down":
            if self._history:
                if self._hist_idx < len(self._history) - 1:
                    self._hist_idx += 1
                    self.value = self._history[self._hist_idx]
                else:
                    self._hist_idx = len(self._history)
                    self.value = ""
                self.cursor_position = len(self.value)
            event.stop()
            return
        if event.key == "enter":
            if self.value:
                self._history.append(self.value)
                self._hist_idx = len(self._history)
        await super()._on_key(event)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._suggestions_fn:
            items = list(self._suggestions_fn(event.value))
            self.suggester._suggestions = items
            self.suggester._for_comparison = (
                items
                if self.suggester.case_sensitive
                else [s.casefold() for s in items]
            )


__all__ = ["TabAutocompleteInput"]
