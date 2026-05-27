"""Search execution and paging behavior for TagsTab."""

from __future__ import annotations

import sqlite3

from PyQt6.QtCore import QThread

from ui.search_worker import SearchWorker


class TagsSearchMixin:
    """Run asynchronous searches and update paging state."""

    @property
    def _offset(self) -> int:
        return self._search_state.offset

    @_offset.setter
    def _offset(self, value: int) -> None:
        self._search_state.offset = max(0, int(value))

    @property
    def _search_busy(self) -> bool:
        return self._search_state.busy

    @_search_busy.setter
    def _search_busy(self, value: bool) -> None:
        self._search_state.busy = bool(value)

    @property
    def _can_load_more(self) -> bool:
        return self._search_state.can_load_more

    @_can_load_more.setter
    def _can_load_more(self, value: bool) -> None:
        self._search_state.can_load_more = bool(value)

    @property
    def _last_search_cancelled(self) -> bool:
        return self._search_state.last_cancelled

    @_last_search_cancelled.setter
    def _last_search_cancelled(self, value: bool) -> None:
        self._search_state.last_cancelled = bool(value)

    def _set_busy(self, busy: bool) -> None:
        self._search_busy = busy
        if busy:
            self._can_load_more = False
        self._update_control_states()

    def _cancel_active_search(self) -> None:
        worker = self._search_worker
        if worker is not None:
            worker.cancel()

    def _on_search_clicked(self) -> None:
        query = self._query_edit.text().strip()
        positive_terms = self._view_model.extract_positive_terms(query) if query else []
        self._highlight_terms = list(positive_terms)
        self._positive_terms = positive_terms
        self._use_relevance = bool(positive_terms)
        if self._conn is not None:
            try:
                self._relevance_thresholds = self._view_model.load_tag_thresholds(self._conn)
            except sqlite3.Error:
                self._relevance_thresholds = {}
        else:
            self._relevance_thresholds = {}
        thresholds = {int(category): float(value) for category, value in (self._tag_thresholds or {}).items()}
        try:
            fragment = self._view_model.translate_query(
                query,
                file_alias="f",
                thresholds=thresholds,
            )
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._set_busy(False)
            self._search_overlay.hide()
            return

        order_clause = "relevance DESC, f.mtime DESC" if self._use_relevance else "f.mtime DESC"
        terms_text = ", ".join(self._positive_terms)
        self._debug_where.setText(f"WHERE: {fragment.where}\nORDER: {order_clause}")
        self._debug_params.setText(f"Params: {fragment.params}\nRelevance terms: [{terms_text}]")
        self._debug_group.setVisible(bool(fragment.where.strip() and fragment.where.strip() != "1=1"))

        self._current_query = query
        self._current_where = fragment.where
        self._current_params = list(fragment.params)
        self._search_state.begin_query()
        self._status_label.setText("Searching...")
        self._search_overlay.show("Searching... (Esc to cancel)")
        self._set_busy(True)
        self._start_async_search(reset=True)

    def _on_load_more_clicked(self) -> None:
        if not self._current_where or self._search_busy:
            return
        self._status_label.setText("Searching...")
        self._search_overlay.show("Searching... (Esc to cancel)")
        self._set_busy(True)
        self._start_async_search(reset=False)

    def _start_async_search(self, *, reset: bool) -> None:
        if not self._current_where:
            self._status_label.setText("Enter a query to search tags.")
            self._search_overlay.hide()
            self._set_busy(False)
            self._show_placeholder(True)
            return
        if self._db_path is None:
            self._db_path = self._resolve_db_path()
        if self._db_path is None:
            self._status_label.setText("Database path unavailable.")
            self._search_overlay.hide()
            self._set_busy(False)
            return

        self._cancel_active_search()
        generation = self._search_state.begin_worker(reset=reset)

        offset = 0 if reset else max(0, int(self._offset))
        thresholds = self._relevance_thresholds if self._relevance_thresholds else None
        tags_for_relevance = tuple(self._positive_terms) if self._use_relevance else tuple()

        worker = SearchWorker(
            self._db_path,
            self._current_where or "1=1",
            tuple(self._current_params),
            tags_for_relevance=tags_for_relevance,
            thresholds=thresholds,
            order="relevance" if self._use_relevance else "mtime",
            chunk=self._search_chunk_size,
            offset=offset,
            max_rows=self._search_chunk_size,
            chunk_delay=self._search_chunk_delay,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        self._search_worker = worker
        self._search_thread = thread

        worker.chunkReady.connect(lambda rows, g=generation: self._handle_search_chunk(rows, g))
        worker.finished.connect(
            lambda success, cancelled, g=generation: self._handle_search_finished(success, cancelled, g)
        )
        worker.error.connect(lambda message, g=generation: self._handle_search_error(message, g))

        thread.started.connect(worker.run)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _handle_search_chunk(self, rows: list[dict[str, object]], generation: int) -> None:
        if generation != self._search_state.generation:
            return
        reset_results = self._search_state.reset_pending
        if self._search_state.reset_pending:
            self._clear_results_for_new_search()
            self._search_state.reset_pending = False
        if not rows:
            return
        self._append_rows(rows)
        if reset_results:
            self._scroll_results_to_top()
        self._search_state.consume_rows(len(rows), chunk_size=self._search_chunk_size)
        query_label = self._current_query or "*"
        self._status_label.setText(f"Showing {self._offset} result(s) for '{query_label}'")
        self._show_placeholder(False)

    def _handle_search_error(self, message: str, generation: int) -> None:
        self._search_state.discard_generation(generation)
        if generation != self._search_state.generation:
            return
        self._search_overlay.hide()
        self._set_busy(False)
        self._can_load_more = False
        self._status_label.setText(f"Search failed: {message}")
        self._update_control_states()

    def _handle_search_finished(self, success: bool, cancelled: bool, generation: int) -> None:
        was_reset = self._search_state.finish_generation(generation)
        if generation != self._search_state.generation:
            return
        self._search_worker = None
        self._search_thread = None
        self._search_overlay.hide()
        self._set_busy(False)
        self._last_search_cancelled = bool(cancelled)
        if cancelled:
            self._status_label.setText("Search cancelled.")
            self._can_load_more = False
        elif success and not self._search_state.received_any:
            if self._search_state.reset_pending or was_reset:
                self._clear_results_for_new_search()
                self._search_state.reset_pending = False
                self._status_label.setText("No results. Try indexing your library.")
                self._show_placeholder(True)
                self._can_load_more = False
            else:
                query_label = self._current_query or "*"
                self._status_label.setText(f"Showing {self._offset} result(s) for '{query_label}'")
                self._can_load_more = False
        elif success:
            query_label = self._current_query or "*"
            self._status_label.setText(f"Showing {self._offset} result(s) for '{query_label}'")
        else:
            self._can_load_more = False
        self._update_control_states()
