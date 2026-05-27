"""Unit tests for helper functions in ``ui.tags_tab``."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel

from tagger.base import TagCategory
from tagger.labels_util import TagMeta
from ui import tags_autocomplete
from ui.tags_autocomplete import TagListModel, TagsAutocompleteMixin
from ui.tags_results import TagsResultsMixin
from ui.tags_tab import TagsTab, _filter_tags_by_threshold


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [{"name": "cute", "score": "0.15"}],
            [("cute", 0.15, None)],
            id="dict-input",
        ),
        pytest.param(
            [("tuple", 0.2)],
            [("tuple", 0.2, None)],
            id="tuple-two",
        ),
        pytest.param(
            [("triple", 0.3, TagCategory.GENERAL)],
            [("triple", 0.3, TagCategory.GENERAL)],
            id="tuple-three",
        ),
        pytest.param(
            [("too", 0.2, "many", "values")],
            [],
            id="invalid-length",
        ),
        pytest.param(
            [("low", 0.05), ("edge", 0.1)],
            [("edge", 0.1, None)],
            id="threshold-boundary",
        ),
        pytest.param(
            [(123, 0.25)],
            [("123", 0.25, None)],
            id="name-coercion",
        ),
        pytest.param(
            [("str-cat", 0.2, "character")],
            [("str-cat", 0.2, TagCategory.CHARACTER)],
            id="string-category",
        ),
    ],
)
def test_filter_tags_by_threshold(rows, expected) -> None:
    assert _filter_tags_by_threshold(rows) == expected


def test_coerce_result_path_rejects_empty_or_non_string_values() -> None:
    assert TagsTab._coerce_result_path("") is None
    assert TagsTab._coerce_result_path("   ") is None
    assert TagsTab._coerce_result_path(None) is None
    path = TagsTab._coerce_result_path("images/a.png")
    assert path is not None
    assert path.parts[-2:] == ("images", "a.png")


def test_tag_list_model_exposes_display_name_and_count_roles() -> None:
    model = TagListModel([TagMeta(name="blue_eyes", category=0, count=1234)])
    index = model.index(0, 0)

    assert model.rowCount() == 1
    assert index.data() == "blue_eyes (1.23k)"
    assert index.data(int(TagListModel.NAME_ROLE)) == "blue_eyes"
    assert index.data(int(TagListModel.COUNT_ROLE)) == 1234

    model.reset_with([TagMeta(name="red_hair", category=0, count="bad")], display_prefix="-")
    index = model.index(0, 0)
    assert index.data() == "-red_hair"
    assert index.data(int(TagListModel.COUNT_ROLE)) == 0


def test_update_completion_candidates_keeps_reserved_and_most_popular_tag() -> None:
    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._all_tags = [
                TagMeta(name="tag", category=0, count=2),
                TagMeta(name="TAG", category=0, count=7),
                TagMeta(name="other", category=0, count=1),
            ]
            self._completion_candidates = []
            self._tag_model = TagListModel()
            self.hide_calls = 0

        def _hide_completion_popup(self) -> None:
            self.hide_calls += 1

    dummy = DummyAutocomplete()

    dummy._update_completion_candidates()

    by_name = {tag.name.lower(): tag for tag in dummy._completion_candidates}
    assert "category:" in by_name
    assert by_name["tag"].count == 7
    assert by_name["other"].count == 1
    assert dummy.hide_calls == 0


def test_query_text_edited_clears_completion_when_no_candidates() -> None:
    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._completion_candidates = []
            self._tag_model = TagListModel([TagMeta(name="old", category=0)])
            self._pending_completion_text = ""
            self.hide_calls = 0

        def _hide_completion_popup(self) -> None:
            self.hide_calls += 1

    dummy = DummyAutocomplete()

    dummy._on_query_text_edited("new")

    assert dummy._pending_completion_text == "new"
    assert dummy._tag_model.rowCount() == 0
    assert dummy.hide_calls == 1


def test_refresh_completions_ranks_matches_and_updates_prefix() -> None:
    class FakeCompleter:
        def __init__(self) -> None:
            self.prefix = ""
            self.completed = 0

        def setCompletionPrefix(self, prefix: str) -> None:
            self.prefix = prefix

        def complete(self) -> None:
            self.completed += 1

    class FakeQueryEdit:
        def text(self) -> str:
            return "-ta"

        def cursorPosition(self) -> int:
            return 3

    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._completion_candidates = [
                TagMeta(name="tag_low", category=0, count=1),
                TagMeta(name="tag_high", category=0, count=100),
                TagMeta(name="other", category=0, count=1000),
            ]
            self._tag_model = TagListModel()
            self._query_edit = FakeQueryEdit()
            self._completer = FakeCompleter()
            self._current_completion_range = (0, 0)
            self.hide_calls = 0

        def _hide_completion_popup(self) -> None:
            self.hide_calls += 1

    dummy = DummyAutocomplete()

    dummy._refresh_completions()

    assert dummy._current_completion_range == (0, 3)
    assert dummy._completer.prefix == "ta"
    assert dummy._completer.completed == 1
    assert dummy._tag_model.rowCount() == 2
    assert dummy._tag_model.index(0, 0).data() == "-tag_high (100)"
    assert dummy._tag_model.index(1, 0).data() == "-tag_low (1)"
    assert dummy.hide_calls == 0


def test_refresh_completions_hides_popup_when_prefix_has_no_matches() -> None:
    class FakeQueryEdit:
        def text(self) -> str:
            return "missing"

        def cursorPosition(self) -> int:
            return 7

    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._completion_candidates = [TagMeta(name="tag", category=0, count=1)]
            self._tag_model = TagListModel([TagMeta(name="old", category=0)])
            self._query_edit = FakeQueryEdit()
            self._current_completion_range = (0, 0)
            self.hide_calls = 0

        def _hide_completion_popup(self) -> None:
            self.hide_calls += 1

    dummy = DummyAutocomplete()

    dummy._refresh_completions()

    assert dummy._current_completion_range == (0, 7)
    assert dummy._tag_model.rowCount() == 0
    assert dummy.hide_calls == 1


def test_completion_activated_falls_back_when_stored_range_is_stale() -> None:
    class FakeQueryEdit:
        def __init__(self) -> None:
            self._text = "rating:safe ta"
            self.cursor = -1
            self.blocks: list[bool] = []

        def text(self) -> str:
            return self._text

        def setText(self, text: str) -> None:
            self._text = text

        def blockSignals(self, blocked: bool) -> bool:
            self.blocks.append(blocked)
            return False

        def setCursorPosition(self, cursor: int) -> None:
            self.cursor = cursor

    class FakeTimer:
        def __init__(self) -> None:
            self.stopped = 0

        def stop(self) -> None:
            self.stopped += 1

    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._tag_model = TagListModel([TagMeta(name="tag_high", category=0, count=100)])
            self._pending_completion_text = "rating:safe ta"
            self._current_completion_range = (999, 1000)
            self._query_edit = FakeQueryEdit()
            self._autocomplete_timer = FakeTimer()
            self.hide_calls = 0

        def _hide_completion_popup(self) -> None:
            self.hide_calls += 1

    dummy = DummyAutocomplete()
    index = dummy._tag_model.index(0, 0)

    dummy._on_completion_activated(index)

    assert dummy._query_edit.text() == "rating:safe tag_high "
    assert dummy._query_edit.cursor == len("rating:safe tag_high ")
    assert dummy._pending_completion_text == "rating:safe tag_high "
    assert dummy._autocomplete_timer.stopped == 1
    assert dummy.hide_calls == 1


def test_accept_completion_uses_first_item_and_applies_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQueryEdit:
        def __init__(self) -> None:
            self._text = "-ta"
            self.cursor = -1
            self.focused = 0

        def text(self) -> str:
            return self._text

        def setText(self, text: str) -> None:
            self._text = text

        def blockSignals(self, _blocked: bool) -> bool:
            return False

        def setCursorPosition(self, cursor: int) -> None:
            self.cursor = cursor

        def setFocus(self) -> None:
            self.focused += 1

    class FakePopup:
        def currentIndex(self):
            return TagListModel().index(0, 0)

    class FakeCompleter:
        def __init__(self, model: TagListModel) -> None:
            self._model = model

        def popup(self) -> FakePopup:
            return FakePopup()

        def completionModel(self) -> TagListModel:
            return self._model

    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._tag_model = TagListModel([TagMeta(name="tag_high", category=0, count=1)])
            self._completer = FakeCompleter(self._tag_model)
            self._query_edit = FakeQueryEdit()
            self._pending_completion_text = "-ta"
            self._current_completion_range = (0, 3)
            self.hide_calls = 0

        def _hide_completion_popup(self) -> None:
            self.hide_calls += 1

    monkeypatch.setattr(tags_autocomplete.QTimer, "singleShot", lambda _delay, callback: callback())
    dummy = DummyAutocomplete()

    dummy._accept_completion(default_if_none=True)

    assert dummy._query_edit.text() == "-tag_high "
    assert dummy._query_edit.cursor == len("-tag_high ")
    assert dummy._pending_completion_text == "-tag_high "
    assert dummy.hide_calls == 1
    assert dummy._query_edit.focused == 1


def test_event_filter_accepts_completion_keys_when_popup_visible() -> None:
    class FakePopup:
        def __init__(self) -> None:
            self.visible = True

        def isVisible(self) -> bool:
            return self.visible

    class FakeCompleter:
        def __init__(self) -> None:
            self._popup = FakePopup()

        def popup(self) -> FakePopup:
            return self._popup

    class FakeEvent:
        def __init__(self, key: Qt.Key) -> None:
            self._key = key
            self.accepted = False

        def type(self) -> QEvent.Type:
            return QEvent.Type.KeyPress

        def key(self) -> Qt.Key:
            return self._key

        def accept(self) -> None:
            self.accepted = True

    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._query_edit = object()
            self._completer = FakeCompleter()
            self._suppress_return_once = False
            self.accept_calls = 0

        def _accept_completion(self, *, default_if_none: bool = False) -> None:
            assert default_if_none
            self.accept_calls += 1

    dummy = DummyAutocomplete()
    tab_event = FakeEvent(Qt.Key.Key_Tab)
    return_event = FakeEvent(Qt.Key.Key_Return)

    assert dummy.eventFilter(dummy._query_edit, tab_event) is True
    assert tab_event.accepted
    assert dummy.eventFilter(dummy._query_edit, return_event) is True
    assert return_event.accepted
    assert dummy._suppress_return_once is True
    assert dummy.accept_calls == 2


def test_reload_autocomplete_merges_csv_ip_and_db_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_tags = [
        TagMeta(name="character", category=TagCategory.CHARACTER.value, count=10, ips=["series"]),
        TagMeta(name="db_only", category=0, count=1),
    ]

    class DummyViewModel:
        def list_tag_names(self, _conn):
            return ["db_only", "new_db"]

    class DummyAutocomplete(TagsAutocompleteMixin):
        def __init__(self) -> None:
            self._view_model = DummyViewModel()
            self._conn = object()
            self._all_tags: list[TagMeta] = []
            self._completion_candidates: list[TagMeta] = []
            self._tag_model = TagListModel()
            self.refresh_calls = 0
            self.threshold_settings = []

        def _update_thresholds(self, settings) -> None:
            self.threshold_settings.append(settings)

        def _refresh_completions(self) -> None:
            self.refresh_calls += 1

    settings = type("Settings", (), {"tagger": type("Tagger", (), {"model_path": "model", "tags_csv": None})()})()
    monkeypatch.setattr(tags_autocomplete.labels_util, "discover_labels_csv", lambda *_args: Path("tags.csv"))
    monkeypatch.setattr(tags_autocomplete.labels_util, "load_selected_tags", lambda _path: list(csv_tags))
    dummy = DummyAutocomplete()

    dummy.reload_autocomplete(settings)

    by_name = {tag.name: tag for tag in dummy._all_tags}
    assert set(by_name) == {"character", "series", "db_only", "new_db"}
    assert by_name["series"].category == TagCategory.COPYRIGHT.value
    assert by_name["series"].count == 10
    assert dummy.refresh_calls == 1
    assert dummy.threshold_settings == [settings]


def test_append_rows_populates_table_grid_and_queues_thumbnail(tmp_path: Path) -> None:
    class FakeTableView:
        def __init__(self) -> None:
            self.heights: list[tuple[int, int]] = []

        def setRowHeight(self, row: int, height: int) -> None:
            self.heights.append((row, height))

    class DummyResults(TagsResultsMixin):
        def __init__(self) -> None:
            self._THUMB_SIZE = 64
            self._results_cache = []
            self._table_model = QStandardItemModel(0, 7)
            self._grid_model = QStandardItemModel()
            self._table_view = FakeTableView()
            self.queued: list[tuple[int, int, Path]] = []

        def _queue_thumbnail(self, row: int, file_id: int, path: Path) -> None:
            self.queued.append((row, file_id, path))

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    dummy = DummyResults()

    dummy._append_rows(
        [
            {
                "id": "42",
                "path": str(image_path),
                "size": 2048,
                "width": 12,
                "height": 34,
                "mtime": "bad",
                "tags": [("tag", 0.9, TagCategory.GENERAL)],
            }
        ]
    )

    assert len(dummy._results_cache) == 1
    assert dummy._table_model.rowCount() == 1
    assert dummy._grid_model.rowCount() == 1
    assert dummy._table_model.item(0, 1).text() == "image.png"
    assert dummy._table_model.item(0, 3).text() == "2.00 KiB"
    assert dummy._table_model.item(0, 4).text() == "12×34"
    assert dummy._grid_model.item(0).data(Qt.ItemDataRole.UserRole) == 0
    assert dummy._table_view.heights == [(0, 80)]
    assert dummy.queued == [(0, 42, image_path)]


def test_queue_thumbnail_skips_duplicate_pending_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakePool:
        def __init__(self) -> None:
            self.tasks = []

        def start(self, task) -> None:
            self.tasks.append(task)

    class DummyResults(TagsResultsMixin):
        def __init__(self) -> None:
            self._THUMB_SIZE = 32
            self._pending_thumbs = set()
            self._thumb_pool = FakePool()
            self._thumb_signal = object()

    monkeypatch.setattr("ui.tags_results._ThumbnailTask", lambda *args: ("task", args))
    dummy = DummyResults()
    image_path = tmp_path / "image.png"

    dummy._queue_thumbnail(1, 2, image_path)
    dummy._queue_thumbnail(1, 2, image_path)

    assert dummy._pending_thumbs == {(1, 2)}
    assert len(dummy._thumb_pool.tasks) == 1


def test_remove_results_by_file_ids_updates_models_roles_and_selection() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.cleared = 0

        def clear(self) -> None:
            self.cleared += 1

    class FakeTableView:
        def __init__(self) -> None:
            self.selected = []
            self.updated = 0

        def selectRow(self, row: int) -> None:
            self.selected.append(row)

        def viewport(self):
            return self

        def update(self) -> None:
            self.updated += 1

    class FakeGridView(FakeTableView):
        def __init__(self) -> None:
            super().__init__()
            self.current = None

        def setCurrentIndex(self, index) -> None:
            self.current = index.row()

    class DummyResults(TagsResultsMixin):
        def __init__(self) -> None:
            self._results_cache = [{"id": 1}, {"id": 2}, {"id": 3}]
            self._table_model = QStandardItemModel()
            self._grid_model = QStandardItemModel()
            for row in range(3):
                self._table_model.appendRow(QStandardItem(str(row)))
                item = QStandardItem(str(row))
                item.setData(row, Qt.ItemDataRole.UserRole)
                self._grid_model.appendRow(item)
            self._offset = 3
            self._thumb_pool = FakePool()
            self._pending_thumbs = {(0, 1)}
            self._table_view = FakeTableView()
            self._grid_view = FakeGridView()
            self.requeued = 0

        def _requeue_missing_thumbnails(self) -> None:
            self.requeued += 1

    dummy = DummyResults()

    assert dummy._remove_results_by_file_ids([2], offset_file_ids=[2]) is True

    assert dummy._results_cache == [{"id": 1}, {"id": 3}]
    assert dummy._table_model.rowCount() == 2
    assert dummy._grid_model.rowCount() == 2
    assert [dummy._grid_model.item(row).data(Qt.ItemDataRole.UserRole) for row in range(2)] == [0, 1]
    assert dummy._offset == 2
    assert dummy._thumb_pool.cleared == 1
    assert dummy._pending_thumbs == set()
    assert dummy._table_view.selected == [1]
    assert dummy._grid_view.current == 1
    assert dummy.requeued == 1


def test_selected_result_rows_reads_table_and_grid_selection() -> None:
    class FakeIndex:
        def __init__(self, row: int, *, stored=None, valid: bool = True) -> None:
            self._row = row
            self._stored = stored
            self._valid = valid

        def isValid(self) -> bool:
            return self._valid

        def row(self) -> int:
            return self._row

        def data(self, _role):
            return self._stored

    class FakeSelection:
        def __init__(self, indexes) -> None:
            self._indexes = indexes

        def selectedRows(self):
            return self._indexes

        def selectedIndexes(self):
            return self._indexes

    class FakeView:
        def __init__(self, indexes, *, current=None, focused: bool = False) -> None:
            self._selection = FakeSelection(indexes)
            self._current = current or FakeIndex(0, valid=False)
            self._focused = focused

        def selectionModel(self) -> FakeSelection:
            return self._selection

        def currentIndex(self) -> FakeIndex:
            return self._current

        def hasFocus(self) -> bool:
            return self._focused

    class FakeStack:
        def __init__(self, current) -> None:
            self._current = current

        def currentWidget(self):
            return self._current

    class DummyResults(TagsResultsMixin):
        pass

    dummy = DummyResults()
    dummy._results_cache = [{}, {}, {}, {}]
    dummy._table_view = FakeView([FakeIndex(2), FakeIndex(99), FakeIndex(1)])
    dummy._grid_view = FakeView([])
    dummy._stack = FakeStack(dummy._table_view)
    assert dummy._selected_result_rows() == [1, 2]

    dummy._grid_view = FakeView([FakeIndex(0, stored=3), FakeIndex(1, stored=99)], focused=True)
    dummy._stack = FakeStack(dummy._grid_view)
    assert dummy._selected_result_rows() == [3]


def test_results_formatters_are_stable_for_bad_and_valid_values() -> None:
    assert TagsResultsMixin._format_size(None) == "-"
    assert TagsResultsMixin._format_size(512) == "512 B"
    assert TagsResultsMixin._format_size(2048) == "2.00 KiB"
    assert TagsResultsMixin._format_size(2 * 1024 * 1024) == "2.00 MiB"
    assert TagsResultsMixin._format_dimensions("64", 32) == "64×32"
    assert TagsResultsMixin._format_dimensions("bad", 32) == "-"
    assert TagsResultsMixin._format_mtime("bad") == "-"
    assert TagsResultsMixin._format_tags([("tag", 0.987, TagCategory.GENERAL)]) == "tag (0.99)"
    assert TagsResultsMixin._format_grid_text("image.png", [("tag1", 0.9, None), ("tag2", 0.8, None)]) == (
        "image.png\ntag1, tag2"
    )


def test_filter_display_tags_uses_category_thresholds() -> None:
    class DummyResults(TagsResultsMixin):
        def __init__(self) -> None:
            self._tag_thresholds = {TagCategory.GENERAL: 0.5}

    dummy = DummyResults()

    assert dummy._filter_display_tags(
        [
            ("low", 0.49, TagCategory.GENERAL),
            ("high", "0.50", "general"),
            ("uncategorized", 0.1, None),
            ("bad-score", "nope", TagCategory.GENERAL),
        ]
    ) == [
        ("high", 0.5, TagCategory.GENERAL),
        ("uncategorized", 0.1, None),
    ]


def test_update_thresholds_ignores_unknown_or_invalid_values() -> None:
    class DummyViewModel:
        def load_settings(self):
            raise AssertionError("settings should be supplied")

    class DummyResults(TagsResultsMixin):
        def __init__(self) -> None:
            self._view_model = DummyViewModel()
            self._tag_thresholds = {}

    settings = type(
        "Settings",
        (),
        {
            "tagger": type(
                "Tagger",
                (),
                {
                    "thresholds": {
                        "general": "0.42",
                        TagCategory.CHARACTER: 0.7,
                        "unknown": 0.9,
                        "artist": "bad",
                    }
                },
            )()
        },
    )()
    dummy = DummyResults()

    dummy._update_thresholds(settings)

    assert dummy._tag_thresholds == {
        TagCategory.GENERAL: 0.42,
        TagCategory.CHARACTER: 0.7,
    }
