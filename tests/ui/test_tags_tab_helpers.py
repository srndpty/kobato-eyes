"""Unit tests for helper functions in ``ui.tags_tab``."""

from __future__ import annotations

import pytest

from tagger.base import TagCategory
from tagger.labels_util import TagMeta
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
    assert str(TagsTab._coerce_result_path("C:/images/a.png")) == "C:\\images\\a.png"


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
