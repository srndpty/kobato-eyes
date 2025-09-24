from tagger.labels_util import TagMeta, sort_by_popularity
from ui.autocomplete import abbreviate_count


def _filter(prefix: str, candidates: list[TagMeta]) -> list[str]:
    matches = [tag for tag in candidates if tag.name.lower().startswith(prefix.lower())]
    ranked = sort_by_popularity(matches)
    return [tag.name for tag in ranked[:50]]


def test_autocomplete_prefers_popular_tags() -> None:
    candidates = [
        TagMeta(name="a", category=0, count=100),
        TagMeta(name="ab", category=0, count=2),
        TagMeta(name="abc", category=0, count=50),
    ]
    assert _filter("a", candidates) == ["a", "abc", "ab"]


def test_delegate_formats_popularity_counts() -> None:
    assert abbreviate_count(0) == ""
    assert abbreviate_count(12) == "12"
    assert abbreviate_count(1_400) == "1.4k"
    assert abbreviate_count(2_100_000) == "2.1M"
    assert abbreviate_count(None) == ""
