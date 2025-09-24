from tagger.labels_util import TagMeta, sort_by_popularity


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
