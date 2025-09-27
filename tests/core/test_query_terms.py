import pytest

from core.query import extract_positive_tag_terms


@pytest.mark.parametrize(
    "query,expected",
    [
        ("", []),
        ("haruhi", ["haruhi"]),
        ("Haruhi", ["haruhi"]),
        ("haruhi OR miku", ["haruhi", "miku"]),
        ("a AND (b OR NOT c)", ["a", "b"]),
        ("megurine_luka -hatsune_miku", ["megurine_luka"]),
        ("megurine_luka NOT hatsune_miku", ["megurine_luka"]),
        ("NOT haruhi", []),
        ("character: AND miku", ["miku"]),
        ("miku AND NOT (rin OR NOT luka)", ["miku", "luka"]),
        ("kobato'", ["kobato'"]),
        ("miku miku", ["miku"]),
    ],
)
def test_extract_positive_tag_terms(query: str, expected: list[str]) -> None:
    """Positive tags should be collected in sorted, lower-cased order."""

    assert extract_positive_tag_terms(query) == expected
