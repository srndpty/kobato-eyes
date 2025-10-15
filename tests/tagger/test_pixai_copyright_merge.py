from pathlib import Path

import pytest

from tagger.base import TagCategory
from tagger.labels_util import TagMeta, load_selected_tags
from tagger.pixai_onnx import PixaiOnnxTagger


def _make_tagger(meta: dict[str, TagMeta]) -> PixaiOnnxTagger:
    tagger = PixaiOnnxTagger.__new__(PixaiOnnxTagger)
    tagger._tag_meta_index = meta  # type: ignore[attr-defined]
    return tagger  # type: ignore[return-value]


def test_resolve_output_names_prefers_prediction() -> None:
    tagger = PixaiOnnxTagger.__new__(PixaiOnnxTagger)
    selected = tagger._resolve_output_names(["embedding", "logits", "prediction"])
    assert selected == ["prediction"]


def test_resolve_output_names_falls_back_to_logits() -> None:
    tagger = PixaiOnnxTagger.__new__(PixaiOnnxTagger)
    selected = tagger._resolve_output_names(["embedding", "logits"])
    assert selected == ["logits"]


def test_merge_adds_missing_copyright_and_uses_max_score() -> None:
    meta = {
        "character_a": TagMeta(name="character_a", category=1, count=0, ips=("ip1", "ip2")),
    }
    tagger = _make_tagger(meta)
    merged = tagger._merge_copyrights(  # type: ignore[attr-defined]
        {
            "character_a": (0.83, TagCategory.CHARACTER),
            "ip2": (0.60, TagCategory.COPYRIGHT),
        }
    )
    assert pytest.approx(0.83) == merged["ip1"][0]
    assert merged["ip1"][1] == int(TagCategory.COPYRIGHT)
    assert pytest.approx(0.83) == merged["ip2"][0]
    assert merged["ip2"][1] == int(TagCategory.COPYRIGHT)


def test_merge_aggregates_multiple_characters() -> None:
    meta = {
        "character_b": TagMeta(name="character_b", category=1, count=0, ips=("ipx",)),
        "character_c": TagMeta(name="character_c", category=1, count=0, ips=("ipx",)),
    }
    tagger = _make_tagger(meta)
    merged = tagger._merge_copyrights(  # type: ignore[attr-defined]
        {
            "character_b": (0.5, TagCategory.CHARACTER),
            "character_c": (0.9, TagCategory.CHARACTER),
        }
    )
    assert pytest.approx(0.9) == merged["ipx"][0]
    assert merged["ipx"][1] == int(TagCategory.COPYRIGHT)


def test_merge_ignores_characters_without_ips() -> None:
    meta = {
        "character_d": TagMeta(name="character_d", category=1, count=0, ips=()),
    }
    tagger = _make_tagger(meta)
    original = {
        "character_d": (0.4, TagCategory.CHARACTER),
    }
    merged = tagger._merge_copyrights(original)  # type: ignore[attr-defined]
    assert merged == {"character_d": (0.4, int(TagCategory.CHARACTER))}


def test_load_selected_tags_parses_pixai_ips(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        """id,tag_id,name,category,count,ips
0,1,char_empty,1,10,[]
1,2,char_plain,1,20,"[""ip-one""]"
2,3,char_escaped,1,30,"[""ip-a"", ""ip-b""]"
""",
        encoding="utf-8",
    )
    tags = load_selected_tags(csv_path)
    meta = {tag.name: tag for tag in tags}
    assert meta["char_empty"].ips == ()
    assert meta["char_plain"].ips == ("ip-one",)
    assert meta["char_escaped"].ips == ("ip-a", "ip-b")
