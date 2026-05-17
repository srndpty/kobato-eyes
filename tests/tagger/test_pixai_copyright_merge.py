from pathlib import Path

import numpy as np
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


def test_pixai_tag_metadata_parse_failure_uses_empty_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    labels_csv = tmp_path / "selected_tags.csv"
    labels_csv.write_text("broken", encoding="utf-8")

    def fail_load_selected_tags(path: Path):  # noqa: ANN001
        raise ValueError(f"bad metadata: {path.name}")

    monkeypatch.setattr("tagger.pixai_onnx.load_selected_tags", fail_load_selected_tags)
    with caplog.at_level("WARNING", logger="tagger.pixai_onnx"):
        index = PixaiOnnxTagger._build_tag_meta_index(labels_csv)

    assert index == {}
    assert "failed to parse tag metadata" in caplog.text
    assert "bad metadata" in caplog.text


def test_pixai_postprocess_reports_output_label_mismatch() -> None:
    tagger = PixaiOnnxTagger.__new__(PixaiOnnxTagger)
    tagger._default_thresholds = {}  # type: ignore[attr-defined]
    tagger._default_max_tags = {}  # type: ignore[attr-defined]
    tagger._score_floor = 0.0  # type: ignore[attr-defined]
    tagger._label_name_cache = ["tag_a", "tag_b"]  # type: ignore[attr-defined]
    tagger._effective_cats = [int(TagCategory.GENERAL), int(TagCategory.GENERAL)]  # type: ignore[attr-defined]
    tagger._tag_meta_index = {}  # type: ignore[attr-defined]
    tagger._topk_cap = 128  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="PixAI: model output dimension 3 does not match label count 2"):
        tagger._postprocess_logits_topk(  # type: ignore[attr-defined]
            np.zeros((1, 3), dtype=np.float32),
            thresholds=None,
            max_tags=None,
        )


def test_pixai_postprocess_collects_candidates_per_category_before_limits() -> None:
    tagger = PixaiOnnxTagger.__new__(PixaiOnnxTagger)
    names = [f"general_{idx}" for idx in range(128)] + ["character_a"]
    cats = [int(TagCategory.GENERAL)] * 128 + [int(TagCategory.CHARACTER)]
    tagger._default_thresholds = {}  # type: ignore[attr-defined]
    tagger._default_max_tags = {}  # type: ignore[attr-defined]
    tagger._score_floor = 0.0  # type: ignore[attr-defined]
    tagger._label_name_cache = names  # type: ignore[attr-defined]
    tagger._effective_cats = cats  # type: ignore[attr-defined]
    tagger._pixai_label_names = np.array(names, dtype=object)  # type: ignore[attr-defined]
    tagger._pixai_effective_cats = np.array(cats, dtype=np.int16)  # type: ignore[attr-defined]
    tagger._pixai_cat_to_idx = {  # type: ignore[attr-defined]
        int(TagCategory.GENERAL): np.arange(0, 128),
        int(TagCategory.CHARACTER): np.array([128]),
    }
    tagger._pixai_default_thr_vec = np.zeros((129,), dtype=np.float32)  # type: ignore[attr-defined]
    tagger._pixai_name_to_idx = {name: idx for idx, name in enumerate(names)}  # type: ignore[attr-defined]
    tagger._tag_meta_index = {}  # type: ignore[attr-defined]
    tagger._topk_cap = 128  # type: ignore[attr-defined]

    logits = np.linspace(1.0, 0.5, num=129, dtype=np.float32).reshape(1, 129)
    results = tagger._postprocess_logits_topk(  # type: ignore[attr-defined]
        logits,
        thresholds={TagCategory.GENERAL: 0.1, TagCategory.CHARACTER: 0.1},
        max_tags={TagCategory.GENERAL: 1, TagCategory.CHARACTER: 1},
    )

    assert [prediction.name for prediction in results[0].tags] == ["general_0", "character_a"]
