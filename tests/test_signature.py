from core.pipeline.signature import _build_max_tags_map, _build_threshold_map, current_tagger_sig
from core.settings import PipelineSettings
from tagger.base import TagCategory


def _settings(base_thr=None):
    s = PipelineSettings()
    s.tagger.name = "wd14-onnx"
    s.tagger.model_path = "C:/models/wd14.onnx"
    s.tagger.tags_csv = "C:/models/tags.csv"
    s.tagger.thresholds = base_thr or {"general": 0.35, "character": 0.25, "copyright": 0.25}
    return s


def test_signature_changes_when_thresholds_change():
    s1 = _settings()
    sig1 = current_tagger_sig(s1)
    s2 = _settings({"general": 0.40, "character": 0.25, "copyright": 0.25})
    sig2 = current_tagger_sig(s2)
    assert sig1 != sig2


def test_build_maps_keys():
    thr = _build_threshold_map({"general": 0.3, "character": 0.9})
    assert TagCategory.GENERAL in thr and TagCategory.CHARACTER in thr
    mx = _build_max_tags_map({"general": 25})
    assert TagCategory.GENERAL in mx
