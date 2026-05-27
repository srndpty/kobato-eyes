from pathlib import Path

from tagger.labels_util import BROKEN_TAG_PREFIX, TagMeta, discover_labels_csv, load_selected_tags, sort_by_popularity


def test_load_selected_tags_single_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_single.csv"
    csv_path.write_text("# comment\n1girl\n\n", encoding="utf-8")
    tags = load_selected_tags(csv_path)
    assert tags == [TagMeta(name="1girl", category=0, count=0)]


def test_load_selected_tags_two_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_two.csv"
    csv_path.write_text("123,1girl\ncharacter:kobato,character\n", encoding="utf-8")
    tags = load_selected_tags(csv_path)
    assert TagMeta(name="1girl", category=0, count=0) in tags
    assert TagMeta(name="character:kobato", category=4, count=0) in tags


def test_load_selected_tags_four_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_four.csv"
    csv_path.write_text(
        "id,tag_id,name,category,count,ips\n1,1,solo,0,1000,[]\n2,2,artist:name,1,50,[]\n",
        encoding="utf-8",
    )
    tags = load_selected_tags(csv_path)
    assert TagMeta(name="solo", category=0, count=1000) in tags
    assert TagMeta(name="artist:name", category=1, count=50) in tags


def test_load_selected_tags_wd14_tag_id_name_category_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_wd14.csv"
    csv_path.write_text(
        "tag_id,name,category,count\n9999999,general,9,807858\n470575,1girl,0,4225150\n212816,solo,0,3515897\n",
        encoding="utf-8",
    )

    tags = load_selected_tags(csv_path)

    assert tags == [
        TagMeta(name="general", category=9, count=807858),
        TagMeta(name="1girl", category=0, count=4225150),
        TagMeta(name="solo", category=0, count=3515897),
    ]


def test_load_selected_tags_wd14_numeric_name_uses_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_numeric_name.csv"
    csv_path.write_text("tag_id,name,category,count\n123456,2024,0,42\n", encoding="utf-8")

    tags = load_selected_tags(csv_path)

    assert tags == [TagMeta(name="2024", category=0, count=42)]


def test_load_selected_tags_legacy_id_tag_id_name_category_count_ips(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_legacy.csv"
    csv_path.write_text(
        'id,tag_id,name,category,count,ips\n1,470575,1girl,0,4225150,"[""pokemon""]"\n',
        encoding="utf-8",
    )

    tags = load_selected_tags(csv_path)

    assert tags == [TagMeta(name="1girl", category=0, count=4225150, ips=("pokemon",))]


def test_sort_by_popularity_orders_by_count_then_name() -> None:
    tags = [
        TagMeta(name="a", category=0, count=100),
        TagMeta(name="ab", category=0, count=2),
        TagMeta(name="abc", category=0, count=50),
        TagMeta(name="aa", category=0, count=100),
    ]
    ordered = sort_by_popularity(tags)
    assert [tag.name for tag in ordered] == ["a", "aa", "abc", "ab"]


def test_load_selected_tags_parses_name_category_count_and_ips(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_extra.csv"
    csv_path.write_text(
        "tag,category,count,ips\n"
        'rating:safe,rating,25,["safe"]\n'
        'bad-count,meta,not-a-number,"ignored"\n'
        "  ,general,1,[]\n",
        encoding="utf-8",
    )

    tags = load_selected_tags(csv_path)

    assert TagMeta(name="rating:safe", category=2, count=25, ips=("safe",)) in tags
    assert TagMeta(name="bad-count", category=5, count=0, ips=()) in tags
    assert any(tag.name.startswith(BROKEN_TAG_PREFIX) and tag.category == 5 for tag in tags)


def test_discover_labels_csv_prefers_explicit_then_model_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    default_csv = tmp_path / "selected_tags.csv"
    default_csv.write_text("1girl\n", encoding="utf-8")
    explicit_csv = tmp_path / "explicit.csv"
    explicit_csv.write_text("solo\n", encoding="utf-8")

    assert discover_labels_csv(model_path, explicit_csv) == explicit_csv
    assert discover_labels_csv(model_path, tmp_path / "missing.csv") is None
    assert discover_labels_csv(model_path, None) == default_csv
    assert discover_labels_csv(None, None) is None
