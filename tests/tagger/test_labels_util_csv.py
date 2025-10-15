from pathlib import Path

from tagger.labels_util import TagMeta, load_selected_tags, sort_by_popularity


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
    assert TagMeta(name="character:kobato", category=1, count=0) in tags


def test_load_selected_tags_four_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags_four.csv"
    csv_path.write_text(
        "id,tag_id,name,category,count,ips\n" "1,1,solo,0,1000,[]\n" "2,2,artist:name,4,50,[]\n",
        encoding="utf-8",
    )
    tags = load_selected_tags(csv_path)
    assert TagMeta(name="solo", category=0, count=1000) in tags
    assert TagMeta(name="artist:name", category=4, count=50) in tags


def test_sort_by_popularity_orders_by_count_then_name() -> None:
    tags = [
        TagMeta(name="a", category=0, count=100),
        TagMeta(name="ab", category=0, count=2),
        TagMeta(name="abc", category=0, count=50),
        TagMeta(name="aa", category=0, count=100),
    ]
    ordered = sort_by_popularity(tags)
    assert [tag.name for tag in ordered] == ["a", "aa", "abc", "ab"]
