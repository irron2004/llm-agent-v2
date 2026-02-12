from backend.domain.doc_type_mapping import group_doc_type_items


def test_group_doc_type_items_groups_raw_doc_type_variants() -> None:
    items = [
        {"name": "Trouble Shooting Guide", "doc_count": 4},
        {"name": "Installation Manual", "doc_count": 3},
        {"name": "SOP/Manual", "doc_count": 7},
        {"name": "myservice", "doc_count": 2},
        {"name": "maintenance", "doc_count": 5},
    ]

    grouped = {item["name"]: item["doc_count"] for item in group_doc_type_items(items)}

    assert grouped["ts"] == 4
    assert grouped["setup"] == 3
    assert grouped["SOP"] == 7
    assert grouped["myservice"] == 2
    assert grouped["gcb"] == 5


def test_group_doc_type_items_accepts_already_grouped_entries() -> None:
    items = [
        {"name": "SOP", "doc_count": 11},
        {"name": "ts", "doc_count": 6},
        {"name": "setup", "doc_count": 4},
        {"name": "myservice", "doc_count": 1},
        {"name": "gcb", "doc_count": 3},
    ]

    grouped = {item["name"]: item["doc_count"] for item in group_doc_type_items(items)}

    assert grouped == {
        "myservice": 1,
        "SOP": 11,
        "ts": 6,
        "setup": 4,
        "gcb": 3,
    }
