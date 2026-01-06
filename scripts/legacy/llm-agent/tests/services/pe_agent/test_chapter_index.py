import pytest

from services.pe_agent.pe_core.chapter_index import build_chapter_index


@pytest.fixture()
def sample_doc():
    return {
        0: {
            "content": "Cover page",
            "metadata": {"page": 0},
        },
        1: {
            "content": (
                "목차\n"
                " 0. Safety ........................................ 4\n"
                " 1. Install Preperation(※환경안전 보호구 : 안전모, 안전화) .... 16\n"
                " 2. Fab In (Moving) (※환경안전 보호구 : 안전모, 안전화) .... 19\n"
            ),
            "metadata": {"page": 1},
        },
        3: {"content": "0. Safety procedures ...", "metadata": {"page": 3}},
        4: {"content": "4. Safety notes", "metadata": {"page": 4}},
        15: {"content": "1. Install Preperation steps", "metadata": {"page": 15}},
        18: {"content": "2. Fab In processes", "metadata": {"page": 18}},
    }


def test_build_chapter_index_basic(sample_doc):
    index = build_chapter_index(sample_doc, toc_page=1)

    chapters = index["chapters"]
    titles = [ch.title for ch in chapters]

    assert titles == [
        "Safety",
        "Install Preperation",
        "Fab In",
    ]

    first_page = index["first_page"]
    assert first_page["Safety"] == 3
    assert first_page["Install Preperation"] == 15
    assert first_page["Fab In"] == 18

    assert index["missing"] == []
