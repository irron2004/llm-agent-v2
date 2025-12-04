from services.graph_ingest.reporting import summarize_patterns


def test_summarize_patterns_counts_patterns_and_verbs() -> None:
    edges = [
        {
            "props": {
                "pattern": "REF_SOP_FWD",
                "verb": "참조",
            }
        },
        {
            "props": {
                "pattern": "cross_ref_sentence",
                "verb": "see",
            }
        },
        {
            "props": {
                "pattern": "REF_SOP_FWD",
            }
        },
    ]

    result = summarize_patterns(edges)

    assert result["pattern_counts"] == {"REF_SOP_FWD": 2, "cross_ref_sentence": 1}
    assert result["verb_counts"] == {"참조": 1, "see": 1}
