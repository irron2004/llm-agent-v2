from pathlib import Path
import sys
from typing import Any, cast


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


from llm_infrastructure.llm.langgraph_agent import AgentState, auto_parse_node


_DUMMY = cast(Any, None)


def test_strict_filter_doc_types_preserved_in_auto_parse_node() -> None:
    state: AgentState = {
        "query": "sop 교체 절차 알려줘",
        "selected_doc_types": ["Trouble Shooting Guide"],
        "selected_doc_types_strict": True,
        "parsed_query": {
            "selected_doc_types": ["Trouble Shooting Guide"],
            "doc_types_strict": True,
        },
    }

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["SUPRA"],
        doc_type_names=["myservice", "ts", "sop", "setup", "gcb"],
    )

    assert result["selected_doc_types"] == ["Trouble Shooting Guide"]
