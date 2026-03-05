from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import random
import sys
from pathlib import Path
from typing import Protocol, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BuildFamilyMapFn(Protocol):
    def __call__(
        self,
        doc_rows: list[dict[str, object]],
        shared_topics: dict[str, object],
        *,
        tau: float = 0.2,
    ) -> dict[str, object]: ...


_module_path = ROOT / "scripts" / "paper_a" / "build_family_map.py"
_spec = spec_from_file_location("paper_a_build_family_map", _module_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load module spec: {_module_path}")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_family_map = cast(BuildFamilyMapFn, _module.build_family_map)


def _row(device: str, topic: str) -> dict[str, object]:
    return {
        "source_file": f"{device}_{topic}.pdf",
        "topic": topic,
        "manifest_doc_type": "sop_pdf",
        "es_doc_id": f"{device}_{topic}",
        "es_doc_type": "sop",
        "es_device_name": device,
        "es_equip_id": "",
    }


def test_build_family_map_is_deterministic_and_stable_ids() -> None:
    doc_rows = [
        _row("ALPHA", "t1"),
        _row("ALPHA", "t2"),
        _row("BETA", "t1"),
        _row("GAMMA", "t3"),
        _row("DELTA", "t3"),
        _row("DELTA", "t4"),
        _row("EPSILON", "t5"),
    ]
    shared_topics: dict[str, object] = {
        "t1": {"deg": 2},
        "t3": {"deg": 2},
    }

    baseline = build_family_map(doc_rows, shared_topics, tau=0.2)

    shuffled_rows = list(doc_rows)
    rng = random.Random(7)
    rng.shuffle(shuffled_rows)
    shuffled_topics = {k: shared_topics[k] for k in reversed(list(shared_topics.keys()))}
    re_run = build_family_map(shuffled_rows, shuffled_topics, tau=0.2)

    assert baseline == re_run
    params = cast(dict[str, object], baseline["params"])
    assert params["tau"] == 0.2

    families = baseline["families"]
    assert families == {
        "F00": ["ALPHA", "BETA"],
        "F01": ["DELTA", "GAMMA"],
        "F02": ["EPSILON"],
    }

    assert baseline["device_to_family"] == {
        "ALPHA": "F00",
        "BETA": "F00",
        "DELTA": "F01",
        "EPSILON": "F02",
        "GAMMA": "F01",
    }
