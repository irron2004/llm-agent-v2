from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a._io import JsonValue, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic device-family map via weighted Jaccard"
    )
    _ = parser.add_argument(
        "--corpus-meta",
        type=str,
        required=True,
        help="Path to corpus doc_meta.jsonl (Task 4 output)",
    )
    _ = parser.add_argument(
        "--shared-topics",
        type=str,
        required=True,
        help="Path to shared_topics.json (Task 5 output)",
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output family_map.json",
    )
    _ = parser.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="Edge threshold for weighted Jaccard (default: 0.2)",
    )
    return parser.parse_args()


def _collect_device_topics(
    doc_rows: list[dict[str, object]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    device_to_topics: dict[str, set[str]] = defaultdict(set)
    topic_to_devices: dict[str, set[str]] = defaultdict(set)

    for row in doc_rows:
        topic = str(row.get("topic") or "")
        device = str(row.get("es_device_name") or "")
        if not topic or not device:
            continue
        device_to_topics[device].add(topic)
        topic_to_devices[topic].add(device)

    return dict(device_to_topics), dict(topic_to_devices)


def _resolve_topic_degrees(
    topics: set[str],
    topic_to_devices: dict[str, set[str]],
    shared_topics: dict[str, object],
) -> dict[str, int]:
    deg_map: dict[str, int] = {}

    for topic in topics:
        shared_entry = shared_topics.get(topic)
        from_shared = 0
        if isinstance(shared_entry, dict):
            shared_entry_dict = cast(dict[str, object], shared_entry)
            deg_raw: object = shared_entry_dict.get("deg")
            if isinstance(deg_raw, int):
                from_shared = deg_raw
        from_coverage = len(topic_to_devices.get(topic, set()))
        deg_map[topic] = max(1, from_shared if from_shared > 0 else from_coverage)

    return deg_map


def _weighted_jaccard(
    left: set[str], right: set[str], topic_weights: dict[str, float]
) -> float:
    if not left and not right:
        return 0.0

    inter = left & right
    union = left | right
    inter_weight = sum(topic_weights[t] for t in inter)
    union_weight = sum(topic_weights[t] for t in union)
    if union_weight <= 0.0:
        return 0.0
    return inter_weight / union_weight


def build_family_map(
    doc_rows: list[dict[str, object]],
    shared_topics: dict[str, object],
    *,
    tau: float = 0.2,
) -> dict[str, JsonValue]:
    device_to_topics, topic_to_devices = _collect_device_topics(doc_rows)
    all_topics: set[str] = set(topic_to_devices.keys())
    deg_map = _resolve_topic_degrees(all_topics, topic_to_devices, shared_topics)
    topic_weights = {
        topic: 1.0 / math.log(1.0 + float(deg))
        for topic, deg in deg_map.items()
        if deg >= 1
    }

    devices = sorted(device_to_topics.keys())
    adjacency: dict[str, set[str]] = {device: set() for device in devices}

    for i, left in enumerate(devices):
        for right in devices[i + 1 :]:
            wj = _weighted_jaccard(
                device_to_topics[left], device_to_topics[right], topic_weights
            )
            if wj >= tau:
                adjacency[left].add(right)
                adjacency[right].add(left)

    components: list[list[str]] = []
    visited: set[str] = set()
    for device in devices:
        if device in visited:
            continue
        stack = [device]
        visited.add(device)
        comp: list[str] = []

        while stack:
            current = stack.pop()
            comp.append(current)
            for nxt in sorted(adjacency[current]):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)

        comp.sort()
        components.append(comp)

    components.sort(key=lambda comp: comp[0] if comp else "")
    width = max(2, len(str(max(0, len(components) - 1))))

    families: dict[str, JsonValue] = {}
    device_to_family: dict[str, JsonValue] = {}
    for idx, members in enumerate(components):
        family_id = f"F{idx:0{width}d}"
        member_json: list[JsonValue] = [m for m in members]
        families[family_id] = member_json
        for device in members:
            device_to_family[device] = family_id

    params: dict[str, JsonValue] = {
        "tau": tau,
        "weighting": "w(topic)=1/log(1+deg(topic)); deg from shared_topics[topic].deg else doc_meta device coverage; deg>=1",
        "clustering": "connected_components",
    }

    return {
        "device_to_family": device_to_family,
        "families": families,
        "params": params,
    }


def run() -> int:
    args = parse_args()
    corpus_meta_path = Path(cast(str, args.corpus_meta))
    shared_topics_path = Path(cast(str, args.shared_topics))
    out_path = Path(cast(str, args.out))
    args_dict = cast(dict[str, object], vars(args))
    tau_raw = args_dict.get("tau")
    if not isinstance(tau_raw, (int, float)):
        print("invalid tau value", file=sys.stderr)
        return 1
    tau = float(tau_raw)

    if not corpus_meta_path.exists() or not corpus_meta_path.is_file():
        print(f"corpus-meta file not found: {corpus_meta_path}", file=sys.stderr)
        return 1
    if not shared_topics_path.exists() or not shared_topics_path.is_file():
        print(f"shared-topics file not found: {shared_topics_path}", file=sys.stderr)
        return 1

    try:
        doc_rows: list[dict[str, object]] = []
        for raw in read_jsonl(corpus_meta_path):
            if not isinstance(raw, dict):
                raise RuntimeError("doc_meta.jsonl row is not an object")
            doc_rows.append(cast(dict[str, object], raw))

        shared_raw = cast(
            object, json.loads(shared_topics_path.read_text(encoding="utf-8"))
        )
        if not isinstance(shared_raw, dict):
            raise RuntimeError("shared_topics.json must be an object")

        family_map = build_family_map(
            doc_rows, cast(dict[str, object], shared_raw), tau=tau
        )
        write_json(out_path, family_map)
        return 0
    except Exception as exc:
        print(f"build_family_map failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
