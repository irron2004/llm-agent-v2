from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a._io import JsonValue, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build shared topics + scope levels from corpus doc_meta"
    )
    _ = parser.add_argument(
        "--corpus-meta",
        type=str,
        required=True,
        help="Path to corpus doc_meta.jsonl (Task 4 output)",
    )
    _ = parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for policy artifacts",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    corpus_meta_path = Path(cast(str, args.corpus_meta))
    out_dir = Path(cast(str, args.out_dir))

    if not corpus_meta_path.exists() or not corpus_meta_path.is_file():
        print(f"corpus-meta file not found: {corpus_meta_path}", file=sys.stderr)
        return 1

    shared_topic_device_doc_types = {"sop_pdf", "sop_pptx"}
    shared_doc_doc_types = {"sop_pdf", "sop_pptx", "ts"}

    docs: list[dict[str, str]] = []
    topic_to_devices_allowed: dict[str, set[str]] = defaultdict(set)

    for raw in read_jsonl(corpus_meta_path):
        if not isinstance(raw, dict):
            raise RuntimeError("doc_meta.jsonl row is not an object")
        source_file = str(raw.get("source_file") or "")
        topic = str(raw.get("topic") or "")
        manifest_doc_type = str(raw.get("manifest_doc_type") or "")
        es_doc_id = str(raw.get("es_doc_id") or "")
        es_doc_type = str(raw.get("es_doc_type") or "")
        es_device_name = str(raw.get("es_device_name") or "")
        es_equip_id = str(raw.get("es_equip_id") or "")

        if not source_file or not manifest_doc_type or not es_doc_id:
            raise RuntimeError(
                f"doc_meta row missing required keys: source_file={source_file!r} manifest_doc_type={manifest_doc_type!r} es_doc_id={es_doc_id!r}"
            )

        if not topic and manifest_doc_type != "setup_manual":
            raise RuntimeError(
                f"doc_meta row missing topic for non-setup doc: source_file={source_file!r} manifest_doc_type={manifest_doc_type!r} es_doc_id={es_doc_id!r}"
            )

        docs.append(
            {
                "source_file": source_file,
                "topic": topic,
                "manifest_doc_type": manifest_doc_type,
                "es_doc_id": es_doc_id,
                "es_doc_type": es_doc_type,
                "es_device_name": es_device_name,
                "es_equip_id": es_equip_id,
            }
        )

        if (
            topic
            and es_device_name
            and manifest_doc_type in shared_topic_device_doc_types
        ):
            topic_to_devices_allowed[topic].add(es_device_name)

    shared_topics: dict[str, JsonValue] = {}
    for topic in sorted(topic_to_devices_allowed.keys()):
        devices = sorted(topic_to_devices_allowed[topic])
        devices_json: list[JsonValue] = [d for d in devices]
        deg = len(devices)

        is_shared = deg >= 3

        topic_entry: dict[str, JsonValue] = {
            "deg": deg,
            "devices": devices_json,
            "is_shared": is_shared,
        }
        shared_topics[topic] = topic_entry

    shared_doc_ids: list[str] = []
    doc_scope_rows: list[dict[str, JsonValue]] = []
    for doc in docs:
        topic = doc["topic"]
        shared_obj = shared_topics.get(topic)
        shared_dict = shared_obj if isinstance(shared_obj, dict) else {}
        is_shared_topic = bool(shared_dict.get("is_shared"))
        is_shared = doc["manifest_doc_type"] in shared_doc_doc_types and is_shared_topic

        es_doc_type = doc["es_doc_type"]
        if is_shared:
            scope_level = "shared"
        elif es_doc_type in {"myservice", "gcb"}:
            scope_level = "equip"
        else:
            scope_level = "device"

        row: dict[str, JsonValue] = {
            "es_doc_id": doc["es_doc_id"],
            "es_device_name": doc["es_device_name"],
            "es_doc_type": doc["es_doc_type"],
            "topic": topic,
            "is_shared": is_shared,
            "scope_level": scope_level,
        }
        doc_scope_rows.append(row)
        if is_shared:
            shared_doc_ids.append(doc["es_doc_id"])

    shared_topic_count = 0
    for t in shared_topics.values():
        if isinstance(t, dict) and t.get("is_shared") is True:
            shared_topic_count += 1
    shared_doc_count = len(shared_doc_ids)
    snapshot: dict[str, JsonValue] = {
        "shared_topic_count": shared_topic_count,
        "shared_doc_count": shared_doc_count,
    }

    if shared_topic_count != 13:
        raise RuntimeError(
            f"shared_topic_count mismatch: expected 13, got {shared_topic_count}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "shared_topics.json", shared_topics)
    write_jsonl(out_dir / "doc_scope.jsonl", doc_scope_rows)
    _ = (out_dir / "shared_doc_ids.txt").write_text(
        "\n".join(shared_doc_ids) + "\n", encoding="utf-8"
    )
    write_json(out_dir / "policy_snapshot.json", snapshot)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
