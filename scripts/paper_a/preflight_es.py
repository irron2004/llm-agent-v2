from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticsearch import NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager

ERROR_EVIDENCE_PATH = ROOT / ".sisyphus/evidence/task-01-preflight-es-error.txt"

JsonValue = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight check for Elasticsearch connectivity and index alias resolution"
    )
    _ = parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Elasticsearch host override (default: SEARCH_ES_HOST)",
    )
    _ = parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Alias or index name (default: rag_chunks_{SEARCH_ES_ENV}_current)",
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path for preflight report",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_error_evidence(message: str) -> None:
    _ = ERROR_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ = ERROR_EVIDENCE_PATH.write_text(message.strip() + "\n", encoding="utf-8")


def resolve_alias_or_index(manager: EsIndexManager, requested: str) -> tuple[str, str]:
    default_alias = manager.get_alias_name()

    if requested == default_alias:
        target = manager.get_alias_target()
        if not target:
            raise RuntimeError(
                f"Alias '{requested}' does not exist or is not pointing to an index"
            )
        return requested, target

    try:
        alias_response = manager.es.indices.get_alias(name=requested)
        alias_result = cast(Mapping[str, object], cast(object, alias_response.body))
        alias_targets = list(alias_result.keys())
        if not alias_targets:
            raise RuntimeError(f"Alias '{requested}' exists but has no target index")
        return requested, alias_targets[0]
    except NotFoundError:
        if manager.es.indices.exists(index=requested):
            return requested, requested
        raise RuntimeError(f"Alias/index '{requested}' was not found") from None


def run() -> int:
    args = parse_args()
    host_arg = cast(str | None, args.host)
    index_arg = cast(str | None, args.index)
    out_arg = cast(str, args.out)

    host = host_arg or search_settings.es_host
    env = search_settings.es_env
    index_prefix = search_settings.es_index_prefix
    alias_or_index = index_arg or f"{index_prefix}_{env}_current"
    out_path = Path(out_arg)

    manager = EsIndexManager(
        es_host=host,
        env=env,
        index_prefix=index_prefix,
        es_user=search_settings.es_user or None,
        es_password=search_settings.es_password or None,
        verify_certs=True,
    )

    try:
        if not manager.es.ping():
            raise RuntimeError(f"Cannot connect to Elasticsearch at {host}")

        alias_name, resolved_index = resolve_alias_or_index(manager, alias_or_index)
        count_response = manager.es.count(index=resolved_index)
        info_response = manager.es.info()
        count_result = cast(dict[str, object], cast(object, count_response.body))
        info = cast(dict[str, object], cast(object, info_response.body))
        info_version = cast(dict[str, object], info.get("version") or {})

        diagnostics: dict[str, JsonValue] = {
            "cluster_name": cast(str | None, info.get("cluster_name")),
            "version": cast(str | None, info_version.get("number")),
            "doc_count": cast(int, count_result.get("count", 0)),
            "es_index_prefix": index_prefix,
        }
        report: dict[str, JsonValue] = {
            "host": host,
            "env": env,
            "alias": alias_name,
            "resolved_index": resolved_index,
            "timestamp": datetime.now(UTC).isoformat(),
            "diagnostics": diagnostics,
        }
        write_json(out_path, report)
        return 0
    except Exception as exc:
        error_message = (
            "Preflight failed: "
            f"{exc}. host={host} env={env} alias_or_index={alias_or_index}"
        )
        write_error_evidence(error_message)
        print(error_message, file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
