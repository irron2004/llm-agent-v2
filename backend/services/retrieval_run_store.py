from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol


class RetrievalRunStore(Protocol):
    def put(self, run_id: str, snapshot: Mapping[str, Any]) -> None: ...

    def get(self, run_id: str) -> dict[str, Any] | None: ...


class RetrievalRunSnapshotStore:
    def __init__(
        self,
        *,
        ttl_seconds: float = 900.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock or time.monotonic
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}

    def put(self, run_id: str, snapshot: Mapping[str, Any]) -> None:
        now = self._clock()
        self.cleanup(now=now)
        expires_at = now + self._ttl_seconds
        self._entries[str(run_id)] = (expires_at, dict(snapshot))

    def get(self, run_id: str) -> dict[str, Any] | None:
        now = self._clock()
        self.cleanup(now=now)

        item = self._entries.get(str(run_id))
        if item is None:
            return None

        expires_at, snapshot = item
        if expires_at <= now:
            self._entries.pop(str(run_id), None)
            return None

        return dict(snapshot)

    def cleanup(self, *, now: float | None = None) -> int:
        current = self._clock() if now is None else now
        expired = [
            run_id for run_id, (expires_at, _) in self._entries.items() if expires_at <= current
        ]
        for run_id in expired:
            self._entries.pop(run_id, None)
        return len(expired)


_default_retrieval_run_store: RetrievalRunStore = RetrievalRunSnapshotStore(ttl_seconds=900.0)


def get_default_retrieval_run_store() -> RetrievalRunStore:
    return _default_retrieval_run_store


__all__ = [
    "RetrievalRunStore",
    "RetrievalRunSnapshotStore",
    "get_default_retrieval_run_store",
]
