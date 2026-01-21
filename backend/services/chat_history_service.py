"""Elasticsearch-backed chat history service.

Provides storage and retrieval of conversation turns for chat history feature.
Each turn consists of a user question and assistant response with optional document references.

Usage:
    # Create from settings
    svc = ChatHistoryService.from_settings()

    # Save a turn
    svc.save_turn(session_id, turn)

    # Get session history
    turns = svc.get_session(session_id)

    # List recent sessions
    sessions = svc.list_sessions(limit=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.mappings import (
    get_chat_turns_mapping,
    get_index_settings,
)

logger = logging.getLogger(__name__)

# Index naming
CHAT_TURNS_INDEX_PREFIX = "chat_turns"
CHAT_TURNS_SCHEMA_VERSION = "v1"


@dataclass
class DocRef:
    """Reference to a document shown in assistant response."""
    slot: int  # User-visible number (1, 2, 3...)
    doc_id: str
    title: str
    snippet: str
    page: Optional[int] = None
    pages: Optional[list[int]] = None
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "page": self.page,
            "pages": self.pages,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocRef":
        return cls(
            slot=data.get("slot", 0),
            doc_id=data.get("doc_id", ""),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
            page=data.get("page"),
            pages=data.get("pages"),
            score=data.get("score"),
        )


@dataclass
class ChatTurn:
    """A single conversation turn (user + assistant)."""
    session_id: str
    turn_id: int
    user_text: str
    assistant_text: str
    doc_refs: list[DocRef] = field(default_factory=list)
    title: Optional[str] = None  # Session title (set on first turn)
    summary: Optional[str] = None  # Turn summary for history selection
    feedback_rating: Optional[str] = None  # "up" | "down"
    feedback_reason: Optional[str] = None
    feedback_ts: Optional[datetime] = None
    ts: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "doc_refs": [ref.to_dict() for ref in self.doc_refs],
            "title": self.title,
            "summary": self.summary,
            "feedback_rating": self.feedback_rating,
            "feedback_reason": self.feedback_reason,
            "feedback_ts": self.feedback_ts.isoformat() if self.feedback_ts else None,
            "ts": (self.ts or datetime.utcnow()).isoformat(),
            "schema_version": CHAT_TURNS_SCHEMA_VERSION,
            "created_at": self.created_at.isoformat() if self.created_at else now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatTurn":
        doc_refs = [DocRef.from_dict(ref) for ref in data.get("doc_refs", [])]
        return cls(
            session_id=data.get("session_id", ""),
            turn_id=data.get("turn_id", 0),
            user_text=data.get("user_text", ""),
            assistant_text=data.get("assistant_text", ""),
            doc_refs=doc_refs,
            title=data.get("title"),
            summary=data.get("summary"),
            feedback_rating=data.get("feedback_rating"),
            feedback_reason=data.get("feedback_reason"),
            feedback_ts=datetime.fromisoformat(data["feedback_ts"]) if data.get("feedback_ts") else None,
            ts=datetime.fromisoformat(data["ts"]) if data.get("ts") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class SessionSummary:
    """Summary of a chat session for listing."""
    session_id: str
    title: str
    preview: str  # First user message or summary
    turn_count: int
    created_at: datetime
    updated_at: datetime


class ChatHistoryService:
    """Elasticsearch-backed chat history storage and retrieval."""

    def __init__(
        self,
        *,
        es_client: Elasticsearch,
        index: str,
    ) -> None:
        """Initialize chat history service.

        Args:
            es_client: Elasticsearch client instance.
            index: Index name or alias to use.
        """
        self.es = es_client
        self.index = index

    @classmethod
    def from_settings(
        cls,
        *,
        es_client: Elasticsearch | None = None,
        env: str | None = None,
    ) -> "ChatHistoryService":
        """Create ChatHistoryService from global settings.

        Args:
            es_client: Pre-configured ES client. Created from settings if None.
            env: Environment name. Defaults to search_settings.es_env.

        Returns:
            Configured ChatHistoryService instance.
        """
        if es_client is None:
            client_kwargs: dict[str, Any] = {
                "hosts": [search_settings.es_host],
                "verify_certs": True,
            }
            if search_settings.es_user and search_settings.es_password:
                client_kwargs["basic_auth"] = (
                    search_settings.es_user,
                    search_settings.es_password,
                )
            es_client = Elasticsearch(**client_kwargs)

        env = env or search_settings.es_env
        index = f"{CHAT_TURNS_INDEX_PREFIX}_{env}_current"

        return cls(es_client=es_client, index=index)

    def ensure_index(self) -> bool:
        """Ensure the chat turns index exists, create if not.

        Returns:
            True if index exists or was created successfully.
        """
        # Check if alias exists
        if self.es.indices.exists_alias(name=self.index):
            return True

        # Create versioned index
        env = search_settings.es_env
        versioned_index = f"{CHAT_TURNS_INDEX_PREFIX}_{env}_v1"

        if not self.es.indices.exists(index=versioned_index):
            self.es.indices.create(
                index=versioned_index,
                body={
                    "settings": get_index_settings(),
                    "mappings": get_chat_turns_mapping(),
                },
            )
            logger.info(f"Created chat turns index: {versioned_index}")

        # Create alias pointing to the versioned index
        if not self.es.indices.exists_alias(name=self.index):
            self.es.indices.put_alias(index=versioned_index, name=self.index)
            logger.info(f"Created alias {self.index} -> {versioned_index}")

        return True

    def save_turn(self, turn: ChatTurn) -> str:
        """Save a conversation turn to ES.

        Args:
            turn: ChatTurn to save.

        Returns:
            Document ID of the saved turn.
        """
        doc_id = f"{turn.session_id}:{turn.turn_id}"
        self.es.index(
            index=self.index,
            id=doc_id,
            body=turn.to_dict(),
            routing=turn.session_id,  # Route by session for colocation
            refresh=True,  # Immediate refresh for correct next turn_id calculation
        )
        logger.debug(f"Saved turn {doc_id}")
        return doc_id

    def get_session(self, session_id: str) -> list[ChatTurn]:
        """Get all turns for a session.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            List of ChatTurn objects sorted by turn_id.
        """
        query = {
            "query": {"term": {"session_id": session_id}},
            "sort": [{"turn_id": "asc"}],
            "size": 1000,  # Reasonable max turns per session
        }
        try:
            result = self.es.search(
                index=self.index,
                body=query,
                routing=session_id,
            )
            hits = result.get("hits", {}).get("hits", [])
            return [ChatTurn.from_dict(hit["_source"]) for hit in hits]
        except NotFoundError:
            return []

    def get_turn(self, session_id: str, turn_id: int) -> ChatTurn | None:
        """Get a specific turn.

        Args:
            session_id: Session ID.
            turn_id: Turn number within session.

        Returns:
            ChatTurn if found, None otherwise.
        """
        doc_id = f"{session_id}:{turn_id}"
        try:
            result = self.es.get(
                index=self.index,
                id=doc_id,
                routing=session_id,
            )
            return ChatTurn.from_dict(result["_source"])
        except NotFoundError:
            return None

    def update_turn_feedback(
        self,
        session_id: str,
        turn_id: int,
        *,
        rating: str,
        reason: Optional[str] = None,
    ) -> ChatTurn | None:
        """Update feedback for a specific turn."""
        doc_id = f"{session_id}:{turn_id}"
        now = datetime.utcnow()
        payload = {
            "feedback_rating": rating,
            "feedback_reason": reason,
            "feedback_ts": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        try:
            self.es.update(
                index=self.index,
                id=doc_id,
                routing=session_id,
                body={"doc": payload},
                refresh=True,
            )
        except NotFoundError:
            return None
        return self.get_turn(session_id, turn_id)

    def list_sessions(self, limit: int = 50, offset: int = 0, include_hidden: bool = False) -> list[SessionSummary]:
        """List recent chat sessions.

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            include_hidden: If True, include hidden (soft-deleted) sessions.

        Returns:
            List of SessionSummary objects sorted by most recent.
        """
        # Build filter clause
        filter_clause = []
        if not include_hidden:
            # Exclude hidden sessions: is_hidden must be false or missing
            filter_clause.append({
                "bool": {
                    "should": [
                        {"term": {"is_hidden": False}},
                        {"bool": {"must_not": {"exists": {"field": "is_hidden"}}}},
                    ],
                    "minimum_should_match": 1,
                }
            })

        # Aggregate by session_id, get latest turn per session
        query: dict[str, Any] = {
            "size": 0,
            "aggs": {
                "sessions": {
                    "terms": {
                        "field": "session_id",
                        "size": limit + offset,
                        "order": {"latest_ts": "desc"},
                    },
                    "aggs": {
                        "latest_ts": {"max": {"field": "ts"}},
                        "earliest_ts": {"min": {"field": "ts"}},
                        "turn_count": {"value_count": {"field": "turn_id"}},
                        "first_turn": {
                            "top_hits": {
                                "size": 1,
                                "sort": [{"turn_id": "asc"}],
                                "_source": ["title", "user_text", "created_at"],
                            }
                        },
                    },
                }
            },
        }

        # Add filter if needed
        if filter_clause:
            query["query"] = {"bool": {"filter": filter_clause}}
        try:
            result = self.es.search(index=self.index, body=query)
            buckets = result.get("aggregations", {}).get("sessions", {}).get("buckets", [])

            sessions = []
            for bucket in buckets[offset:offset + limit]:
                session_id = bucket["key"]
                turn_count = int(bucket["turn_count"]["value"])
                latest_ts = bucket["latest_ts"]["value_as_string"]
                earliest_ts = bucket["earliest_ts"]["value_as_string"]

                # Get first turn data
                first_hit = bucket["first_turn"]["hits"]["hits"][0]["_source"] if bucket["first_turn"]["hits"]["hits"] else {}
                title = first_hit.get("title") or first_hit.get("user_text", "")[:50]
                preview = first_hit.get("user_text", "")[:100]

                sessions.append(SessionSummary(
                    session_id=session_id,
                    title=title,
                    preview=preview,
                    turn_count=turn_count,
                    created_at=datetime.fromisoformat(earliest_ts.replace("Z", "+00:00")) if earliest_ts else datetime.utcnow(),
                    updated_at=datetime.fromisoformat(latest_ts.replace("Z", "+00:00")) if latest_ts else datetime.utcnow(),
                ))

            return sessions
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> int:
        """Delete all turns for a session (hard delete).

        Args:
            session_id: Session ID to delete.

        Returns:
            Number of documents deleted.
        """
        query = {
            "query": {"term": {"session_id": session_id}},
        }
        result = self.es.delete_by_query(
            index=self.index,
            body=query,
            routing=session_id,
        )
        deleted = result.get("deleted", 0)
        logger.info(f"Deleted {deleted} turns for session {session_id}")
        return deleted

    def hide_session(self, session_id: str) -> int:
        """Hide a session (soft delete) - marks all turns as hidden.

        Args:
            session_id: Session ID to hide.

        Returns:
            Number of documents updated.
        """
        query = {
            "script": {
                "source": "ctx._source.is_hidden = true",
                "lang": "painless",
            },
            "query": {"term": {"session_id": session_id}},
        }
        result = self.es.update_by_query(
            index=self.index,
            body=query,
            routing=session_id,
        )
        updated = result.get("updated", 0)
        logger.info(f"Hid {updated} turns for session {session_id}")
        return updated

    def unhide_session(self, session_id: str) -> int:
        """Unhide a session - marks all turns as visible.

        Args:
            session_id: Session ID to unhide.

        Returns:
            Number of documents updated.
        """
        query = {
            "script": {
                "source": "ctx._source.is_hidden = false",
                "lang": "painless",
            },
            "query": {"term": {"session_id": session_id}},
        }
        result = self.es.update_by_query(
            index=self.index,
            body=query,
            routing=session_id,
        )
        updated = result.get("updated", 0)
        logger.info(f"Unhid {updated} turns for session {session_id}")
        return updated

    def get_next_turn_id(self, session_id: str) -> int:
        """Get the next turn ID for a session.

        Args:
            session_id: Session ID.

        Returns:
            Next turn ID (1 for new session, max + 1 for existing).
        """
        query = {
            "query": {"term": {"session_id": session_id}},
            "aggs": {"max_turn": {"max": {"field": "turn_id"}}},
            "size": 0,
        }
        try:
            result = self.es.search(
                index=self.index,
                body=query,
                routing=session_id,
            )
            max_turn = result.get("aggregations", {}).get("max_turn", {}).get("value")
            return int(max_turn) + 1 if max_turn is not None else 1
        except Exception:
            return 1
