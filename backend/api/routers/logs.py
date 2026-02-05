"""System logs streaming API (RAG trace + Docker logs)."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Literal

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/logs", tags=["System Logs"])
logger = logging.getLogger(__name__)


class LogSource(str, Enum):
    RAG_TRACE = "rag-trace"
    DOCKER = "docker"
    ALL = "all"


def _get_rag_trace_log_path() -> Path:
    """Get the rag_trace.log path."""
    log_dir = Path(os.getenv("RAG_TRACE_LOG_DIR", "logs"))
    return log_dir / "rag_trace.log"


async def _stream_rag_trace(n: int = 100) -> AsyncGenerator[str, None]:
    """Stream rag_trace.log using tail -F."""
    log_path = _get_rag_trace_log_path()

    if not log_path.exists():
        yield f"data: [rag-trace] Log file not found: {log_path}\n\n"
        return

    proc = await asyncio.create_subprocess_exec(
        "tail", "-n", str(n), "-F", str(log_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        while True:
            if proc.stdout is None:
                break
            line = await proc.stdout.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            text = line.decode("utf-8", errors="replace").rstrip()
            yield f"data: [rag-trace] {text}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        proc.terminate()
        await proc.wait()


async def _stream_docker_logs(n: int = 100) -> AsyncGenerator[str, None]:
    """Stream docker logs for rag-api container."""
    # Check if docker socket is available
    docker_socket = Path("/var/run/docker.sock")
    if not docker_socket.exists():
        yield "data: [docker] Docker socket not available\n\n"
        return

    container_name = os.getenv("DOCKER_CONTAINER_NAME", "rag-api")

    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", str(n), container_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # Combine stderr into stdout
    )

    try:
        while True:
            if proc.stdout is None:
                break
            line = await proc.stdout.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            text = line.decode("utf-8", errors="replace").rstrip()
            yield f"data: [docker] {text}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        proc.terminate()
        await proc.wait()


async def _stream_all_logs(n: int = 100) -> AsyncGenerator[str, None]:
    """Stream both rag-trace and docker logs merged."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _rag_worker():
        try:
            async for line in _stream_rag_trace(n):
                await queue.put(line)
        except asyncio.CancelledError:
            pass

    async def _docker_worker():
        try:
            async for line in _stream_docker_logs(n):
                await queue.put(line)
        except asyncio.CancelledError:
            pass

    rag_task = asyncio.create_task(_rag_worker())
    docker_task = asyncio.create_task(_docker_worker())

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                if item is None:
                    break
                yield item
            except asyncio.TimeoutError:
                # Check if both tasks are done
                if rag_task.done() and docker_task.done():
                    break
                continue
    except asyncio.CancelledError:
        pass
    finally:
        rag_task.cancel()
        docker_task.cancel()
        try:
            await asyncio.gather(rag_task, docker_task, return_exceptions=True)
        except Exception:
            pass


@router.get("/stream")
async def stream_logs(
    source: Literal["rag-trace", "docker", "all"] = Query("all", description="Log source to stream"),
    n: int = Query(100, ge=1, le=1000, description="Number of initial lines to fetch"),
):
    """Stream system logs via SSE.

    - source=rag-trace: RAG trace logs only (tail -F logs/rag_trace.log)
    - source=docker: Docker container logs only
    - source=all: Both sources merged
    """

    async def _generate():
        yield f"data: [system] Starting log stream (source={source}, n={n})\n\n"

        if source == "rag-trace":
            async for line in _stream_rag_trace(n):
                yield line
        elif source == "docker":
            async for line in _stream_docker_logs(n):
                yield line
        else:  # all
            async for line in _stream_all_logs(n):
                yield line

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
