"""YAML prompt loader for LLM agents.

This helper reads prompt YAML files from ``llm/prompts`` and exposes a
minimal structure (system + first user message) suitable for LangGraph
nodes. It keeps the templates lightweight so placeholders like
``{sys.query}`` can be filled by simple string replacement without
breaking JSON examples inside the prompt body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass
class PromptTemplate:
    """Minimal prompt container.

    Attributes:
        name: Logical prompt name
        version: Version string from the YAML
        system: System prompt text
        user: First user message content (used as template)
        raw: Full YAML payload for debugging/extensibility
    """

    name: str
    version: str
    system: str
    user: str
    raw: dict[str, Any]


def _find_prompt_file(name: str, version: str) -> Path:
    """Resolve prompt file path (e.g., router_v1.yaml)."""
    candidate = PROMPT_DIR / f"{name}_{version}.yaml"
    if not candidate.exists():
        available = sorted(p.stem for p in PROMPT_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Prompt '{name}' version '{version}' not found. Available: {available}"
        )
    return candidate


def load_prompt_template(name: str, version: str = "v1") -> PromptTemplate:
    """Load a single prompt template from YAML.

    Args:
        name: Prompt base name (e.g., "router")
        version: Version suffix (default: v1)

    Returns:
        PromptTemplate with system + first user message populated.
    """

    path = _find_prompt_file(name, version)
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    system_prompt = str(data.get("system", "")).strip()
    messages: Iterable[dict[str, str]] = data.get("messages", []) or []
    user_message = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_message = str(msg.get("content", ""))
            break

    if not user_message:
        raise ValueError(f"Prompt '{name}_{version}' is missing a user message")

    return PromptTemplate(
        name=data.get("name", name),
        version=data.get("version", version),
        system=system_prompt,
        user=user_message,
        raw=data,
    )


def list_prompt_templates() -> list[str]:
    """List available prompt template stems (without extension)."""
    if not PROMPT_DIR.exists():
        return []
    return sorted(p.stem for p in PROMPT_DIR.glob("*.yaml"))


__all__ = ["PromptTemplate", "load_prompt_template", "list_prompt_templates", "PROMPT_DIR"]
