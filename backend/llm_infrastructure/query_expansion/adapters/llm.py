"""LLM-based query expander."""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.config.settings import vllm_settings
from backend.llm_infrastructure.llm import get_llm
from backend.llm_infrastructure.llm.base import BaseLLM

from ..base import BaseQueryExpander, ExpandedQueries
from ..registry import register_query_expander
from ..prompts import get_prompt_template

logger = logging.getLogger(__name__)


@register_query_expander("llm", version="v1")
class LLMQueryExpander(BaseQueryExpander):
    """Query expander using LLM to generate alternative queries.

    Uses an LLM to generate semantically related queries that can help
    improve retrieval recall by searching with multiple query variations.
    """

    DEFAULT_PROMPT = "general_mq_v1"

    def __init__(
        self,
        llm: BaseLLM | None = None,
        prompt_template: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> None:
        """Initialize LLM query expander.

        Args:
            llm: LLM instance to use (default: vLLM from settings)
            prompt_template: Name of prompt template or custom prompt string
            temperature: LLM temperature for query generation
            max_tokens: Max tokens for LLM response
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM
        if llm is not None:
            self._llm = llm
        else:
            self._llm = get_llm(
                "vllm",
                version="v1",
                base_url=vllm_settings.base_url,
                model=vllm_settings.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=vllm_settings.timeout,
            )

        # Load prompt template
        if prompt_template is None:
            self.prompt_template = get_prompt_template(self.DEFAULT_PROMPT)
        elif prompt_template in ["general_mq_v1", "general_mq_v1_ko",
                                  "technical_mq_v1", "semiconductor_mq_v1"]:
            self.prompt_template = get_prompt_template(prompt_template)
        else:
            # Assume it's a custom prompt string
            self.prompt_template = prompt_template

    def expand(
        self,
        query: str,
        n: int = 3,
        include_original: bool = True,
        **kwargs: Any,
    ) -> ExpandedQueries:
        """Expand a query using LLM.

        Args:
            query: Original search query
            n: Number of expanded queries to generate
            include_original: Whether to include original query in results
            **kwargs: Additional parameters (passed to LLM)

        Returns:
            ExpandedQueries with original and generated queries
        """
        if n <= 0:
            return ExpandedQueries(
                original_query=query,
                expanded_queries=[],
                include_original=include_original,
            )

        # Format prompt
        prompt = self.prompt_template.format(query=query, n=n)

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._llm.generate(
                messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            # Parse response
            expanded = self._parse_response(response.text, n)

            # Remove duplicates and empty queries
            expanded = self._deduplicate(expanded, query)

            logger.debug(
                f"Expanded query '{query}' into {len(expanded)} queries: {expanded}"
            )

            return ExpandedQueries(
                original_query=query,
                expanded_queries=expanded,
                include_original=include_original,
            )

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Returning original query only.")
            return ExpandedQueries(
                original_query=query,
                expanded_queries=[],
                include_original=include_original,
            )

    def _parse_response(self, response_text: str, n: int) -> list[str]:
        """Parse LLM response into list of queries.

        Args:
            response_text: Raw LLM response
            n: Expected number of queries

        Returns:
            List of parsed queries
        """
        lines = response_text.strip().split("\n")
        queries = []

        for line in lines:
            # Clean up the line
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes like "1.", "1)", "-", "*", etc.
            line = re.sub(r"^[\d]+[.)\-:\s]+", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()

            # Remove quotes if present
            if (line.startswith('"') and line.endswith('"')) or \
               (line.startswith("'") and line.endswith("'")):
                line = line[1:-1]

            if line:
                queries.append(line)

        return queries[:n]  # Limit to requested number

    def _deduplicate(self, queries: list[str], original_query: str) -> list[str]:
        """Remove duplicate queries and those too similar to original.

        Args:
            queries: List of expanded queries
            original_query: Original query to compare against

        Returns:
            Deduplicated list of queries
        """
        seen = set()
        seen.add(original_query.lower().strip())

        deduplicated = []
        for q in queries:
            q_normalized = q.lower().strip()
            if q_normalized not in seen and q_normalized:
                seen.add(q_normalized)
                deduplicated.append(q)

        return deduplicated

    def __repr__(self) -> str:
        return f"LLMQueryExpander(temperature={self.temperature})"


__all__ = ["LLMQueryExpander"]
