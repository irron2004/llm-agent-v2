"""Prompt templates for query expansion."""

# General multi-query expansion prompt (English)
GENERAL_MQ_V1 = """You are an AI assistant that helps generate alternative search queries.

Given an original search query, generate {n} different search queries that would help find relevant information.
Each query should:
- Approach the topic from a different angle
- Use different keywords or phrasings
- Cover different aspects of what the user might be looking for

Original query: {query}

Generate exactly {n} alternative queries, one per line. Do not number them or add any other text.
Output only the queries:"""

# Korean multi-query expansion prompt
GENERAL_MQ_V1_KO = """당신은 검색 쿼리를 다양하게 확장하는 AI 어시스턴트입니다.

주어진 원본 검색 쿼리에 대해, 관련 정보를 찾는 데 도움이 될 {n}개의 다른 검색 쿼리를 생성하세요.
각 쿼리는:
- 주제를 다른 각도에서 접근해야 합니다
- 다른 키워드나 표현을 사용해야 합니다
- 사용자가 찾고자 하는 것의 다른 측면을 다뤄야 합니다

원본 쿼리: {query}

정확히 {n}개의 대체 쿼리를 한 줄에 하나씩 생성하세요. 번호를 붙이거나 다른 텍스트를 추가하지 마세요.
쿼리만 출력하세요:"""

# Technical/domain-specific expansion prompt
TECHNICAL_MQ_V1 = """You are a technical search assistant specializing in generating precise search queries.

Given a technical query, generate {n} alternative queries that:
- Include relevant technical terms and synonyms
- Cover related concepts and technologies
- Address potential underlying problems or solutions

Original query: {query}

Generate exactly {n} technical search queries, one per line:"""

# Semiconductor domain prompt (Korean)
SEMICONDUCTOR_MQ_V1 = """당신은 반도체 장비 문제 해결을 위한 검색 쿼리 전문가입니다.

주어진 쿼리에 대해 {n}개의 관련 검색 쿼리를 생성하세요.
각 쿼리는:
- 관련 장비명, 모듈명, 알람 코드를 포함할 수 있습니다
- 유사한 문제나 해결책을 다룰 수 있습니다
- PM, 점검, 유지보수 관점을 포함할 수 있습니다

원본 쿼리: {query}

{n}개의 검색 쿼리를 한 줄에 하나씩 생성하세요:"""


# Prompt registry
PROMPT_TEMPLATES = {
    "general_mq_v1": GENERAL_MQ_V1,
    "general_mq_v1_ko": GENERAL_MQ_V1_KO,
    "technical_mq_v1": TECHNICAL_MQ_V1,
    "semiconductor_mq_v1": SEMICONDUCTOR_MQ_V1,
}


def get_prompt_template(name: str) -> str:
    """Get a prompt template by name.

    Args:
        name: Template name

    Returns:
        Prompt template string

    Raises:
        ValueError: If template not found
    """
    if name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown prompt template: '{name}'. Available: {available}"
        )
    return PROMPT_TEMPLATES[name]


def list_prompt_templates() -> list[str]:
    """List available prompt template names."""
    return list(PROMPT_TEMPLATES.keys())


__all__ = [
    "PROMPT_TEMPLATES",
    "get_prompt_template",
    "list_prompt_templates",
    "GENERAL_MQ_V1",
    "GENERAL_MQ_V1_KO",
    "TECHNICAL_MQ_V1",
    "SEMICONDUCTOR_MQ_V1",
]