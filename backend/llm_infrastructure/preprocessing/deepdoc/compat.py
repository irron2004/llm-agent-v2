#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""RAGFlow 호환 모듈.

RAGFlow의 common, rag 패키지에 대한 의존성을 제공합니다.
실제 구현이 필요한 경우 이 파일을 수정하거나 확장하세요.
"""

import os
import logging
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# common.file_utils
# ============================================================================

def get_project_base_directory() -> str:
    """프로젝트 루트 디렉토리를 반환합니다.

    Returns:
        프로젝트 루트 디렉토리 경로
    """
    # backend 디렉토리의 상위 디렉토리를 프로젝트 루트로 간주
    current = Path(__file__).resolve()
    # deepdoc/compat.py -> preprocessing -> llm_infrastructure -> backend -> project_root
    return str(current.parents[4])


def traversal_files(directory: str) -> list[str]:
    """디렉토리 내 모든 파일을 재귀적으로 탐색합니다.

    Args:
        directory: 탐색할 디렉토리 경로

    Yields:
        파일 경로
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


# ============================================================================
# common.misc_utils
# ============================================================================

def pip_install_torch() -> None:
    """PyTorch 설치를 확인하고 필요시 설치합니다.

    Note:
        이 함수는 현재 no-op입니다. PyTorch는 requirements.txt에서 관리됩니다.
    """
    try:
        import torch
        logger.debug(f"PyTorch {torch.__version__} 사용 가능")
    except ImportError:
        logger.warning("PyTorch가 설치되어 있지 않습니다. pip install torch로 설치하세요.")


# ============================================================================
# common.constants
# ============================================================================

class LLMType(str, Enum):
    """LLM 타입 상수."""
    CHAT = "chat"
    EMBEDDING = "embedding"
    SPEECH2TEXT = "speech2text"
    IMAGE2TEXT = "image2text"
    RERANK = "rerank"


# ============================================================================
# common.connection_utils
# ============================================================================

def timeout(seconds: int = 30):
    """타임아웃 데코레이터.

    Args:
        seconds: 타임아웃 시간(초)

    Note:
        현재는 단순 pass-through 데코레이터입니다.
        실제 타임아웃이 필요한 경우 signal 또는 threading으로 구현하세요.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# rag.nlp
# ============================================================================

def find_codec(data: bytes) -> str:
    """바이트 데이터의 인코딩을 감지합니다.

    Args:
        data: 바이트 데이터

    Returns:
        감지된 인코딩 이름 (기본값: utf-8)
    """
    try:
        import chardet
        result = chardet.detect(data)
        return result.get("encoding", "utf-8") or "utf-8"
    except ImportError:
        logger.warning("chardet가 설치되어 있지 않습니다. utf-8로 가정합니다.")
        return "utf-8"


class RagTokenizer:
    """RAGFlow 토크나이저 stub.

    Note:
        실제 구현이 필요한 경우 jieba 또는 다른 토크나이저로 교체하세요.
    """

    def tokenize(self, text: str) -> list[str]:
        """텍스트를 토큰화합니다."""
        # 간단한 공백 기반 토큰화
        return text.split()

    def fine_grained_tokenize(self, text: str) -> list[str]:
        """텍스트를 세밀하게 토큰화합니다."""
        return self.tokenize(text)


rag_tokenizer = RagTokenizer()

# surname 데이터 (한국/중국 성씨)
surname = {
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍",
    # 중국 성씨
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周",
}


# ============================================================================
# rag.app.picture
# ============================================================================

def vision_llm_chunk(
    binary: bytes,
    filename: str = "",
    llm_factory: Optional[str] = None,
    llm_id: Optional[str] = None,
    **kwargs
) -> tuple[str, list]:
    """비전 LLM으로 이미지를 분석합니다.

    Args:
        binary: 이미지 바이너리 데이터
        filename: 파일명
        llm_factory: LLM 팩토리 이름
        llm_id: LLM ID

    Returns:
        (description, chunks) 튜플

    Note:
        실제 구현이 필요합니다.
    """
    logger.warning("vision_llm_chunk은 아직 구현되지 않았습니다.")
    return "", []


# ============================================================================
# rag.prompts.generator
# ============================================================================

def vision_llm_describe_prompt(lang: str = "Korean") -> str:
    """비전 LLM 설명 프롬프트를 생성합니다."""
    return f"이 이미지에 대해 {lang}로 설명해주세요."


def vision_llm_figure_describe_prompt(lang: str = "Korean") -> str:
    """비전 LLM 그림 설명 프롬프트를 생성합니다."""
    return f"이 그림/차트에 대해 {lang}로 설명해주세요."


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # common.file_utils
    "get_project_base_directory",
    "traversal_files",
    # common.misc_utils
    "pip_install_torch",
    # common.constants
    "LLMType",
    # common.connection_utils
    "timeout",
    # rag.nlp
    "find_codec",
    "rag_tokenizer",
    "surname",
    # rag.app.picture
    "vision_llm_chunk",
    # rag.prompts.generator
    "vision_llm_describe_prompt",
    "vision_llm_figure_describe_prompt",
]
