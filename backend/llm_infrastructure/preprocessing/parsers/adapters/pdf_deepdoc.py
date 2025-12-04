"""DeepDoc PDF 파서 어댑터.

DeepDocPdfEngine을 BaseParser 인터페이스로 래핑하여 파서 레지스트리에 등록합니다.
"""

from __future__ import annotations

from typing import BinaryIO, Optional

from ..base import BaseParser, ParsedDocument, PdfParseOptions
from ..engines.pdf_deepdoc_engine import DeepDocPdfEngine
from ..registry import register_parser


class DeepDocPdfAdapter(BaseParser):
    """DeepDoc PDF 파서 어댑터.

    DeepDocPdfEngine을 BaseParser 프로토콜에 맞게 래핑합니다.
    파서 레지스트리를 통해 "pdf_deepdoc" 이름으로 접근 가능합니다.

    Attributes:
        content_type: 처리 가능한 MIME 타입
        engine: 실제 파싱을 수행하는 DeepDocPdfEngine 인스턴스
    """

    content_type: str = "application/pdf"

    def __init__(self, engine: Optional[DeepDocPdfEngine] = None) -> None:
        """어댑터를 초기화합니다.

        Args:
            engine: DeepDocPdfEngine 인스턴스. 미제공 시 기본 생성.
        """
        self.engine = engine or DeepDocPdfEngine()

    def parse(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        """PDF 파일을 파싱합니다.

        Args:
            file: PDF 파일의 바이너리 스트림
            options: 파싱 옵션 (preferred_backend 필수)

        Returns:
            파싱된 문서
        """
        return self.engine.run(file, options=options)


register_parser("pdf_deepdoc", DeepDocPdfAdapter)

__all__ = ["DeepDocPdfAdapter"]
