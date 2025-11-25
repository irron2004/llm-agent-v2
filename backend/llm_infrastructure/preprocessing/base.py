"""Base classes for preprocessing."""

from abc import ABC, abstractmethod
from typing import Any, Iterable


class BasePreprocessor(ABC):
    """Base class for all preprocessing methods.

    Each preprocessing method should:
    1. Inherit from this class
    2. Implement the preprocess() method
    3. Register itself using @register_preprocessor decorator

    Example:
        ```python
        from .registry import register_preprocessor

        @register_preprocessor("my_method", version="v1")
        class MyPreprocessor(BasePreprocessor):
            def preprocess(self, docs):
                for doc in docs:
                    yield doc.lower()  # Example transformation
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize preprocessor with optional config."""
        self.config = kwargs

    @abstractmethod
    def preprocess(self, docs: Iterable[str]) -> Iterable[str]:
        """Preprocess documents.

        Args:
            docs: Input documents (strings or dict with 'content' field)

        Yields:
            Preprocessed documents
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
