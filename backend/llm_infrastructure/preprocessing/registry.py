"""전처리(preprocessing) 메서드 레지스트리 모듈."""

from typing import Any, Type

from .base import BasePreprocessor


class PreprocessorRegistry:
    """전처리 클래스를 이름/버전으로 등록·조회하는 전역 레지스트리.

    설정(.env, 프리셋 등)에 따라 런타임에 전처리 구현을 교체할 수 있게 해준다.

    예시:
        ```python
        @register_preprocessor("method_a", version="v1")
        class PreprocessorA(BasePreprocessor):
            ...

        preprocessor = get_preprocessor("method_a", version="v1")
        results = preprocessor.preprocess(docs)
        ```
    """

    _registry: dict[str, dict[str, Type[BasePreprocessor]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        preprocessor_cls: Type[BasePreprocessor],
        version: str = "v1",
    ) -> None:
        """전처리 클래스를 레지스트리에 등록한다."""
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"Preprocessor '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = preprocessor_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BasePreprocessor:
        """이름/버전으로 전처리 인스턴스를 생성해 반환한다."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown preprocessing method: '{name}'. "
                f"Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for method '{name}'. "
                f"Available versions: {available_versions}"
            )

        preprocessor_cls = cls._registry[name][version]
        return preprocessor_cls(**kwargs)

    @classmethod
    def list_methods(cls) -> dict[str, list[str]]:
        """등록된 전처리 이름과 버전 목록을 반환한다."""
        return {
            name: list(versions.keys())
            for name, versions in cls._registry.items()
        }


def register_preprocessor(name: str, version: str = "v1"):
    """전처리 클래스를 데코레이터로 등록하기 위한 헬퍼."""
    def decorator(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
        PreprocessorRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_preprocessor(name: str, version: str = "v1", **kwargs: Any) -> BasePreprocessor:
    """전처리 인스턴스를 조회하는 편의 함수."""
    return PreprocessorRegistry.get(name, version=version, **kwargs)
