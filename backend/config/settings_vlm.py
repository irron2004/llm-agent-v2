"""Separate settings for VLM (vision/multimodal LLM) if needed."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class VLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = Field(default="http://localhost:8004", description="VLM OpenAI-compatible endpoint")
    model_name: str = Field(default="", description="VLM model id")
    timeout: int = Field(default=600, description="Request timeout seconds")
    max_tokens: int = Field(default=2048, description="Max tokens for VLM responses")
    temperature: float = Field(default=0.0, description="Generation temperature")


vlm_settings = VLMSettings()


__all__ = ["vlm_settings", "VLMSettings"]

