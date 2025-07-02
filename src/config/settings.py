# src/config/settings.py
import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")  # Added for query rewriting

    # Vector Database
    weaviate_url: str = Field(..., env="WEAVIATE_URL")
    weaviate_api_key: str = Field(..., env="WEAVIATE_API_KEY")

    # Search
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")

    # LangSmith
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT"
    )
    langchain_api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("vietnamese_rag", env="LANGCHAIN_PROJECT")

    # Model Settings
    embedding_model: str = Field(
        "AITeamVN/Vietnamese_Embedding_v2", env="EMBEDDING_MODEL"
    )
    chunk_size: int = Field(1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    top_k: int = Field(5, env="TOP_K")
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")

    # Collections
    wikipedia_collection: str = "VietnameseWikipedia"

    # Query Rewriting Settings
    max_rewrite_attempts: int = Field(3, env="MAX_REWRITE_ATTEMPTS")
    enable_query_logging: bool = Field(True, env="ENABLE_QUERY_LOGGING")

    # Add ROOT path property
    @property
    def ROOT(self) -> Path:
        """Get the root directory of the project."""
        return Path(__file__).parent.parent.parent

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
