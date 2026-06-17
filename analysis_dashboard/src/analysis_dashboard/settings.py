"""
analysis_dashboard/settings.py

Centralised settings for the analysis dashboard / log analysis agent.
All values are loaded from the .env file at the project root.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalysisSettings(BaseSettings):
    # LLM
    openrouter_api_key: SecretStr | None = None
    openai_api_key: str = ""
    groq_api_key: SecretStr | None = None
    use_groq: bool = False
    groq_model_name: str = "llama-3.3-70b-versatile"
    model_name: str = "nvidia/nemotron-3-super-120b-a12b:free"
    model_temperature: float = 0.0
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Neo4j Aura DB
    neo4j_uri: str = "neo4j+s://315b1756.databases.neo4j.io"
    neo4j_username: str = "315b1756"
    neo4j_password: SecretStr | None = None

    # SQLite log store path
    log_db_path: str = "mcp_agent_log.db"

    # Charts output directory
    charts_output_dir: str = "charts"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AnalysisSettings()  # type: ignore[call-arg]
