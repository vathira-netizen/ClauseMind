"""Application configuration.

Flat, env-driven settings read via ``pydantic-settings``. Defaults are
sensible for local development against the docker-compose stack (Qdrant +
Jaeger). Access the singleton via :func:`get_settings`, cached so the
environment is parsed exactly once per process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Root application settings.

    Field names map to environment variables of the same name
    (case-insensitive), e.g. ``qdrant_url`` reads ``QDRANT_URL``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant REST endpoint."
    )
    google_api_key: SecretStr = Field(
        default=SecretStr(""), description="Google ADK / Gemini API key."
    )
    lyzr_api_key: SecretStr = Field(
        default=SecretStr(""), description="Lyzr governance layer API key."
    )
    model_name: str = Field(
        default="gemini-2.0-flash", description="Default LLM used by ADK agents."
    )
    jwt_secret: str = Field(
        default="dev-secret-change-me",
        description="Secret used to sign API JWTs. Override in staging/production.",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint (Jaeger in local dev).",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""

    return Settings()
