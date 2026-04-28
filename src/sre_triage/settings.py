"""Centralized runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    env_url: str = os.getenv("ENV_URL", "http://localhost:7860")
    api_base_url: str = os.getenv(
        "API_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model_name: str = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
    api_key: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
    local_image_name: str = os.getenv("LOCAL_IMAGE_NAME", "sre-mern-env:latest")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))
    workers: int = int(os.getenv("WORKERS", "1"))
    service_name: str = os.getenv("SERVICE_NAME", "sre-triage-env")
    require_api_key: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    service_api_key: str = os.getenv("SERVICE_API_KEY", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    protected_mode: bool = os.getenv("PROTECTED_MODE", "true").lower() == "true"
    benchmark_max_steps: int = int(os.getenv("BENCHMARK_MAX_STEPS", "10"))


settings = Settings()
