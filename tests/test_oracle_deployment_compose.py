from __future__ import annotations

from pathlib import Path

import yaml


def _backend_service() -> dict:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["backend"]


def test_ai_service_binds_to_internal_oracle_port_8001() -> None:
    backend = _backend_service()

    assert backend["ports"] == ["127.0.0.1:8001:8000"]


def test_ai_service_uses_production_env_and_restart_always() -> None:
    backend = _backend_service()

    assert ".env.production" in backend["env_file"]
    assert backend["restart"] == "always"


def test_ai_service_mounts_model_artifact_directory_read_only() -> None:
    backend = _backend_service()

    assert "./models:/app/models:ro" in backend["volumes"]


def test_ai_service_healthcheck_requires_loaded_model() -> None:
    backend = _backend_service()
    healthcheck = backend["healthcheck"]
    command = " ".join(str(part) for part in healthcheck["test"])

    assert "127.0.0.1:8000/api/v1/health" in command
    assert "model_loaded" in command
    assert "is True" in command
