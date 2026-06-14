from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from scripts.monitor_services import (
    CheckResult,
    ServiceCheck,
    append_log,
    build_alerts,
    check_service,
    parse_services,
)


def test_parse_default_services_checks_laravel_and_ai_independently() -> None:
    services = parse_services(None)

    assert services == [
        ServiceCheck("laravel", "http://127.0.0.1:8000/up", False),
        ServiceCheck("ai", "http://127.0.0.1:8001/api/v1/health", True),
    ]


def test_build_alerts_emits_down_and_recovered_state_changes() -> None:
    previous = {
        "laravel": {"ok": True},
        "ai": {"ok": False},
    }
    results = [
        CheckResult("laravel", "http://127.0.0.1:8000/up", False, None, 25, "request_failed=timeout"),
        CheckResult("ai", "http://127.0.0.1:8001/api/v1/health", True, 200, 30, "http_status=200 model_loaded=True"),
    ]

    alerts, next_state = build_alerts(results, previous)

    assert len(alerts) == 2
    assert alerts[0].startswith("DOWN: laravel")
    assert alerts[1].startswith("RECOVERED: ai")
    assert next_state["laravel"]["ok"] is False
    assert next_state["ai"]["ok"] is True


def test_build_alerts_does_not_hide_second_service_failure() -> None:
    results = [
        CheckResult("laravel", "http://127.0.0.1:8000/up", False, None, 25, "request_failed=timeout"),
        CheckResult("ai", "http://127.0.0.1:8001/api/v1/health", False, 503, 30, "http_error=503"),
    ]

    alerts, next_state = build_alerts(results, {})

    assert [alert.split(":", 1)[0] for alert in alerts] == ["DOWN", "DOWN"]
    assert next_state["laravel"]["ok"] is False
    assert next_state["ai"]["ok"] is False


def test_append_log_writes_timestamped_json_line_per_service(tmp_path: Path) -> None:
    log_path = tmp_path / "uptime.log"
    results = [
        CheckResult("laravel", "http://127.0.0.1:8000/up", True, 200, 10, "http_status=200"),
        CheckResult("ai", "http://127.0.0.1:8001/api/v1/health", False, 200, 12, "model_loaded=False"),
    ]

    append_log(log_path, results)

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["service"] for entry in entries] == ["laravel", "ai"]
    assert all(entry["timestamp"] for entry in entries)
    assert entries[0]["ok"] is True
    assert entries[1]["ok"] is False


def test_check_service_requires_ai_model_loaded(monkeypatch) -> None:
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return b'{"status":"ok","model_loaded":false}'

    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=_Response()))

    result = check_service(
        ServiceCheck("ai", "http://127.0.0.1:8001/api/v1/health", True),
        timeout_sec=1,
    )

    assert result.ok is False
    assert "model_loaded=False" in result.detail
