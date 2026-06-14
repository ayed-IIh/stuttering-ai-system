#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import smtplib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


DEFAULT_SERVICES = (
    ("laravel", "http://127.0.0.1:8000/up", False),
    ("ai", "http://127.0.0.1:8001/api/v1/health", True),
)


@dataclass(frozen=True)
class ServiceCheck:
    name: str
    url: str
    require_model_loaded: bool = False


@dataclass(frozen=True)
class CheckResult:
    name: str
    url: str
    ok: bool
    status_code: int | None
    latency_ms: int
    detail: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_services(raw: str | None) -> list[ServiceCheck]:
    if not raw:
        return [
            ServiceCheck(name, url, require_model_loaded)
            for name, url, require_model_loaded in DEFAULT_SERVICES
        ]
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("MONITOR_SERVICES must be a non-empty JSON array")

    services: list[ServiceCheck] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each MONITOR_SERVICES item must be an object")
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not name or not url:
            raise ValueError("Each MONITOR_SERVICES item must include name and url")
        services.append(
            ServiceCheck(
                name=name,
                url=url,
                require_model_loaded=bool(item.get("require_model_loaded", False)),
            )
        )
    return services


def check_service(service: ServiceCheck, timeout_sec: float) -> CheckResult:
    started = time.perf_counter()
    status_code: int | None = None
    detail = ""
    ok = False

    try:
        with urllib.request.urlopen(service.url, timeout=timeout_sec) as response:
            status_code = int(response.status)
            body = response.read(64 * 1024)
        ok = 200 <= status_code < 300
        detail = f"http_status={status_code}"

        if ok and service.require_model_loaded:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                ok = False
                detail = f"invalid_json={exc}"
            else:
                model_loaded = payload.get("model_loaded")
                ok = model_loaded is True
                detail = f"http_status={status_code} model_loaded={model_loaded!r}"
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        detail = f"http_error={exc.code}"
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        detail = f"request_failed={exc}"
    except Exception as exc:
        detail = f"unexpected_error={exc.__class__.__name__}: {exc}"

    latency_ms = int((time.perf_counter() - started) * 1000)
    return CheckResult(
        name=service.name,
        url=service.url,
        ok=ok,
        status_code=status_code,
        latency_ms=latency_ms,
        detail=detail,
    )


def load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_log(path: Path, results: list[CheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            entry = {
                "timestamp": utc_now(),
                "service": result.name,
                "url": result.url,
                "ok": result.ok,
                "status_code": result.status_code,
                "latency_ms": result.latency_ms,
                "detail": result.detail,
            }
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def build_alerts(
    results: list[CheckResult],
    previous_state: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    next_state = dict(previous_state)
    alerts: list[str] = []
    checked_at = utc_now()

    for result in results:
        prev = previous_state.get(result.name, {})
        was_ok = bool(prev.get("ok", True))
        if not result.ok and was_ok:
            alerts.append(
                f"DOWN: {result.name} health check failed at {checked_at}. "
                f"url={result.url} detail={result.detail}"
            )
        elif result.ok and not was_ok:
            alerts.append(
                f"RECOVERED: {result.name} health check passed at {checked_at}. "
                f"url={result.url} detail={result.detail}"
            )

        next_state[result.name] = {
            "ok": result.ok,
            "last_checked_at": checked_at,
            "last_detail": result.detail,
        }

    return alerts, next_state


def send_email_alert(subject: str, body: str) -> bool:
    to_addr = os.getenv("MONITOR_ALERT_EMAIL_TO", "").strip()
    if not to_addr:
        return False

    from_addr = os.getenv("MONITOR_ALERT_EMAIL_FROM", "monitor@localhost").strip()
    smtp_host = os.getenv("MONITOR_SMTP_HOST", "").strip()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    if smtp_host:
        smtp_port = int(os.getenv("MONITOR_SMTP_PORT", "25"))
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
            username = os.getenv("MONITOR_SMTP_USERNAME", "").strip()
            password = os.getenv("MONITOR_SMTP_PASSWORD", "").strip()
            if os.getenv("MONITOR_SMTP_STARTTLS", "false").lower() == "true":
                smtp.starttls()
            if username:
                smtp.login(username, password)
            smtp.send_message(msg)
        return True

    sendmail = os.getenv("MONITOR_SENDMAIL", "/usr/sbin/sendmail").strip()
    try:
        subprocess.run(
            [sendmail, "-t", "-oi"],
            input=msg.as_bytes(),
            check=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        sys.stderr.write(f"sendmail timeout after {exc.timeout}s: {exc}\n")
        raise
    return True


def send_webhook_alert(body: str) -> bool:
    webhook_url = os.getenv("MONITOR_ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False

    payload = json.dumps({"text": body}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return 200 <= int(response.status) < 300


def send_alerts(alerts: list[str]) -> None:
    if not alerts:
        return

    hostname = socket.gethostname()
    subject = f"[stuttering-ai] service health alert on {hostname}"
    body = "\n".join(alerts)
    sent = False
    errors: list[str] = []

    for sender in (
        lambda: send_webhook_alert(body),
        lambda: send_email_alert(subject, body),
    ):
        try:
            sent = sender() or sent
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")

    if not sent:
        error_text = "; ".join(errors) if errors else "no alert destination configured"
        raise RuntimeError(f"Failed to send health alert: {error_text}")


def acquire_lock(path: Path, stale_after_sec: int) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return acquire_lock(path, stale_after_sec)
        if age > stale_after_sec:
            path.unlink(missing_ok=True)
            return acquire_lock(path, stale_after_sec)
        return None

    os.write(fd, str(os.getpid()).encode("ascii"))
    return fd


def release_lock(path: Path, fd: int | None) -> None:
    if fd is None:
        return
    os.close(fd)
    path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Laravel and AI service health endpoints.")
    parser.add_argument("--log-file", default=os.getenv("MONITOR_LOG_FILE", "/var/log/stuttering-ai/uptime.log"))
    parser.add_argument(
        "--state-file",
        default=os.getenv("MONITOR_STATE_FILE", "/var/lib/stuttering-ai/monitor_state.json"),
    )
    parser.add_argument("--lock-file", default=os.getenv("MONITOR_LOCK_FILE", "/tmp/stuttering-ai-monitor.lock"))
    parser.add_argument("--timeout-sec", type=float, default=float(os.getenv("MONITOR_TIMEOUT_SEC", "5")))
    parser.add_argument("--stale-lock-sec", type=int, default=int(os.getenv("MONITOR_STALE_LOCK_SEC", "600")))
    args = parser.parse_args()

    lock_path = Path(args.lock_file)
    lock_fd = acquire_lock(lock_path, args.stale_lock_sec)
    if lock_fd is None:
        return 0

    try:
        services = parse_services(os.getenv("MONITOR_SERVICES"))
        results = [check_service(service, args.timeout_sec) for service in services]
        append_log(Path(args.log_file), results)
        previous_state = load_state(Path(args.state_file))
        alerts, next_state = build_alerts(results, previous_state)
        save_state(Path(args.state_file), next_state)
        send_alerts(alerts)
        return 0
    except Exception as exc:
        timestamp = utc_now()
        sys.stderr.write(f"{timestamp} monitor_error={exc.__class__.__name__}: {exc}\n")
        return 1
    finally:
        release_lock(lock_path, lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
