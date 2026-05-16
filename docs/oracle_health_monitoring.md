# Oracle Health Monitoring

This project uses a small cron-based monitor for the Oracle Cloud deployment.
It checks both services every five minutes:

- Laravel: `http://127.0.0.1:8000/up`
- AI service: `http://127.0.0.1:8001/api/v1/health`

The AI health check is considered healthy only when the endpoint returns HTTP
2xx and `model_loaded` is `true`.

## Install

Run these commands on the Oracle server from the repository root:

```bash
sudo mkdir -p /var/log/stuttering-ai /var/lib/stuttering-ai
sudo touch /var/log/stuttering-ai/uptime.log /var/log/stuttering-ai/monitor-cron.log
sudo chown -R "$USER":"$USER" /var/log/stuttering-ai /var/lib/stuttering-ai
chmod +x scripts/monitor_services.py
(crontab -l 2>/dev/null; cat ops/cron/stuttering-ai-health-monitor.cron) | crontab -
```

The cron entry runs every five minutes.

## Alerts

Configure at least one alert destination in the server environment used by
cron. The script supports either webhook alerts or email alerts.

Webhook:

```bash
export MONITOR_ALERT_WEBHOOK_URL="https://hooks.example.com/services/..."
```

Email through local sendmail:

```bash
export MONITOR_ALERT_EMAIL_TO="team@example.com"
export MONITOR_ALERT_EMAIL_FROM="monitor@your-domain.example"
```

Email through SMTP:

Store secrets in a restricted environment file to avoid exposing them in shell
history. Create a file such as `~/.monitor_env` with permissions set to 600:

```bash
# ~/.monitor_env
export MONITOR_ALERT_EMAIL_TO="team@example.com"
export MONITOR_ALERT_EMAIL_FROM="monitor@your-domain.example"
export MONITOR_SMTP_HOST="smtp.example.com"
export MONITOR_SMTP_PORT="587"
export MONITOR_SMTP_STARTTLS="true"
export MONITOR_SMTP_USERNAME="smtp-user"
export MONITOR_SMTP_PASSWORD="smtp-password"
```

Set permissions and source from your shell or cron:

```bash
chmod 600 ~/.monitor_env
source ~/.monitor_env
```

**Note:** Avoid exporting secrets (MONITOR_SMTP_PASSWORD, MONITOR_SMTP_USERNAME,
MONITOR_SMTP_HOST, MONITOR_SMTP_PORT, MONITOR_SMTP_STARTTLS,
MONITOR_ALERT_EMAIL_TO, MONITOR_ALERT_EMAIL_FROM) directly in your shell to
prevent them from being saved in shell history.

To make these variables available to cron, place them above the cron command in
`ops/cron/stuttering-ai-health-monitor.cron` on the server before installing it,
or source them from an environment file readable by the user running that cron
job in the cron command.

## Logs And State

The monitor writes one JSON line per service per run to:

```text
/var/log/stuttering-ai/uptime.log
```

It stores alert state in:

```text
/var/lib/stuttering-ai/monitor_state.json
```

Alerts are sent only when a service changes state:

- healthy to unhealthy: `DOWN`
- unhealthy to healthy: `RECOVERED`

This prevents repeated alert spam while a service remains down, while still
confirming recovery after restart.

## Manual Verification

From the Oracle server:

```bash
python3 scripts/monitor_services.py
tail -n 20 /var/log/stuttering-ai/uptime.log
```

Stop the AI service and wait up to five minutes:

```bash
docker compose stop backend
```

Expected result: a `DOWN` alert for the AI service and a timestamped failed
entry in `uptime.log`.

Restart it:

```bash
docker compose up -d backend
```

Expected result: a `RECOVERED` alert after the next successful cron run.
