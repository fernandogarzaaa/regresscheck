# Packages the eval runner, golden dataset, prompts and reporting layer.
# Configure with env vars: OPENAI_API_KEY / OPENAI_BASE_URL / MODEL, SLACK_WEBHOOK_URL,
# REGRESSCHECK_PROVIDER (mock|openai), REGRESSCHECK_*_THRESHOLD. Mount /app/.regresscheck to keep history.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY regresscheck ./regresscheck
RUN pip install . && useradd --create-home --uid 10001 rc

COPY regresscheck.toml ./
COPY golden ./golden
COPY prompts ./prompts
COPY baselines ./baselines
COPY demo ./demo
RUN mkdir -p /app/.regresscheck /app/reports && chown -R rc /app/.regresscheck /app/reports
USER rc

ENTRYPOINT ["regresscheck"]
CMD ["check", "--report", "reports/latest.html", "--slack"]
