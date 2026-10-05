FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY cardputer_monitor/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && groupadd --gid 10001 cardputer-monitor \
    && useradd --system --uid 10001 --gid 10001 cardputer-monitor
COPY management_bot/__init__.py management_bot/clients.py management_bot/risk_display.py /app/management_bot/
COPY live_guard/trading_status.py /app/live_guard/trading_status.py
COPY cardputer_monitor/ /app/cardputer_monitor/
USER cardputer-monitor
ENTRYPOINT ["python", "-m", "cardputer_monitor"]
CMD ["--config", "/app/cardputer_monitor/config.example.json"]
