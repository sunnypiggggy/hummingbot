#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/ubuntu/extra_drive/hummingbot}
PAYLOAD=${2:-/tmp/stocks-limits-a0b17d104.tar}
TAG=${3:-a0b17d104}
BACKUP="$ROOT/.deployment-backups/stocks-$TAG"
POSTGRES_CONTAINER=hummingbot-api-postgres
REPORT_CONTAINER=sunnypiggy-trade-bot

[[ -f "$PAYLOAD" ]] || { echo "deployment payload missing: $PAYLOAD" >&2; exit 2; }
sudo mkdir -p "$BACKUP"

mapfile -t payload_files < <(tar -tf "$PAYLOAD" | sed '/\/$/d')
existing=()
for file in "${payload_files[@]}"; do
  [[ -e "$ROOT/$file" ]] && existing+=("$file")
done
if ((${#existing[@]})); then
  sudo tar -cf "$BACKUP/source-before.tar" -C "$ROOT" "${existing[@]}"
fi

docker exec "$POSTGRES_CONTAINER" sh -lc \
  'pg_dump -U "$POSTGRES_USER" -d hummingbot_stocks -Fc -f /tmp/stocks-release.dump'
docker cp "$POSTGRES_CONTAINER:/tmp/stocks-release.dump" /tmp/stocks-release.dump
sudo mv /tmp/stocks-release.dump "$BACKUP/postgres.dump"
docker exec "$POSTGRES_CONTAINER" rm /tmp/stocks-release.dump

docker exec "$REPORT_CONTAINER" python -c \
  'import sqlite3; s=sqlite3.connect("/state/management_bot.sqlite"); d=sqlite3.connect("/state/management_bot.pre-deploy.sqlite"); s.backup(d); d.close(); s.close()'
sudo mv "$ROOT/telegram-management-data/management_bot.pre-deploy.sqlite" "$BACKUP/management_bot.sqlite"

docker exec "$POSTGRES_CONTAINER" sh -lc \
  'psql -U "$POSTGRES_USER" -d hummingbot_stocks -At -F , -c "SELECT schedule_id,executor_id,status FROM binance_stocks_paper.scheduled_executors ORDER BY schedule_id"' \
  | sudo tee "$BACKUP/schedules-before.csv" >/dev/null

tar -xf "$PAYLOAD" -C "$ROOT"

set_env() {
  local key=$1 value=$2
  if grep -q "^${key}=" "$ROOT/.env.control"; then
    sed -i "s/^${key}=.*/${key}=${value}/" "$ROOT/.env.control"
  else
    printf '%s=%s\n' "$key" "$value" >> "$ROOT/.env.control"
  fi
}
set_env BINANCE_STOCKS_MAX_ORDER_USDC 500
set_env BINANCE_STOCKS_MAX_SYMBOL_USDC 1000
set_env BINANCE_STOCKS_MAX_EXPOSURE_USDC 2000
set_env BINANCE_STOCKS_DAILY_LOSS_USDC 200

# Docker Compose variable interpolation reads .env (not service env_file).
# This path contains no credential material; it selects the existing OCI secret.
touch "$ROOT/.env"
if grep -q '^BINANCE_STOCKS_CREDENTIALS_PATH=' "$ROOT/.env"; then
  sed -i 's|^BINANCE_STOCKS_CREDENTIALS_PATH=.*|BINANCE_STOCKS_CREDENTIALS_PATH=/home/ubuntu/secrets/binance_stocks_credentials.json|' "$ROOT/.env"
else
  printf '%s\n' 'BINANCE_STOCKS_CREDENTIALS_PATH=/home/ubuntu/secrets/binance_stocks_credentials.json' >> "$ROOT/.env"
fi

docker exec -i "$POSTGRES_CONTAINER" sh -lc \
  'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d hummingbot_stocks' \
  < "$ROOT/scripts/migrate_binance_stocks_limits_v3.sql"

cd "$ROOT"
docker compose --profile stocks config --quiet
docker compose --profile stocks build binance-stocks-runtime
docker compose build sunnypiggy-trade-bot
docker compose --profile stocks up -d --no-deps binance-stocks-runtime
docker compose up -d --no-deps sunnypiggy-trade-bot

docker exec "$POSTGRES_CONTAINER" sh -lc \
  'psql -U "$POSTGRES_USER" -d hummingbot_stocks -At -F , -c "SELECT schedule_id,executor_id,status FROM binance_stocks_paper.scheduled_executors ORDER BY schedule_id"' \
  | sudo tee "$BACKUP/schedules-after.csv" >/dev/null

cmp "$BACKUP/schedules-before.csv" "$BACKUP/schedules-after.csv"
sudo sha256sum "$BACKUP"/* | sudo tee "$BACKUP/SHA256SUMS" >/dev/null
echo "deployment complete; backup=$BACKUP"
