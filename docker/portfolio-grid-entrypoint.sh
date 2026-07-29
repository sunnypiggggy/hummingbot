#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG_PASSWORD:?CONFIG_PASSWORD is required}"
: "${SCRIPT_CONFIG:?SCRIPT_CONFIG is required}"
exec /bin/bash -lc 'conda activate hummingbot && exec ./bin/hummingbot_quickstart.py --config-password "$CONFIG_PASSWORD" --v2 "$SCRIPT_CONFIG" --headless true'
