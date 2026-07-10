#!/usr/bin/env bash
set -euo pipefail

cat > /opt/condor/config.yml <<EOF
servers:
  main:
    host: ${CONDOR_API_HOST:-hummingbot-api}
    port: ${CONDOR_API_PORT:-8000}
    username: ${HUMMINGBOT_API_USERNAME:-admin}
    password: ${HUMMINGBOT_API_PASSWORD:-admin}
default_server: main
admin_id: ${ADMIN_USER_ID:-918010832}
users:
  ${ADMIN_USER_ID:-918010832}:
    user_id: ${ADMIN_USER_ID:-918010832}
    role: admin
    notes: Primary admin from ADMIN_USER_ID
server_access:
  main:
    owner_id: ${ADMIN_USER_ID:-918010832}
    created_at: 0
    shared_with: {}
chat_defaults:
  ${ADMIN_USER_ID:-918010832}: main
audit_log: []
EOF

exec uv run python main.py
