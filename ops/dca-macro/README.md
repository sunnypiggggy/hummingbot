# Hermes DCA Macro Gateway

This deployment keeps live DCA disabled while the gateway runs in shadow mode.
It never changes DCA capital, layers, TP, SL, time limits, or hard circuit
breakers.

## 1. Provision secrets and PKI

Create a private client CA and one Hermes client certificate on the trusted
local machine. Keep the CA private key and client private key off OCI. Copy
only the CA certificate to the Caddy host at:

```text
/etc/caddy/pki/hermes-client-ca.pem
```

For the existing host-network Caddy container, mount the host certificate
read-only and pass the dedicated DNS-only hostname:

```yaml
services:
  caddy:
    volumes:
      - ./pki/hermes-client-ca.pem:/etc/caddy/pki/hermes-client-ca.pem:ro
    environment:
      - HERMES_DCA_DOMAIN=trading-control.sunnypiggy.fun
```

Keep the Cloudflare API token in `.env` or a Docker secret rather than writing
it directly in `docker-compose.yml`.

The repository includes a non-overwriting helper:

```bash
python ops/dca-macro/bootstrap_credentials.py \
  --output-dir /path/outside-the-repository/hermes-dca-credentials
```

Create a JSON HMAC keyring outside the repository using the shape in
`hermes_hmac_secrets.example.json`. Each value must contain at least 32 random
characters. Point `HERMES_HMAC_SECRETS_PATH` at that host file.

For rotation, add the next key to the JSON file, restart only the gateway,
switch Hermes to the new `X-Hermes-Key-Id`, then remove the old key.

## 2. Network isolation

The existing Caddy container uses host networking, so the gateway publishes
only `127.0.0.1:8791` on OCI and also joins `hummingbot-control`. The port is
not reachable through the public interface; only host-network Caddy can proxy
to it.

## 3. Configure Caddy

Add `Caddyfile.example` as a site block to the existing Caddyfile. Set
`HERMES_DCA_DOMAIN` to a dedicated Cloudflare DNS-only hostname. The existing
server uses port 443, so this endpoint intentionally listens on 8443. Caddy
can share `8443` between the existing site and the dedicated DCA hostname by
TLS SNI. Only `/v1/*` is proxied; `/healthz` is not public.

Validate before reload:

```bash
docker exec serverdocker-caddy-1 caddy validate --config /etc/caddy/Caddyfile
docker exec serverdocker-caddy-1 caddy reload --config /etc/caddy/Caddyfile
```

## 4. Start shadow mode

Keep these values:

```dotenv
DCA_LIVE_TRADING_ENABLED=false
DCA_MACRO_EXECUTION_ENABLED=false
```

### Independent Guard emergency channel

An armed Guard now refuses to start unless it has both an independent Binance
signed-REST credential and access to the Docker Engine socket. Create the host
file named by `DCA_BINANCE_EMERGENCY_CREDENTIALS_PATH` outside the repository:

```json
{"api_key":"REPLACE","api_secret":"REPLACE"}
```

Set its mode to `0600`. The key must allow spot trading, must not allow
withdrawals, and should be restricted to the OCI public IP. On startup the
Guard checks server time, account trading permission, both configured symbols,
and Docker socket access without placing orders. Binance's account endpoint
does not reveal the API key's withdrawal permission, so that restriction must
be verified in Binance API Management.

The Docker socket grants host-level container control to the Guard container;
do not add shells, public ports, or unrelated code to that image. During a
breaker the Guard first attempts normal MQTT shutdown, then independently
cancels all Binance orders for the isolated symbol, stops the matching Docker
container, rechecks exchange open orders, rereads strategy fills, and restores
only a remaining strategy-owned inventory delta.

Then start only the gateway:

```bash
docker compose build dca-live-report
docker compose up -d dca-live-report
docker compose -f ops/dca-macro/docker-compose.yml build
docker compose -f ops/dca-macro/docker-compose.yml up -d
docker compose -f ops/dca-macro/docker-compose.yml ps
```

The loopback-only health endpoint is `/healthz`. Public API requests still
require both mTLS and HMAC.

## 5. Hermes client

Set the local-only environment:

```dotenv
DCA_MACRO_GATEWAY_URL=https://trading-control.sunnypiggy.fun:8443
HERMES_HMAC_KEY_ID=primary
HERMES_HMAC_SECRET=<matching secret>
HERMES_CLIENT_CERT=<client certificate path>
HERMES_CLIENT_KEY=<client private key path>
HERMES_APPROVER_TELEGRAM_USER_ID=<owner Telegram user ID>
HERMES_APPROVER_TELEGRAM_CHAT_ID=<owner private chat ID>
HERMES_CONVERSATION_SURFACE=telegram
```

Install the portable Skill into the local Hermes Skill directory:

```powershell
$skillTarget = Join-Path $env:HERMES_SKILLS_DIR "dca-macro-control"
Copy-Item -Recurse -Force hermes/skills/dca-macro-control $skillTarget
```

The repository copy remains the canonical version. The legacy project-local
`.agents/skills/dca-macro-control` has a deny-write ACL and is not modified by
this installation.

Validate the proposal, create a Hermes conversation approval, bind a fresh
snapshot, submit, and inspect the resulting lease:

```bash
python -m macro_control.hermes_cli validate-dossier \
  --dossier ops/dca-macro/decision.example.json
python -m macro_control.hermes_cli conversation-request \
  --dossier /path/to/hermes-authored-proposal.json \
  --output /path/to/approval-request.json
# Hermes calls clarify with the request's prompt and choices, then passes the
# exact user_response:
python -m macro_control.hermes_cli conversation-resolve \
  --request /path/to/approval-request.json \
  --response "<exact clarify user_response>" \
  --output /path/to/approved-decision.json
python -m macro_control.hermes_cli submit-approved \
  --dossier /path/to/approved-decision.json
python -m macro_control.hermes_cli status
python -m macro_control.hermes_cli report
```

Missing, rejected, expired, mismatched, paraphrased, or replayed conversation
responses cannot create a non-neutral lease. `neutral` proposals are
audit-only. Do not run the legacy direct Bot `getUpdates` loop beside Hermes
gateway with the same token.

`report` is read-only. It combines gate and executor status with the isolated
strategy inventory delta, all-time/seven-day mark-to-market profit, and a
seven-day BTC/ETH PNG chart. The collector reads bot SQLite files through a
read-only mount and never reads account-wide balances.

The append-only backtest ledger defaults to
`data/hermes/dca_macro_v3/event_ledger.jsonl`:

```bash
python -m macro_control.hermes_cli ledger-validate
python -m macro_control.hermes_cli ledger-replay \
  --output results/backtests/dca_macro_v3_gate_timeline.json
```

Do not enable execution until shadow decisions, lease expiry, audit output,
paper trading, and the BTC 24-hour canary have passed.

## Single-writer live gate

`DCA_MACRO_EXECUTION_ENABLED` remains `false` after promotion. The gateway
records approvals, maintains leases, and publishes `desired_gates` in
`dca-macro-data/state.json`; it no longer writes controller configuration.
The existing `dca-live-guard` mounts that state read-only, combines it with the
shared v21 BUY contract, and is the sole writer of the final BUY/SELL gates.
This prevents a v21 recovery from reopening BUY while an FOMC lease is active.

The former ROC/SQZMOM BUY guard is retired. v21 is produced once by
`grid-live-guard` and consumed by DCA through the read-only
`grid-live-fdusd-data` mount. Missing, stale, unauthorized, or hash-invalid v21
state fails closed for new BUY orders but never blocks SELL or emergency exits.

## FDUSD live Grid bridge

The FDUSD live Grid scheduler reuses the gateway's approved FOMC leases. It
mounts `dca-macro-data` read-only, converts `state.json` into a minimal
`macro_gate.json`, and publishes that file to every fixed-name
`grid-live-fdusd-400` instance once per scheduler cycle.

During an approved non-neutral FOMC window the Grid cancels only its own open
orders and submits no new orders. It does not flatten inventory solely because
of the event. Normal Grid operation resumes after the lease expires. Missing,
invalid, future-dated, or older-than-150-second macro state fails closed and
keeps the Grid paused until the gateway recovers.

Keep the following setting at or below 180 seconds:

```dotenv
GRID_LIVE_MACRO_MAX_AGE_SECONDS=150
GRID_LIVE_FOMC_EXECUTION_ENABLED=false
```

Keep FOMC execution `false` during shadow validation. A shadow lease is
published for observability but cannot cancel Grid orders. Live Grid deployment
is blocked until this setting is explicitly changed to `true`, the bridge is
fresh and healthy, and no FOMC pause window is active.

This bridge does not relax the existing FDUSD quantitative validation,
private-fee, balance, permission, test-order, or manual deployment gates.
