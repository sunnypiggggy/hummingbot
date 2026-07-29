---
name: dca-macro-control
description: Inspect every running Hummingbot strategy bot, V3 BTC/ETH DCA status, FDUSD Grid health and strategy-owned MTM profit, gates, positions, and seven-day DCA BUY/SELL charts; prepare and conversationally approve macro risk windows. Use for requests about all bots, running bots, Grid status or profit, DCA health, holdings, PnL, recent trades, FOMC, CPI, or NFP.
---

# DCA Macro Control

Treat Hermes output as a proposal, never as permission to trade.

Read [references/event-ledger-v3.md](references/event-ledger-v3.md) before
creating records. Read
[references/telegram-approval.md](references/telegram-approval.md) before
requesting approval or revocation.

## Read-only report

For all running bots, Grid status, or Grid profit requests, run:

```bash
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py bots
```

List every item in `bots`, including status and error count. Summarize Grid
`mtm_pnl_quote`, equity, peak equity, drawdown, per-pair PnL, Guard readiness,
technical BUY gate, and `data_age_seconds`. Call this strategy-owned MTM in
FDUSD, never realized profit or Binance account profit. Warn when `fresh=false`.

For DCA status, position, profit, performance, or recent-trade requests, run:

```bash
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py report
```

Summarize:

- V3 execution switch, BUY/SELL gates, active leases, health, and hard breakers.
- Active and trading executors on each side.
- Strategy-owned BTC/ETH inventory delta. Never call it account inventory or
  a perpetual short position.
- All-time and seven-day mark-to-market profit, recorded fees, and data age.
- Report warnings, including stale or partial data.
- All running strategy bots and the Grid MTM section from `bot_overview`.

If `chart_path` is present, put that absolute PNG path on its own final line so
Hermes sends it as an inline Telegram image. Do not put the path in a code
block. A report query is read-only and never requires approval.

## Control workflow

1. Refresh the official FOMC, CPI, and NFP calendar.
2. Fetch current sanitized telemetry and prepare one `dca-macro-v3` proposal.
3. Preserve Hermes' impact, confidence, reason, evidence, and time window
   exactly; do not repair or reinterpret its judgment.
4. Validate the proposal locally.
5. For `negative` or `positive`, create a conversation approval request. Pass
   its `prompt` as the Hermes `clarify` question and its two `choices` unchanged
   as the `clarify` choices. Hermes renders them as buttons on Telegram.
6. Pass the exact `clarify.user_response` to `conversation-resolve`. Never
   infer approval from prose, `"yes"`, or a paraphrase.
7. On rejection or timeout, append the audit record and stop without calling
   OCI.
8. On approval, bind a fresh `snapshot_id`, submit the exact approved proposal,
   and verify `/v1/status`.
9. To end a lease early, create an `action=revoke` conversation request, repeat
   the `clarify` flow, and call `revoke-approved`. Never bypass an OCI hard
   breaker.

## Commands

Run from the Hummingbot project root:

```bash
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py validate-dossier \
  --dossier /path/proposal.json

python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py conversation-request \
  --dossier /path/proposal.json \
  --output /path/approval-request.json

# Call Hermes clarify with prompt + choices from approval-request.json, then:
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py conversation-resolve \
  --request /path/approval-request.json \
  --response "<exact clarify user_response>" \
  --output /path/approved.json

python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py submit-approved \
  --dossier /path/approved.json

python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py status
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py bots
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py report
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py ledger-validate
python3 hermes/skills/dca-macro-control/scripts/hermes_dca.py ledger-replay
```

Set gateway, mTLS, HMAC, approver user ID, approver chat ID, and conversation
surface through local environment variables. Hermes gateway owns its Telegram
token. Never print or write secrets to the ledger.

When this Skill is installed under Hermes Home instead of inside the project,
set `HUMMINGBOT_PROJECT_ROOT` to the checkout containing `macro_control`.

## Invariants

- Map `negative` to disabling BUY and `positive` to disabling SELL.
- Record `neutral` without approval and without changing a gate.
- Require an exact Hermes conversation approval for every non-neutral proposal
  and revocation. Prefer native `clarify`; do not start a second Telegram
  `getUpdates` consumer beside the Hermes gateway.
- Keep capital, DCA layers, TP, SL, time limit, and guard thresholds immutable.
- Close only executor-owned exposure; never trade account-wide inventory.
- Keep live execution disabled until separately approved.
- Never use a report query to create a proposal, lease, approval, or gate
  update.
- Never omit a running Bot merely because it has no controller performance;
  plain V2 scripts such as Grid legitimately have an empty controller list.
