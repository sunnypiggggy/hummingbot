# Condor removal and Telegram management Bot migration

Condor previously used the `hummingbot-condor` container to reach the internal
Hummingbot API. It was removed from the production Compose definition when the
private `sunnypiggy-trade-bot` service was introduced.

The retired configuration contained one API target named `main`, the internal
host `hummingbot-api:8000`, and a single primary administrator. Credentials,
Telegram tokens, user IDs, chat history, and audit entries are deliberately not
preserved in this archive.

Migration invariants:

- `sunnypiggy-trade-bot` is the only `getUpdates` consumer for the management
  Bot token.
- `dca-live-report` remains the only Telegram channel notification sender and
  uses a different Docker secret.
- The management Bot has no Docker socket or exchange credentials.
- Condor images may be removed only after `docker inspect` confirms that no
  container references them.
- On the first management Bot start, the old webhook and queued Condor updates
  are discarded once; subsequent restarts resume from the SQLite update offset.

