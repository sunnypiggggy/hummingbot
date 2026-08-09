# Risk recovery approval

This workflow is only for a notification event that explicitly has
`requires_manual_action=true`. A public Telegram channel post is a pointer to
the event, not an approval surface.

## Required binding

Preserve the event ID, robot, strategy, pair, mechanism, phase, release/model/
parameter hashes, trigger reason and source. Read the canonical OCI JSONL event
and require it to match the prompt. Refuse an incomplete ID, the wrong schema,
or a phase other than `LATCHED` or a `REENTRY` explicitly blocked by the
automatic-reentry switch.

## Read-only preflight

Before creating an approval, verify:

1. Forced exit is complete and remaining owned risk is below the current
   exchange minimum; dust is recorded.
2. There are no robot-owned active orders and no active/trading DCA executor.
3. Grid reservations or DCA managed inventory reconcile to fills, and ownership
   does not exceed the actual account balance.
4. The v22 contract, market data and filters are fresh; package/model hashes and
   the current signed week are valid.
5. Every other enabled risk gate is open. Recovery cannot clear another gate.
6. Quote funds are sufficient for the configured per-robot reentry.

Return this evidence and its snapshot hash. If any check fails or the recovery
adapter is unavailable, stop. Never directly edit Guard state, controller gates
or authorization files.

## Private approval

Build a one-time proposal bound to the event ID, canonical event hash, fresh
preflight snapshot hash and reset scope. Present its exact prompt and two exact
choices through Hermes `clarify` in the owner's private chat. Timeout, prose, a
channel reply or a hash mismatch is rejection.

After exact approval, submit the proof to the OCI recovery adapter. It must
consume the proof once, clear only the named latch or authorize only the named
reentry, and leave every other gate unchanged. Read status again and report the
consumed approval hash, resulting phase, remaining gates, active orders/
executors and owned inventory.

Never expose either Bot token, account balances, CLI/HMAC secrets or direct
reset commands in the channel prompt or audit ledger.
