# DCA Event Ledger V3

Use the append-only UTF-8 JSONL file:

```text
data/hermes/dca_macro_v3/event_ledger.jsonl
```

Validate every line against
[event-ledger-v3.schema.json](event-ledger-v3.schema.json). Do not migrate,
append, or replay V2 records.

## Common fields

Every line is one JSON object containing:

- `schema_version`: always `hermes-dca-event-ledger-v3`
- `record_type`: `event`, `proposal`, `approval`, `execution`, or `revocation`
- `record_id`: globally unique and immutable
- `recorded_at`: timezone-aware ISO-8601 timestamp

## Record order

Append records in causal order:

1. `event` records the official calendar source.
2. `proposal` records Hermes' immutable semantics and `proposal_sha256`.
3. `approval` records a Hermes conversation or legacy direct-Telegram approval
   or rejection.
4. `execution` records the fresh snapshot and OCI result.
5. `revocation` records a separately approved early end.

Never edit an existing line. A rejected proposal remains part of the audit.
Use a new proposal ID and new approval if any semantic field changes.

## Canonical proposal hash

Hash UTF-8 JSON serialized with sorted keys, no insignificant whitespace, and
unescaped Unicode. Include:

`decision_id`, event identifiers and source, all decision/window timestamps,
`market_impact`, `confidence`, `reason`, `evidence`, Hermes metadata, and
`policy_version`.

Exclude `snapshot_id` and `approval`. This permits binding a fresh telemetry
snapshot after the owner approves the decision semantics.

## Replay

- Replay only approved non-neutral proposals with a matching hash.
- Start a late-approved lease at `max(effective_at, approved_at)`.
- Label leases `on_time` or `late_approval` and report both counts.
- Union overlapping impacts. Active negative and positive leases disable both
  directions.
- Truncate a lease only with a valid approved revocation.
- Output gate intervals only; price and PnL calculations belong to the
  consuming backtester.
