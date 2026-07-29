# Hermes Conversation Approval Protocol

## Required environment

```dotenv
HERMES_APPROVER_TELEGRAM_USER_ID=<only allowed Telegram user>
HERMES_APPROVER_TELEGRAM_CHAT_ID=<only allowed private chat>
HERMES_CONVERSATION_SURFACE=telegram
```

Keep these values on the trusted local Hermes host. Do not place them in a
proposal, ledger, command line, or OCI audit response.

Hermes gateway owns `TELEGRAM_BOT_TOKEN`. Do not give this Skill a second Bot
token or start a second `getUpdates` loop.

## Approval flow

1. Compute `proposal_sha256` from the canonical proposal fields. Exclude
   `snapshot_id` and `approval`.
2. Run `conversation-request` and read its JSON result.
3. Call Hermes `clarify` with the returned `prompt` and `choices` exactly.
   Telegram renders these choices as native inline buttons through Hermes'
   existing gateway.
4. Pass only the exact `clarify.user_response` into `conversation-resolve`.
   The resolver binds the response to the proposal hash and one-time
   `hermes_interaction_id`.
5. Treat rejection or timeout as no permission to execute.
6. Keep approval valid only for the exact proposal hash and no later than
   `resume_at`.

The legacy direct-Telegram approval adapter waits 12 hours by default. A
proposal still expires at `resume_at` if that occurs sooner.

OCI validates the configured owner IDs, semantic hash, action, time window,
response hash, and interaction uniqueness. The trusted Hermes gateway enforces
the Telegram allowlist; OCI does not query Telegram independently.

## Revocation

Use a new `conversation-request --action revoke` and a new `clarify` prompt.
Reference the original proposal hash. Resolve it, then call `revoke-approved`.
A revocation may request gate restoration, but the OCI hard breaker remains
authoritative and can keep the direction disabled.
