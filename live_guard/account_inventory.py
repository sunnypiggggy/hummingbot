"""Transactional shared-account inventory ownership and liquidation ledger.

The Grid and DCA guards use the same Binance spot account.  This module keeps
their base-asset ownership in one SQLite database so independent emergency
paths cannot both sell the same BTC/ETH.  It deliberately contains no trading
API calls; callers must acquire an asset lease, refresh Binance, and then
record the resulting exchange order.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA = "account-inventory-status-v3"
ASSETS = ("BTC", "ETH")
ZERO = Decimal("0")


def decimal(value: Any) -> Decimal:
    return Decimal(str(value if value is not None else "0"))


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def api_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def ownership_from_documents(
    *, reservations: Mapping[str, Any], grid_state: Mapping[str, Any],
    managed_inventory: Mapping[str, Any], dca_state: Mapping[str, Any],
) -> dict[str, dict[str, Decimal]]:
    """Derive current strategy ownership without using account-wide balances."""
    result = {asset: {} for asset in ASSETS}
    grid_bot = next(iter(grid_state.get("bots", {}).values()), {})
    grid_pairs = grid_bot.get("latest", {}).get("pairs", {})
    grid_bases = reservations.get("reservations", {}).get("FDUSD", {}).get("base", {})
    for asset in ASSETS:
        pair = f"{asset}-FDUSD"
        # Guard snapshots already fold emergency fills into net_base.
        quantity = (
            decimal(grid_bases.get(asset))
            + decimal(grid_pairs.get(pair, {}).get("net_base"))
        )
        result[asset]["grid:grid-live-fdusd-400"] = max(quantity, ZERO)

    pair_docs = managed_inventory.get("pairs", {})
    dca_bots = dca_state.get("bots", {})
    for asset in ASSETS:
        pair = f"{asset}-USDT"
        bot_name = f"dca-live-{asset.lower()}usdt-200"
        bot = dca_bots.get(bot_name, {})
        target = bot.get("managed_base_target")
        if target is None:
            target = pair_docs.get(pair, {}).get("managed_base")
        # Guard snapshots already fold emergency fills into net_base.
        quantity = (
            decimal(target)
            + decimal(bot.get("latest", {}).get("net_base"))
        )
        result[asset][f"dca:{bot_name}"] = max(quantity, ZERO)
    return result


class UnifiedInventoryLedger:
    """SQLite-backed ownership, confirmation, lease, and order audit store."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.directory / "account_inventory.sqlite"
        self.status_path = self.directory / "account_inventory_status.json"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS owners (
                    asset TEXT NOT NULL, owner_key TEXT NOT NULL,
                    quantity TEXT NOT NULL, evidence_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(asset, owner_key)
                );
                CREATE TABLE IF NOT EXISTS confirmations (
                    asset TEXT PRIMARY KEY, quantity TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL, first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL, cycles INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deficit_confirmations (
                    asset TEXT PRIMARY KEY, first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL, cycles INTEGER NOT NULL,
                    peak_deficit TEXT NOT NULL, notified INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS inventory_episodes (
                    asset TEXT PRIMARY KEY, episode_id TEXT NOT NULL,
                    phase TEXT NOT NULL, quantity TEXT NOT NULL,
                    stability_sha256 TEXT NOT NULL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, last_notified_transition TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS leases (
                    asset TEXT PRIMARY KEY, holder TEXT NOT NULL,
                    acquired_at REAL NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS liquidation_jobs (
                    job_id TEXT PRIMARY KEY, asset TEXT NOT NULL,
                    scope TEXT NOT NULL, pair TEXT NOT NULL,
                    requested_quantity TEXT NOT NULL,
                    client_order_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, exchange_order_id TEXT,
                    executed_quantity TEXT, quote_quantity TEXT,
                    fee_quote TEXT, fee_details TEXT, error TEXT, created_at REAL NOT NULL,
                    verification_json TEXT, verified_at REAL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS liquidation_attempts (
                    job_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    client_order_id TEXT NOT NULL UNIQUE,
                    requested_quantity TEXT NOT NULL, status TEXT NOT NULL,
                    exchange_order_id TEXT, executed_quantity TEXT,
                    quote_quantity TEXT, response_json TEXT, error TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(job_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
                    delivered_at REAL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bootstrap_caps (
                    asset TEXT PRIMARY KEY, quantity TEXT NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(liquidation_jobs)"
                )
            }
            if "fee_details" not in columns:
                connection.execute(
                    "ALTER TABLE liquidation_jobs ADD COLUMN fee_details TEXT"
                )
            if "verification_json" not in columns:
                connection.execute(
                    "ALTER TABLE liquidation_jobs ADD COLUMN verification_json TEXT"
                )
            if "verified_at" not in columns:
                connection.execute(
                    "ALTER TABLE liquidation_jobs ADD COLUMN verified_at REAL"
                )
            event_columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(events)"
                )
            }
            if "delivered" not in event_columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN delivered INTEGER NOT NULL DEFAULT 1"
                )
            if "delivered_at" not in event_columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN delivered_at REAL"
                )

    @staticmethod
    def _stability_sha256(
        *, account_fingerprint: str, asset: str, total: Decimal,
        owners: Mapping[str, Decimal], unattributed: Decimal,
    ) -> str:
        return canonical_sha256({
            "account_fingerprint": account_fingerprint,
            "asset": asset,
            "exchange_total": str(total),
            "owners": {key: str(value) for key, value in sorted(owners.items())},
            "unattributed": str(unattributed),
        })

    @staticmethod
    def _episode_id(
        *, account_fingerprint: str, asset: str,
        stability_sha256: str, created_at: float,
    ) -> str:
        return canonical_sha256({
            "account_fingerprint": account_fingerprint,
            "asset": asset,
            "stability_sha256": stability_sha256,
            "created_at": created_at,
        })

    @staticmethod
    def _dust_job_covers(
        connection: sqlite3.Connection, asset: str, quantity: Decimal,
    ) -> bool:
        row = connection.execute(
            "SELECT requested_quantity FROM liquidation_jobs "
            "WHERE asset=? AND scope='unattributed_dust' AND status='DUST' "
            "ORDER BY updated_at DESC LIMIT 1",
            (asset,),
        ).fetchone()
        if row is None:
            return False
        tolerance = max(abs(quantity) * Decimal("1e-12"), Decimal("1e-18"))
        return decimal(row["requested_quantity"]) + tolerance >= quantity

    def set_episode_phase(
        self, asset: str, phase: str, *, now: float | None = None,
        reset_confirmation: bool = False,
    ) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "UPDATE inventory_episodes SET phase=?,updated_at=? WHERE asset=?",
                (phase, now, asset),
            )
            if reset_confirmation:
                connection.execute("DELETE FROM confirmations WHERE asset=?", (asset,))

    def mark_episode_notified(
        self, asset: str, transition: str, *, now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "UPDATE inventory_episodes SET last_notified_transition=?,updated_at=? "
                "WHERE asset=?",
                (transition, now, asset),
            )

    def write_status(self, payload: Mapping[str, Any]) -> None:
        _atomic_json(self.status_path, payload)

    def set_bootstrap_caps(self, caps: Mapping[str, Any], *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            for asset, value in caps.items():
                connection.execute(
                    "INSERT OR IGNORE INTO bootstrap_caps(asset,quantity,consumed,updated_at) "
                    "VALUES(?,?,0,?)",
                    (asset, str(decimal(value)), now),
                )

    def bind_account(self, account_fingerprint: str) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM meta WHERE key='account_fingerprint'"
            ).fetchone()
            if row is not None and row["value"] != account_fingerprint:
                raise RuntimeError("shared inventory account fingerprint changed")
            connection.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('account_fingerprint',?)",
                (account_fingerprint,),
            )

    def bootstrap_cap(self, asset: str) -> Decimal | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT quantity,consumed FROM bootstrap_caps WHERE asset=?", (asset,)
            ).fetchone()
        if row is None or int(row["consumed"]):
            return None
        return decimal(row["quantity"])

    def consume_bootstrap_cap(self, asset: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "UPDATE bootstrap_caps SET consumed=1,updated_at=? WHERE asset=?",
                (now, asset),
            )

    def reconcile(
        self, *, account_fingerprint: str,
        balances: Mapping[str, Mapping[str, Any]],
        ownership: Mapping[str, Mapping[str, Any]], evidence_sha256: str,
        open_order_counts: Mapping[str, int], sources_healthy: bool,
        confirmation_cycles: int = 3, confirmation_seconds: int = 30,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT value FROM meta WHERE key='account_fingerprint'"
            ).fetchone()
            if existing is not None and existing["value"] != account_fingerprint:
                raise RuntimeError("shared inventory account fingerprint changed")
            connection.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('account_fingerprint',?)",
                (account_fingerprint,),
            )
            assets: dict[str, Any] = {}
            total_orders = sum(int(value) for value in open_order_counts.values())
            healthy = bool(sources_healthy and total_orders == 0)
            for asset in ASSETS:
                owner_rows = {
                    str(key): max(decimal(value), ZERO)
                    for key, value in ownership.get(asset, {}).items()
                }
                connection.execute("DELETE FROM owners WHERE asset=?", (asset,))
                for owner_key, quantity in owner_rows.items():
                    connection.execute(
                        "INSERT INTO owners(asset,owner_key,quantity,evidence_sha256,updated_at) "
                        "VALUES(?,?,?,?,?)",
                        (asset, owner_key, str(quantity), evidence_sha256, now),
                    )
                total = decimal(balances.get(asset, {}).get("total"))
                free = decimal(balances.get(asset, {}).get("free"))
                locked = decimal(balances.get(asset, {}).get("locked"))
                owned = sum(owner_rows.values(), ZERO)
                deficit = max(owned - total, ZERO)
                unattributed = max(total - owned, ZERO)
                deficit_confirmation = connection.execute(
                    "SELECT * FROM deficit_confirmations WHERE asset=?", (asset,)
                ).fetchone()
                if sources_healthy and deficit > 0:
                    # Fill settlement changes balance and ownership hashes. Keep one
                    # persisted deficit episode until the shortage actually clears.
                    first_deficit_seen = (
                        float(deficit_confirmation["first_seen"])
                        if deficit_confirmation is not None else now
                    )
                    deficit_cycles = (
                        int(deficit_confirmation["cycles"]) + 1
                        if deficit_confirmation is not None else 1
                    )
                    peak_deficit = max(
                        deficit,
                        decimal(deficit_confirmation["peak_deficit"])
                        if deficit_confirmation is not None else ZERO,
                    )
                    deficit_notified = bool(
                        int(deficit_confirmation["notified"])
                        if deficit_confirmation is not None else 0
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO deficit_confirmations"
                        "(asset,first_seen,last_seen,cycles,peak_deficit,notified) "
                        "VALUES(?,?,?,?,?,?)",
                        (asset, first_deficit_seen, now, deficit_cycles,
                         str(peak_deficit), int(deficit_notified)),
                    )
                else:
                    first_deficit_seen, deficit_cycles = now, 0
                    peak_deficit, deficit_notified = ZERO, False
                    connection.execute(
                        "DELETE FROM deficit_confirmations WHERE asset=?", (asset,)
                    )
                deficit_confirmed = bool(
                    sources_healthy and deficit > 0
                    and deficit_cycles >= confirmation_cycles
                    and now - first_deficit_seen >= confirmation_seconds
                )
                stability_sha256 = self._stability_sha256(
                    account_fingerprint=account_fingerprint,
                    asset=asset, total=total, owners=owner_rows,
                    unattributed=unattributed,
                )
                episode = connection.execute(
                    "SELECT * FROM inventory_episodes WHERE asset=?", (asset,)
                ).fetchone()
                if unattributed <= 0:
                    connection.execute("DELETE FROM confirmations WHERE asset=?", (asset,))
                    if episode is not None and episode["phase"] != "RECOVERED":
                        connection.execute(
                            "UPDATE inventory_episodes SET phase='RECOVERED',quantity='0',"
                            "stability_sha256=?,updated_at=? WHERE asset=?",
                            (stability_sha256, now, asset),
                        )
                        episode = connection.execute(
                            "SELECT * FROM inventory_episodes WHERE asset=?", (asset,)
                        ).fetchone()
                elif sources_healthy and deficit == 0:
                    if episode is None or episode["phase"] == "RECOVERED":
                        # Historical dust jobs are migration evidence only for
                        # ledgers that do not yet have an episode.  A balance
                        # that recovered to zero and later reappears starts a
                        # new confirmation episode, even if its quantity is
                        # below a previous dust job.
                        inherited_dust = bool(
                            episode is None
                            and self._dust_job_covers(connection, asset, unattributed)
                        )
                        phase = "DUST" if inherited_dust else "DETECTED"
                        created_at = now
                        episode_id = self._episode_id(
                            account_fingerprint=account_fingerprint, asset=asset,
                            stability_sha256=stability_sha256, created_at=created_at,
                        )
                        notified = "INVENTORY_DUST_CLASSIFIED" if phase == "DUST" else ""
                        connection.execute(
                            "INSERT OR REPLACE INTO inventory_episodes"
                            "(asset,episode_id,phase,quantity,stability_sha256,created_at,"
                            "updated_at,last_notified_transition) VALUES(?,?,?,?,?,?,?,?)",
                            (asset, episode_id, phase, str(unattributed), stability_sha256,
                             created_at, now, notified),
                        )
                    else:
                        phase = str(episode["phase"])
                        if (
                            phase == "CONFIRMED"
                            and episode["stability_sha256"] != stability_sha256
                        ):
                            phase = "DETECTED"
                        connection.execute(
                            "UPDATE inventory_episodes SET phase=?,quantity=?,"
                            "stability_sha256=?,updated_at=? WHERE asset=?",
                            (phase, str(unattributed), stability_sha256, now, asset),
                        )
                    episode = connection.execute(
                        "SELECT * FROM inventory_episodes WHERE asset=?", (asset,)
                    ).fetchone()

                phase = str(episode["phase"]) if episode is not None else "CLEAR"
                confirmation_eligible = bool(
                    sources_healthy and deficit == 0 and unattributed > 0
                    and phase not in {"DUST", "LIQUIDATING"}
                )
                confirmation = connection.execute(
                    "SELECT * FROM confirmations WHERE asset=?", (asset,)
                ).fetchone()
                if confirmation_eligible:
                    stable = bool(
                        confirmation is not None
                        and decimal(confirmation["quantity"]) == unattributed
                        and confirmation["evidence_sha256"] == stability_sha256
                    )
                    first_seen = float(confirmation["first_seen"]) if stable else now
                    cycles = int(confirmation["cycles"]) + 1 if stable else 1
                    connection.execute(
                        "INSERT OR REPLACE INTO confirmations"
                        "(asset,quantity,evidence_sha256,first_seen,last_seen,cycles) "
                        "VALUES(?,?,?,?,?,?)",
                        (asset, str(unattributed), stability_sha256, first_seen, now, cycles),
                    )
                else:
                    first_seen, cycles = now, 0
                    connection.execute("DELETE FROM confirmations WHERE asset=?", (asset,))
                confirmed = bool(
                    confirmation_eligible and cycles >= confirmation_cycles
                    and now - first_seen >= confirmation_seconds
                )
                if confirmed and episode is not None and phase == "DETECTED":
                    phase = "CONFIRMED"
                    connection.execute(
                        "UPDATE inventory_episodes SET phase=?,updated_at=? WHERE asset=?",
                        (phase, now, asset),
                    )
                if not sources_healthy:
                    confirmation_block_reason = "sources_unhealthy"
                elif deficit > 0:
                    confirmation_block_reason = "ownership_deficit"
                elif unattributed <= 0:
                    confirmation_block_reason = "no_unattributed_inventory"
                elif phase == "DUST":
                    confirmation_block_reason = "already_classified_dust"
                elif phase == "LIQUIDATING":
                    confirmation_block_reason = "liquidation_in_progress"
                else:
                    confirmation_block_reason = ""
                assets[asset] = {
                    "exchange": {"free": str(free), "locked": str(locked), "total": str(total)},
                    "owners": {key: str(value) for key, value in owner_rows.items()},
                    "owned_total": str(owned),
                    "unattributed": str(unattributed),
                    "ownership_deficit": str(deficit),
                    "stability_sha256": stability_sha256,
                    "episode_id": str(episode["episode_id"]) if episode is not None else "",
                    "inventory_phase": phase,
                    "last_notified_transition": (
                        str(episode["last_notified_transition"]) if episode is not None else ""
                    ),
                    "confirmation_eligible": confirmation_eligible,
                    "confirmation_block_reason": confirmation_block_reason,
                    "confirmation": {
                        "cycles": cycles, "first_seen": first_seen,
                        "last_seen": now, "confirmed": confirmed,
                    },
                    "deficit_confirmation": {
                        "cycles": deficit_cycles,
                        "first_seen": first_deficit_seen,
                        "last_seen": now,
                        "confirmed": deficit_confirmed,
                        "peak_deficit": str(peak_deficit),
                        "notified": deficit_notified,
                    },
                }
            leases = {
                row["asset"]: {
                    "holder": row["holder"], "acquired_at": row["acquired_at"],
                    "expires_at": row["expires_at"],
                }
                for row in connection.execute(
                    "SELECT * FROM leases WHERE expires_at>?", (now,)
                )
            }
            jobs = [dict(row) for row in connection.execute(
                "SELECT * FROM liquidation_jobs ORDER BY created_at DESC LIMIT 20"
            )]
        payload = {
            "schema": SCHEMA, "generated_at": now,
            "account_fingerprint": account_fingerprint,
            "sources_healthy": bool(sources_healthy),
            "open_order_counts": dict(open_order_counts),
            "active_order_count": total_orders,
            "evidence_sha256": evidence_sha256,
            "assets": assets, "leases": leases, "liquidation_jobs": jobs,
            "healthy": bool(
                sources_healthy and total_orders == 0
                and all(decimal(row["ownership_deficit"]) == 0 for row in assets.values())
            ),
        }
        _atomic_json(self.status_path, payload)
        return payload

    def mark_deficit_notified(self, asset: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "UPDATE deficit_confirmations SET notified=1,last_seen=? WHERE asset=?",
                (now, asset),
            )

    def acquire_lease(
        self, asset: str, holder: str, *, ttl_seconds: int = 30,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM leases WHERE asset=?", (asset,)).fetchone()
            if row is not None and float(row["expires_at"]) > now and row["holder"] != holder:
                return False
            connection.execute(
                "INSERT OR REPLACE INTO leases(asset,holder,acquired_at,expires_at) "
                "VALUES(?,?,?,?)", (asset, holder, now, now + ttl_seconds),
            )
        return True

    def assert_exit_allowed(
        self, *, asset: str, exchange_total: Any,
        owner_key: str = "", requested_quantity: Any = "0",
        tolerance: Any = "0",
    ) -> dict[str, Any]:
        """Recheck shared ownership under the caller's asset lease before selling."""
        exchange_total = decimal(exchange_total)
        requested_quantity = decimal(requested_quantity)
        tolerance = decimal(tolerance)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT owner_key,quantity,evidence_sha256,updated_at "
                "FROM owners WHERE asset=?", (asset,),
            ).fetchall()
        if not rows:
            raise RuntimeError(f"shared ownership evidence is missing for {asset}")
        owners = {str(row["owner_key"]): decimal(row["quantity"]) for row in rows}
        owned_total = sum(owners.values(), ZERO)
        if owned_total > exchange_total + tolerance:
            raise RuntimeError(
                f"ownership_deficit for {asset}: owned={owned_total};exchange={exchange_total}"
            )
        if owner_key:
            if owner_key not in owners:
                raise RuntimeError(f"shared ownership has no {owner_key} for {asset}")
            if requested_quantity > owners[owner_key] + tolerance:
                raise RuntimeError(
                    f"exit exceeds {owner_key} ownership for {asset}: "
                    f"requested={requested_quantity};owned={owners[owner_key]}"
                )
        return {
            "asset": asset, "exchange_total": str(exchange_total),
            "owned_total": str(owned_total),
            "owners": {key: str(value) for key, value in owners.items()},
        }

    def release_lease(self, asset: str, holder: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM leases WHERE asset=? AND holder=?", (asset, holder)
            )

    def renew_lease(
        self, asset: str, holder: str, *, ttl_seconds: int = 30,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT holder,expires_at FROM leases WHERE asset=?", (asset,)
            ).fetchone()
            if row is None or row["holder"] != holder or float(row["expires_at"]) <= now:
                return False
            connection.execute(
                "UPDATE leases SET expires_at=? WHERE asset=? AND holder=?",
                (now + ttl_seconds, asset, holder),
            )
        return True

    @contextmanager
    def lease(self, asset: str, holder: str, *, ttl_seconds: int = 30) -> Iterator[None]:
        if not self.acquire_lease(asset, holder, ttl_seconds=ttl_seconds):
            raise RuntimeError(f"inventory lease for {asset} is held by another guard")
        try:
            yield
        finally:
            self.release_lease(asset, holder)

    def start_job(
        self, *, job_id: str, asset: str, scope: str, pair: str,
        requested_quantity: Decimal, client_order_id: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO liquidation_jobs"
                "(job_id,asset,scope,pair,requested_quantity,client_order_id,status,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,'STARTED',?,?)",
                (job_id, asset, scope, pair, str(requested_quantity), client_order_id, now, now),
            )
            row = connection.execute(
                "SELECT * FROM liquidation_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row)

    def finish_job(
        self, job_id: str, *, status: str, exchange_order_id: str = "",
        executed_quantity: Any = "0", quote_quantity: Any = "0",
        fee_quote: Any = "0", fee_details: Any = None,
        error: str = "", verification: Mapping[str, Any] | None = None,
        consume_bootstrap_asset: str = "",
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        if status == "COMPLETED":
            required = {
                "order_verified", "balance_verified", "no_active_orders",
                "requested_quantity_verified",
            }
            if verification is None or not required.issubset(verification):
                raise ValueError("completed liquidation job requires four-part verification")
            if not all(bool(verification[key]) for key in required):
                raise ValueError("completed liquidation job verification did not pass")
        with self._connection() as connection:
            connection.execute(
                "UPDATE liquidation_jobs SET status=?,exchange_order_id=?,"
                "executed_quantity=?,quote_quantity=?,fee_quote=?,fee_details=?,error=?,"
                "verification_json=?,verified_at=?,updated_at=? "
                "WHERE job_id=?",
                (status, exchange_order_id, str(executed_quantity), str(quote_quantity),
                 str(fee_quote), json.dumps(fee_details or [], sort_keys=True),
                 error, json.dumps(dict(verification or {}), sort_keys=True),
                 now if status == "COMPLETED" else None, now, job_id),
            )
            if consume_bootstrap_asset:
                connection.execute(
                    "UPDATE bootstrap_caps SET consumed=1,updated_at=? WHERE asset=?",
                    (now, consume_bootstrap_asset),
                )

    def start_attempt(
        self, *, job_id: str, sequence: int, client_order_id: str,
        requested_quantity: Any, now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO liquidation_attempts"
                "(job_id,sequence,client_order_id,requested_quantity,status,created_at,updated_at) "
                "VALUES(?,?,?,?,'PLANNED',?,?)",
                (job_id, sequence, client_order_id, str(requested_quantity), now, now),
            )
            row = connection.execute(
                "SELECT * FROM liquidation_attempts WHERE job_id=? AND sequence=?",
                (job_id, sequence),
            ).fetchone()
        return dict(row)

    def finish_attempt(
        self, *, job_id: str, sequence: int, status: str,
        response: Mapping[str, Any] | None = None, error: str = "",
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        response = dict(response or {})
        with self._connection() as connection:
            connection.execute(
                "UPDATE liquidation_attempts SET status=?,exchange_order_id=?,"
                "executed_quantity=?,quote_quantity=?,response_json=?,error=?,updated_at=? "
                "WHERE job_id=? AND sequence=?",
                (status, str(response.get("orderId", "")),
                 str(response.get("executedQty", "0")),
                 str(response.get("cummulativeQuoteQty", "0")),
                 json.dumps(response, sort_keys=True, default=str), error, now,
                 job_id, sequence),
            )

    def attempts(self, job_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM liquidation_attempts WHERE job_id=? ORDER BY sequence",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def completed_job_verified(job: Mapping[str, Any]) -> bool:
        if str(job.get("status")) != "COMPLETED":
            return False
        try:
            verification = json.loads(str(job.get("verification_json") or "{}"))
        except json.JSONDecodeError:
            return False
        required = {
            "order_verified", "balance_verified", "no_active_orders",
            "requested_quantity_verified",
        }
        return required.issubset(verification) and all(
            bool(verification[key]) for key in required
        )

    def set_fee_details(self, job_id: str, fee_details: Any, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "UPDATE liquidation_jobs SET fee_details=?,updated_at=? WHERE job_id=?",
                (json.dumps(fee_details or [], sort_keys=True), now, job_id),
            )

    def record_event_once(
        self, event_id: str, kind: str, payload: Mapping[str, Any],
        *, now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO events"
                "(event_id,kind,payload,delivered,delivered_at,created_at) VALUES(?,?,?,1,?,?)",
                (event_id, kind, json.dumps(dict(payload), sort_keys=True, default=str), now, now),
            )
        return cursor.rowcount == 1

    def stage_event(
        self, event_id: str, kind: str, payload: Mapping[str, Any],
        *, now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO events"
                "(event_id,kind,payload,delivered,created_at) VALUES(?,?,?,0,?)",
                (event_id, kind, json.dumps(dict(payload), sort_keys=True, default=str), now),
            )
            row = connection.execute(
                "SELECT delivered FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        return bool(row is not None and not int(row["delivered"]))

    def mark_event_delivered(self, event_id: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute(
                "UPDATE events SET delivered=1,delivered_at=? WHERE event_id=?",
                (now, event_id),
            )

    def pending_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT event_id,payload FROM events WHERE delivered=0 "
                "ORDER BY created_at,event_id LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            result.append({"event_id": row["event_id"], "payload": payload})
        return result


def liquidation_identity(asset: str, scope: str, quantity: Decimal, evidence_sha256: str) -> tuple[str, str]:
    digest = canonical_sha256({
        "asset": asset, "scope": scope, "quantity": str(quantity),
        "evidence_sha256": evidence_sha256,
    })
    return digest, f"inv-{digest[:24]}"
