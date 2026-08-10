#!/usr/bin/env python3
"""Seeded state-machine scenarios for inventory, gates, faults, and restarts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_guard"))
from account_inventory import UnifiedInventoryLedger, canonical_sha256  # noqa: E402


MECHANISMS = (
    "v22_weekly_buy_gate", "fomc_gate", "strategy_loss_breaker",
    "strategy_drawdown_breaker", "portfolio_loss_breaker",
    "portfolio_drawdown_breaker", "position_protection",
)
ACTIONS = (
    "buy_fill", "sell_fill", "external_deposit", "source_fault",
    "open_order", "ownership_deficit", "restart", "signal_toggle",
)
QUANTA = {"BTC": Decimal("0.00001"), "ETH": Decimal("0.0001")}


def run_seed(seed: int, steps: int, root: Path) -> tuple[int, dict[str, int]]:
    rng = random.Random(seed)
    directory = root / f"seed-{seed}"
    ledger = UnifiedInventoryLedger(directory)
    balances = {"BTC": Decimal("0.003"), "ETH": Decimal("0.08")}
    owners = {
        "BTC": {"grid": Decimal("0.0007"), "dca": Decimal("0.0015")},
        "ETH": {"grid": Decimal("0.015"), "dca": Decimal("0.05")},
    }
    gates = {mechanism: True for mechanism in MECHANISMS}
    action_counts = {action: 0 for action in ACTIONS}
    assertions = 0

    for index in range(steps):
        action = rng.choice(ACTIONS)
        action_counts[action] += 1
        asset = rng.choice(("BTC", "ETH"))
        owner = rng.choice(("grid", "dca"))
        quantum = QUANTA[asset]
        sources_healthy, open_count = True, 0
        reconcile_owners = {
            name: dict(values) for name, values in owners.items()
        }

        if action == "buy_fill":
            quantity = quantum * rng.randint(1, 20)
            balances[asset] += quantity
            owners[asset][owner] += quantity
            reconcile_owners[asset][owner] += quantity
        elif action == "sell_fill":
            available = owners[asset][owner]
            quantity = min(available, quantum * rng.randint(1, 20))
            balances[asset] -= quantity
            owners[asset][owner] -= quantity
            reconcile_owners[asset][owner] -= quantity
        elif action == "external_deposit":
            balances[asset] += quantum * rng.randint(1, 20)
        elif action == "source_fault":
            sources_healthy = False
        elif action == "open_order":
            open_count = rng.randint(1, 3)
        elif action == "ownership_deficit":
            reconcile_owners[asset][owner] += balances[asset] + quantum
        elif action == "restart":
            ledger = UnifiedInventoryLedger(directory)
        elif action == "signal_toggle":
            mechanism = rng.choice(MECHANISMS)
            gates[mechanism] = not gates[mechanism]

        # The persistent canonical state follows confirmed exchange fills. A
        # one-cycle ownership_deficit is deliberately not committed to it.
        if action != "ownership_deficit":
            reconcile_owners = {
                name: dict(values) for name, values in owners.items()
            }
        evidence = canonical_sha256({
            "balances": balances, "owners": reconcile_owners,
            "sequence": index // 3, "seed": seed,
        })
        status = ledger.reconcile(
            account_fingerprint=f"scenario-random-{seed}",
            balances={
                name: {"free": value, "locked": 0, "total": value}
                for name, value in balances.items()
            },
            ownership=reconcile_owners, evidence_sha256=evidence,
            open_order_counts={"BTC-USDT": open_count},
            sources_healthy=sources_healthy, now=1000 + index,
        )
        has_deficit = any(
            Decimal(row["ownership_deficit"]) > 0
            for row in status["assets"].values()
        )
        expected_health = sources_healthy and open_count == 0 and not has_deficit
        if status["healthy"] is not expected_health:
            raise AssertionError({"seed": seed, "step": index, "status": status})
        assertions += 1

        for row in status["assets"].values():
            unattributed = Decimal(row["unattributed"])
            deficit = Decimal(row["ownership_deficit"])
            if unattributed < 0 or deficit < 0 or (unattributed > 0 and deficit > 0):
                raise AssertionError({"seed": seed, "step": index, "row": row})
            if not expected_health and row["confirmation"]["confirmed"]:
                raise AssertionError({"seed": seed, "step": index, "row": row})
            assertions += 2

        effective_buy = all(gates.values())
        if effective_buy != all(bool(gates[name]) for name in MECHANISMS):
            raise AssertionError({"seed": seed, "step": index, "gates": gates})
        if any(not value for value in gates.values()) and effective_buy:
            raise AssertionError({"seed": seed, "step": index, "gates": gates})
        assertions += 2

    return assertions, action_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="inventory-random-scenarios-"))
    try:
        results = {
            seed: run_seed(seed, args.steps, directory) for seed in range(args.seeds)
        }
        counts = {seed: result[0] for seed, result in results.items()}
        action_counts = {action: 0 for action in ACTIONS}
        for _assertions, seed_actions in results.values():
            for action, count in seed_actions.items():
                action_counts[action] += count
        if any(count == 0 for count in action_counts.values()):
            raise AssertionError(f"random scenario did not cover every action: {action_counts}")
        report = {
            "schema": "inventory-random-scenario-report-v2",
            "verdict": "PASS", "seeds": args.seeds,
            "steps_per_seed": args.steps,
            "assertions": sum(counts.values()), "seed_assertions": counts,
            "action_counts": action_counts,
            "mechanisms": list(MECHANISMS),
        }
        canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        report["content_sha256"] = hashlib.sha256(canonical).hexdigest()
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return 0
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    raise SystemExit(main())
