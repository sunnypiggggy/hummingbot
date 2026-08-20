"""Produce the sole ethbtc-forced-exit v22 contract inside Grid Live Guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from build_xgboost_v22_shadow_signal import produce_once
    from ethbtc_forced_exit_contract import (
        EXECUTION_POLICY_VERSION, MODEL_VERSION, PACKAGE_ID, REQUIRED_PAIRS,
        STALE_AFTER_SECONDS, atomic_json, event_id, failed_contract, sha256_file, utc,
    )
except ModuleNotFoundError:  # Package import from the repository root/tests.
    from scripts.build_xgboost_v22_shadow_signal import produce_once
    from scripts.ethbtc_forced_exit_contract import (
        EXECUTION_POLICY_VERSION, MODEL_VERSION, PACKAGE_ID, REQUIRED_PAIRS,
        STALE_AFTER_SECONDS, atomic_json, event_id, failed_contract, sha256_file, utc,
    )


CONFIRMATION = "PROMOTE-ETHBTC-FORCED-EXIT"
AUTO_CONFIRMATION = "AUTO-PROMOTE-ETHBTC-FORCED-EXIT-AFTER-12H"
RUNTIME_POINTER_SCHEMA = "ethbtc-forced-exit-runtime-pointer-v1"
RUNTIME_GENERATION_SCHEMA = "ethbtc-forced-exit-runtime-generation-v1"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _resolve_package(package_dir: Path) -> tuple[Path, Path]:
    if (package_dir / "current").exists() and (package_dir / "releases").is_dir():
        package_dir = package_dir / "current"
    lock = package_dir / "shadow_package/shadow_lock.json"
    production = package_dir / "production_lock.json"
    if not lock.exists():
        lock = package_dir / "shadow_lock.json"
    if not lock.is_file() or not production.is_file():
        raise FileNotFoundError("v22 candidate lacks shadow_package or production_lock.json")
    return lock, production


def _authorization(source: Path | dict[str, Any], production: dict[str, Any],
                   observed: int) -> tuple[bool, dict[str, Any], str | None]:
    if isinstance(source, Path):
        if not source.is_file():
            return False, {}, None
        raw = source.read_bytes()
        receipt = json.loads(raw.decode("utf-8"))
        receipt_hash = sha256_file(source)
    else:
        receipt = dict(source)
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        receipt_hash = hashlib.sha256(raw).hexdigest()
    expected = {
        "schema": "ethbtc-forced-exit-authorization-v1",
        "package_id": PACKAGE_ID,
        "release_sha256": production["release_sha256"],
        "model_sha256": production["model_sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("v22 authorization receipt does not match candidate")
    confirmation = receipt.get("confirmation")
    if confirmation not in {CONFIRMATION, AUTO_CONFIRMATION}:
        raise ValueError("v22 authorization confirmation is invalid")
    if confirmation == AUTO_CONFIRMATION:
        if receipt.get("approval_mode") != "automatic_default_after_12h":
            raise ValueError("automatic v22 authorization mode is invalid")
        started = int(receipt.get("review_started_at", 0))
        deadline = int(receipt.get("review_deadline", 0))
        approved = int(receipt.get("approved_at", 0))
        if deadline - started < 12 * 60 * 60 or approved < deadline:
            raise ValueError("automatic v22 authorization did not complete the 12h review")
        if not receipt.get("approval_request_sha256"):
            raise ValueError("automatic v22 authorization lacks review request hash")
    activation = int(receipt["activate_at"])
    if activation % 60 or activation < int(receipt["approved_at"]):
        raise ValueError("v22 authorization activation boundary is invalid")
    if activation >= int(production["effective_end"]):
        raise ValueError("v22 authorization starts after signed coverage")
    if not receipt.get("observation_report_sha256") or not receipt.get("preflight_sha256"):
        raise ValueError("v22 authorization lacks observation/preflight evidence")
    return observed >= activation, receipt, receipt_hash


def _active_deployment(package_dir: Path, authorization_path: Path) -> tuple[Path, Path | dict[str, Any]]:
    pointer_path = package_dir / "active_deployment.json"
    if not pointer_path.is_file():
        return package_dir, authorization_path
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if (pointer.get("schema") != "ethbtc-forced-exit-active-deployment-v1"
            or pointer.get("package_id") != PACKAGE_ID):
        raise ValueError("invalid v22 active deployment pointer")
    release_sha = str(pointer.get("release_sha256", ""))
    receipt = pointer.get("authorization")
    if not isinstance(receipt, dict):
        raise ValueError("v22 active deployment has no embedded authorization")
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != pointer.get("authorization_sha256"):
        raise ValueError("v22 active deployment authorization hash mismatch")
    release = package_dir / "releases" / release_sha
    if not release.is_dir():
        raise FileNotFoundError("v22 active deployment release is missing")
    return release, receipt


def _runtime_deployment(
    package_dir: Path, runtime_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Resolve one committed generation without consulting mutable aliases."""
    pointer_path = runtime_root / "current.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != RUNTIME_POINTER_SCHEMA:
        raise ValueError("invalid v22 runtime pointer schema")
    generation = str(pointer.get("runtime_generation", ""))
    if not generation or generation != pointer.get("generation_manifest_sha256"):
        raise ValueError("invalid v22 runtime generation id")
    manifest_path = runtime_root / "generations" / generation / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != generation:
        raise ValueError("v22 runtime generation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RUNTIME_GENERATION_SCHEMA:
        raise ValueError("invalid v22 runtime generation manifest")
    release_sha = str(manifest.get("release_sha256", ""))
    release = package_dir / "releases" / release_sha
    if not release.is_dir():
        raise FileNotFoundError("v22 runtime generation release is missing")
    receipt = manifest.get("authorization")
    if not isinstance(receipt, dict):
        raise ValueError("v22 runtime generation authorization is missing")
    if canonical_sha256(receipt) != manifest.get("authorization_sha256"):
        raise ValueError("v22 runtime generation authorization hash mismatch")
    final_preflight = manifest.get("final_preflight_sha256")
    if final_preflight is not None and not valid_sha256(final_preflight):
        raise ValueError("v22 runtime generation final preflight hash mismatch")
    return release, receipt, {
        **manifest,
        "runtime_generation": generation,
        "cutover_phase": pointer.get("cutover_phase", manifest.get("cutover_phase")),
    }


class V22LiveGateProducer:
    def __init__(self, *, package_dir: Path, cache_dir: Path, seed_cache_dir: Path,
                 state_dir: Path, authorization_path: Path, refresh_binance: bool = True,
                 runtime_root: Path | None = None):
        self.package_dir = package_dir
        self.cache_dir = cache_dir
        self.seed_cache_dir = seed_cache_dir
        self.state_dir = state_dir
        self.authorization_path = authorization_path
        self.refresh_binance = refresh_binance
        self.runtime_root = runtime_root or state_dir / "v22-runtime"
        self.output = state_dir / "xgboost_risk_gate.json"
        self.shadow_output = state_dir / "xgboost_risk_gate_v22_internal.json"
        self.shadow_state = state_dir / "xgboost_risk_gate_v22_state.json"

    def _deployment(self) -> tuple[Path, Path | dict[str, Any], dict[str, Any]]:
        committed = _runtime_deployment(self.package_dir, self.runtime_root)
        if committed is not None:
            release, receipt, context = committed
            return release, receipt, context
        release, receipt = _active_deployment(self.package_dir, self.authorization_path)
        return release, receipt, {
            "runtime_generation": "legacy",
            "predecessor_release_sha256": None,
            "fold_boundary": None,
            "cutover_phase": "LEGACY_ACTIVE",
        }

    def _generation_paths(self, context: Mapping[str, Any]) -> tuple[Path, Path]:
        generation = str(context.get("runtime_generation") or "legacy")
        if generation == "legacy":
            return self.shadow_output, self.shadow_state
        root = self.runtime_root / "generations" / generation
        return root / "shadow_contract.json", root / "gate_state.json"

    @staticmethod
    def _semantic_pairs(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            pair: {
                "signal_ts": contract.get("pairs", {}).get(pair, {}).get("signal_ts"),
                "risk_off_active": contract.get("pairs", {}).get(pair, {}).get("risk_off_active"),
                "buy_enabled": contract.get("pairs", {}).get(pair, {}).get("buy_enabled"),
                "force_exit": contract.get("pairs", {}).get(pair, {}).get("force_exit"),
                "model_week": contract.get("pairs", {}).get(pair, {}).get("model_week"),
                "week_model_sha256": contract.get("pairs", {}).get(pair, {}).get("week_model_sha256"),
            }
            for pair in REQUIRED_PAIRS
        }

    def prepare_generation(
        self, *, release: Path, authorization: Path | dict[str, Any],
        predecessor_release_sha256: str, fold_boundary: int,
        observed_at: int, live_contract_path: Path | None = None,
        final_preflight_sha256: str | None = None,
        recovery_from_unavailable: bool = False,
    ) -> dict[str, Any]:
        """Build and verify an isolated generation; never publish it live."""
        lock_path, production_path = _resolve_package(release)
        production = json.loads(production_path.read_text(encoding="utf-8"))
        if isinstance(authorization, Path):
            receipt = json.loads(authorization.read_text(encoding="utf-8"))
        else:
            receipt = dict(authorization)
        if final_preflight_sha256 is not None and not valid_sha256(final_preflight_sha256):
            raise ValueError("candidate final preflight hash is invalid")
        identity = {
            "release_sha256": production["release_sha256"],
            "authorization_sha256": canonical_sha256(receipt),
            "predecessor_release_sha256": predecessor_release_sha256,
            "fold_boundary": int(fold_boundary),
            "final_preflight_sha256": final_preflight_sha256,
        }
        staging_id = canonical_sha256(identity)
        staging = self.runtime_root / "staging" / staging_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        staged_state = staging / "gate_state.json"
        committed = _runtime_deployment(self.package_dir, self.runtime_root)
        source_state = self.shadow_state
        if committed is not None:
            source_context = committed[2]
            source_state = (
                self.runtime_root / "generations"
                / str(source_context["runtime_generation"]) / "gate_state.json"
            )
        if source_state.is_file():
            shutil.copy2(source_state, staged_state)
        staged_shadow = staging / "shadow_contract.json"
        context = {
            **identity,
            "runtime_generation": staging_id,
            "cutover_phase": "PREWARMED",
        }
        contract = self._produce_for(
            observed=int(observed_at), active_package=release,
            authorization=receipt, context=context,
            shadow_output=staged_shadow, shadow_state=staged_state,
            prewarm_authorized=True,
        )
        if contract.get("source_healthy") is not True:
            raise RuntimeError(f"candidate prewarm unhealthy: {contract.get('reason')}")
        if not live_contract_path or not live_contract_path.is_file():
            raise RuntimeError("candidate prewarm requires the current live contract")
        live = json.loads(live_contract_path.read_text(encoding="utf-8"))
        if live.get("release_sha256") != predecessor_release_sha256:
            raise RuntimeError(
                "candidate predecessor release does not match the current live contract"
            )
        if recovery_from_unavailable:
            reason = str(live.get("reason") or "")
            unavailable = (
                live.get("source_healthy") is False
                and (
                    "no signed weekly model covers" in reason
                    or "signed_week_unavailable" in reason
                    or "contract is stale" in reason
                )
            )
            if not unavailable:
                raise RuntimeError(
                    "late recovery requires an unavailable signed-week live contract"
                )
            parity = {
                "checked": True,
                "matched": True,
                "recovery_from_unavailable": True,
                "unavailable_reason": reason,
                "differences": {},
            }
        else:
            if live.get("source_healthy") is not True:
                raise RuntimeError("candidate prewarm requires a healthy current live contract")
            wanted = self._semantic_pairs(live)
            actual = self._semantic_pairs(contract)
            differences = {
                pair: {"active": wanted[pair], "candidate": actual[pair]}
                for pair in REQUIRED_PAIRS if wanted[pair] != actual[pair]
            }
            parity = {"checked": True, "matched": not differences, "differences": differences}
            if differences:
                raise RuntimeError(f"candidate semantic parity failed: {differences}")
        manifest = {
            "schema": RUNTIME_GENERATION_SCHEMA,
            **identity,
            "prepared_at": int(observed_at),
            "cutover_phase": "PREWARMED",
            "authorization": receipt,
            "shadow_lock_sha256": sha256_file(lock_path),
            "prepared_state_sha256": sha256_file(staged_state),
            "prepared_contract_sha256": sha256_file(staged_shadow),
            "semantic_parity": parity,
        }
        atomic_json(staging / "manifest.json", manifest)
        generation = sha256_file(staging / "manifest.json")
        target = self.runtime_root / "generations" / generation
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, target)
        pointer = {
            "schema": RUNTIME_POINTER_SCHEMA,
            "runtime_generation": generation,
            "generation_manifest_sha256": generation,
            "release_sha256": production["release_sha256"],
            "prepared_at": int(observed_at),
            "fold_boundary": int(fold_boundary),
        }
        return {"pointer": pointer, "manifest": manifest, "generation": generation}

    def commit_generation(self, pointer: Mapping[str, Any]) -> None:
        generation = str(pointer.get("runtime_generation", ""))
        manifest = self.runtime_root / "generations" / generation / "manifest.json"
        if not manifest.is_file() or sha256_file(manifest) != generation:
            raise ValueError("cannot commit an unverified v22 runtime generation")
        atomic_json(self.runtime_root / "current.json", dict(pointer))

    def produce(self, observed_at: int | None = None) -> dict[str, Any]:
        observed = int(observed_at if observed_at is not None else time.time())
        try:
            active_package, authorization, context = self._deployment()
        except Exception:
            # A torn or unreadable candidate pointer must not hide the still
            # valid deployment aliases. Validation below will independently
            # reject that predecessor once its signed coverage ends.
            active_package, authorization = _active_deployment(
                self.package_dir, self.authorization_path,
            )
            context = {
                "runtime_generation": "legacy",
                "predecessor_release_sha256": None,
                "fold_boundary": None,
                "cutover_phase": "POINTER_READ_FALLBACK",
            }
        shadow_output, shadow_state = self._generation_paths(context)
        contract = self._produce_for(
            observed=observed, active_package=active_package,
            authorization=authorization, context=context,
            shadow_output=shadow_output, shadow_state=shadow_state,
        )
        boundary = context.get("fold_boundary")
        if (
            contract.get("source_healthy") is not True
            and context.get("runtime_generation") != "legacy"
            and boundary is not None and observed < int(boundary)
        ):
            # Before the fold boundary the predecessor still has signed
            # coverage. A candidate runtime failure therefore remains an
            # isolated warm-cutover failure, not a live integrity failure.
            predecessor_package, predecessor_auth = _active_deployment(
                self.package_dir, self.authorization_path,
            )
            failed_reason = contract.get("reason")
            contract = self._produce_for(
                observed=observed, active_package=predecessor_package,
                authorization=predecessor_auth,
                context={
                    "runtime_generation": "legacy",
                    "predecessor_release_sha256": None,
                    "fold_boundary": int(boundary),
                    "cutover_phase": "WARM_ROLLBACK_USING_SIGNED_PREDECESSOR",
                },
                shadow_output=self.shadow_output, shadow_state=self.shadow_state,
            )
            contract["candidate_refresh_failure"] = str(failed_reason)
        atomic_json(self.output, contract)
        return contract

    def _produce_for(
        self, *, observed: int, active_package: Path,
        authorization: Path | dict[str, Any], context: Mapping[str, Any],
        shadow_output: Path, shadow_state: Path,
        prewarm_authorized: bool = False,
    ) -> dict[str, Any]:
        lock_path, production_path = _resolve_package(active_package)
        production = json.loads(production_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        metadata = {
            "release_sha256": production.get("release_sha256"),
            "model_sha256": lock.get("model_sha256"),
            "feature_schema_sha256": lock.get("feature_schema_sha256"),
            "strategy_schema_sha256": lock.get("strategy_schema_sha256"),
            "training_data_sha256": lock.get(
                "training_candle_sha256", lock.get("training_panel_sha256")
            ),
        }
        try:
            if production.get("package_id") != PACKAGE_ID:
                raise ValueError("candidate package id mismatch")
            if production.get("execution_policy_version") != EXECUTION_POLICY_VERSION:
                raise ValueError("candidate execution policy mismatch")
            if production.get("model_sha256") != lock.get("model_sha256"):
                raise ValueError("production/shadow model hash mismatch")
            if sha256_file(lock_path) != production.get("shadow_lock_sha256"):
                raise ValueError("candidate shadow lock hash mismatch")
            shadow = produce_once(SimpleNamespace(
                observed_at=observed, lock=lock_path, cache_dir=self.cache_dir,
                seed_cache_dir=self.seed_cache_dir, output=shadow_output,
                state=shadow_state, refresh_binance=self.refresh_binance,
            ))
            authorized, receipt, receipt_hash = _authorization(
                authorization, production, observed,
            )
            # Prewarm is isolated and cannot publish or execute.  Treat a
            # fully validated future activation receipt as authorized only so
            # effective permission parity can be checked before T-30m.
            if prewarm_authorized:
                authorized = True
            source_healthy = bool(shadow.get("source_healthy"))
            if not source_healthy:
                # Failed shadow contracts intentionally carry null model values.
                # Do not try to coerce those values while constructing the live
                # contract: doing so masks the primary integrity failure with an
                # unhelpful ``float(None)`` error.  Publish the canonical
                # fail-closed shape and preserve the original source reason.
                contract = failed_contract(
                    generated_at=observed,
                    reason=str(shadow.get("reason") or "v22 shadow source unhealthy"),
                    metadata=metadata,
                )
                return self._with_runtime_context(contract, context, shadow_state)
            pairs = {}
            for pair in REQUIRED_PAIRS:
                source = shadow["pairs"][pair]
                long = source["long"]
                week_start = int(long["week_test_start"])
                week_end = int(long["week_test_end"])
                # The shadow producer stores the authoritative latest signal in
                # its persistent state; use it rather than parsing display time.
                state_payload = json.loads(shadow_state.read_text(encoding="utf-8"))
                signal_ts = int(state_payload["pairs"][pair]["gate_state"]["last_signal_ts"])
                risk_off = bool(source["risk_off_active"])
                transition = str(long.get("transition", "hold"))
                pairs[pair] = {
                    "pair": pair, "source_pair": pair, "signal_ts": signal_ts,
                    "model_week": int(long["fold"]), "week_start": week_start,
                    "week_end": week_end, "week_model_sha256": long["week_model_sha256"],
                    "probability": float(long["probability"]),
                    "entry_threshold": float(long["entry_threshold"]),
                    "risk_off_active": risk_off,
                    "model_signal": "RISK_OFF" if risk_off else "RISK_ON",
                    "recommended_buy_enabled": not risk_off,
                    "buy_enabled": bool(authorized and not risk_off),
                    "force_exit": bool(authorized and risk_off),
                    "transition": transition, "reason": str(source.get("reason", long.get("reason", "v22"))),
                    "event_id": event_id(production["release_sha256"], pair, signal_ts, transition),
                }
            valid_until = min(observed + STALE_AFTER_SECONDS, int(production["effective_end"]))
            contract = {
                "schema": "ethbtc-forced-exit-live-contract-v1", "package_id": PACKAGE_ID,
                "execution_policy_version": EXECUTION_POLICY_VERSION,
                "model_version": MODEL_VERSION, "generated_at": utc(observed),
                "valid_until": utc(valid_until), "stale_after_seconds": STALE_AFTER_SECONDS,
                **metadata, "source_healthy": source_healthy,
                "execution_authorized": bool(authorized), "observation_mode": not authorized,
                "activation_at": receipt.get("activate_at"),
                "approval_receipt_sha256": receipt_hash,
                "deployment_allowed": bool(authorized), "promotion_authorized": bool(authorized),
                "market_sell_action": True, "previous_model_fallback_allowed": False,
                "runtime_action": "execute" if authorized else "observe_only",
                "reason": "v22_live_healthy" if authorized else "v22_observation_healthy",
                "pairs": pairs,
            }
        except Exception as exc:
            contract = failed_contract(
                generated_at=observed, reason=f"fail_closed:{exc}", metadata=metadata,
            )
        return self._with_runtime_context(contract, context, shadow_state)

    @staticmethod
    def _with_runtime_context(
        contract: dict[str, Any], context: Mapping[str, Any], state_path: Path,
    ) -> dict[str, Any]:
        healthy = bool(contract.get("source_healthy"))
        for item in contract.get("pairs", {}).values():
            item.setdefault(
                "model_signal",
                "RISK_OFF" if healthy and item.get("risk_off_active") else
                "RISK_ON" if healthy else "UNAVAILABLE",
            )
        contract.update({
            "runtime_generation": context.get("runtime_generation", "legacy"),
            "predecessor_release_sha256": context.get("predecessor_release_sha256"),
            "state_lineage_sha256": (
                sha256_file(state_path) if state_path.is_file() else canonical_sha256({})
            ),
            "cutover_phase": context.get("cutover_phase", "ACTIVE"),
            "fold_boundary": context.get("fold_boundary"),
            "system_health": "HEALTHY" if healthy else "FAILED",
        })
        return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed-cache-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--observed-at", type=int)
    args = parser.parse_args()
    producer = V22LiveGateProducer(
        package_dir=args.package_dir, cache_dir=args.cache_dir,
        seed_cache_dir=args.seed_cache_dir, state_dir=args.state_dir,
        authorization_path=args.authorization,
    )
    print(json.dumps(producer.produce(args.observed_at), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
