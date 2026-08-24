from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA = "management-parameter-catalog-v1"
EVIDENCE_SCHEMA = "management-model-evidence-catalog-v2"
LIFECYCLE_SCHEMA = "management-model-lifecycle-v1"
HISTORY_SCHEMA = "management-parameter-history-v2"
HISTORY_LIMIT = 3

GRID_FIELDS = (
    "capital_limit_quote", "strategy_budget_quote", "reserve_quote",
    "pair_budget_quote", "side_budget_quote", "grid_range", "grid_levels",
    "take_profit", "move_threshold", "min_grid_move_seconds",
    "order_refresh_time", "min_order_quote", "fee_rate",
    "active_parameter_version", "parameter_poll_seconds",
)
DCA_FIELDS = (
    "id", "trading_pair", "total_amount_quote", "buy_spreads", "sell_spreads",
    "buy_amounts_pct", "sell_amounts_pct", "dca_spreads", "dca_amounts",
    "executor_refresh_time", "cooldown_time", "stop_loss", "take_profit",
    "time_limit", "time_limit_from_first_fill", "stop_loss_on_partial_fills",
    "take_profit_order_type", "sell_trend_gate_enabled", "sell_trend_interval",
    "sell_trend_fast_ema", "sell_trend_slow_ema", "sell_trend_roc_bars",
    "sell_trend_trigger_roc", "sell_trend_trigger_ema_gap",
    "sell_trend_recovery_roc", "sell_trend_recovery_bars",
    "sell_stop_cooldown_seconds", "long_only_enabled", "policy_version",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return {}


def _public(value: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: value.get(key) for key in fields if key in value}


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def _different(configured: Mapping[str, Any], runtime: Mapping[str, Any]) -> bool:
    comparable = set(configured) & set(runtime)
    return any(str(configured[key]) != str(runtime[key]) for key in comparable)


class ManagementParameterPublisher:
    """Publish a secret-free, content-addressed view for the management Bot."""

    def __init__(self, *, bots_path: Path, dca_state: Path, grid_state: Path,
                 release_root: Path, approval_root: Path, output: Path) -> None:
        self.bots_path = bots_path
        self.dca_state = dca_state
        self.grid_state = grid_state
        self.release_root = release_root
        self.approval_root = approval_root
        self.output = output / "management"
        self.history_root = self.output / "history"
        self.evidence_root = self.output / "evidence"

    def _grid(self) -> dict[str, Any]:
        bot = "grid-live-fdusd-400"
        config_path = _latest(list(self.bots_path.glob(
            f"instances/{bot}*/conf/scripts/walk_forward_portfolio_grid_live_fdusd_400.yml"
        )))
        runtime_path = _latest(list(self.bots_path.glob(
            f"instances/{bot}*/data/live_grid_runtime_state.json"
        )))
        configured = _public(_yaml(config_path), GRID_FIELDS) if config_path else {}
        runtime_raw = _json(runtime_path) if runtime_path else {}
        runtime_shared = _public(runtime_raw.get("active_parameters", {}), GRID_FIELDS)
        for key in ("active_parameter_version", "active_parameter_sha256"):
            if runtime_raw.get(key) is not None:
                runtime_shared[key] = runtime_raw.get(key)
        pairs = {}
        for pair in ("BTC-FDUSD", "ETH-FDUSD"):
            pair_value = dict(runtime_raw.get("active_pair_parameters", {}).get(pair, {}))
            pairs[pair] = {
                "effective": pair_value or runtime_shared or configured,
                "runtime_available": bool(runtime_raw),
            }
        runtime_override = bool(
            runtime_raw.get("active_parameter_sha256")
            and all(pair in runtime_raw.get("active_pair_parameters", {})
                    for pair in ("BTC-FDUSD", "ETH-FDUSD"))
        )
        mismatch = bool(
            runtime_shared and not runtime_override
            and _different(configured, runtime_shared)
        )
        return {
            "bot": bot, "configured": configured, "runtime": runtime_shared,
            "pairs": pairs, "configured_sha256": _canonical_sha(configured),
            "runtime_sha256": _canonical_sha(runtime_shared) if runtime_shared else None,
            "application_state": "MISMATCH" if mismatch else "APPLIED" if runtime_raw else "UNAVAILABLE",
            "runtime_override_active": runtime_override,
            "configured_differs_from_runtime": bool(
                runtime_shared and _different(configured, runtime_shared)
            ),
            "difference_reason": "approved_runtime_parameter_override" if runtime_override else None,
            "source": "live_instance_config+runtime_state",
        }

    def _dca(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for pair, bot, name in (
            ("BTC-USDT", "dca-live-btcusdt-200", "dca_btcusdt_live_200.yml"),
            ("ETH-USDT", "dca-live-ethusdt-200", "dca_ethusdt_live_200.yml"),
        ):
            path = _latest(list(self.bots_path.glob(f"instances/{bot}*/conf/controllers/{name}")))
            configured = _public(_yaml(path), DCA_FIELDS) if path else {}
            result[pair] = {
                "bot": bot, "effective": configured,
                "parameter_sha256": _canonical_sha(configured),
                "application_state": "APPLIED" if configured else "UNAVAILABLE",
                "source": "live_instance_controller_config",
            }
        return result

    @staticmethod
    def _gate_states(robots: list[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for robot in robots:
            key = f"{robot.get('strategy')}:{robot.get('pair')}"
            result[key] = {
                "strategy": robot.get("strategy"), "pair": robot.get("pair"),
                "trading_normal": robot.get("trading_normal"),
                "final_permissions": robot.get("final_permissions", {}),
                "gates": robot.get("gate_statuses", []),
            }
        return result

    def _risks(self, robots: list[Mapping[str, Any]]) -> dict[str, Any]:
        grid = _json(self.grid_state / "guard_state.json")
        dca = _json(self.dca_state / "guard_state.json")
        return {
            "grid": {
                "enabled": grid.get("mechanisms", {}),
                "parameters": grid.get("mechanism_parameters", {}),
            },
            "dca": {
                "enabled": dca.get("mechanisms", {}),
                "parameters": dca.get("mechanism_parameters", {}),
            },
            "current": self._gate_states(robots),
        }

    def _approval_candidates(self) -> list[dict[str, Any]]:
        state = _json(self.approval_root / "automation_state.json")
        candidate_release = str(state.get("candidate_release_sha256") or "")
        terminal = {"ACTIVE", "REJECTED"}
        if not candidate_release or str(state.get("phase")) in terminal:
            return []
        output = []
        for path in self.approval_root.glob("approval-request-*.json"):
            value = _json(path)
            if str(value.get("release_sha256") or "") != candidate_release:
                continue
            release = str(value["release_sha256"])
            checks = value.get("checks") or state.get("checks") or {}
            output.append({
                "release_sha256": release,
                "model_sha256": value.get("model_sha256"),
                "model_week": value.get("candidate_model_week") or state.get("candidate_model_week"),
                "review_started_at": value.get("review_started_at"),
                "review_deadline": value.get("review_deadline"),
                "effective_start": value.get("activation_boundary"),
                "effective_end": value.get("candidate_effective_end"),
                "checks": checks,
                "status": str(state.get("phase") or "UNKNOWN"),
                "last_error": state.get("last_error"),
                "approval_mode": state.get("approval_mode"),
                "request_sha256": _file_sha(path),
            })
        return output

    def _model_lifecycle(self, active: Mapping[str, Any], *, now: datetime) -> list[dict[str, Any]]:
        """Persist only observed, healthy production activations; never infer from directory mtimes."""
        path = self.output / "model_lifecycle.json"
        ledger = _json(path)
        if ledger.get("schema") != LIFECYCLE_SCHEMA:
            ledger = {"schema": LIFECYCLE_SCHEMA, "current": {}, "history": []}
        release = str(active.get("release_sha256") or "")
        model = str(active.get("model_sha256") or "")
        generation = str(active.get("runtime_generation") or "")
        healthy = bool(active.get("source_healthy")) and str(active.get("cutover_phase")) in {
            "ACTIVE", "WARM_ACTIVE", "WARM_ACTIVE_PENDING_FOLD",
        }
        current = ledger.get("current") if isinstance(ledger.get("current"), dict) else {}
        identity = (release, model, generation)
        previous = (str(current.get("release_sha256") or ""),
                    str(current.get("model_sha256") or ""),
                    str(current.get("runtime_generation") or ""))
        cutover_at = None
        event_path = self.grid_state / "telegram_events.jsonl"
        try:
            for line in event_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                details = event.get("details", {}) if isinstance(event.get("details"), dict) else {}
                if (event.get("transition") == "MODEL_CUTOVER_STABLE"
                        and str(event.get("release_sha256") or "") == release
                        and str(details.get("runtime_generation") or "") == generation):
                    cutover_at = event.get("occurred_at")
        except (OSError, ValueError, TypeError):
            cutover_at = None
        if all(identity) and healthy:
            if not all(previous):
                ledger["current"] = {
                    "release_sha256": release, "model_sha256": model,
                    "runtime_generation": generation,
                    "model_week": active.get("model_week"), "week_start": active.get("week_start"),
                    "week_end": active.get("week_end"),
                    "activated_at": cutover_at,
                    "first_observed_at": now.astimezone(timezone.utc).isoformat(),
                    "activation_evidence": ("MODEL_CUTOVER_STABLE" if cutover_at
                                            else "bootstrap_current_without_cutover_timestamp"),
                }
            elif identity != previous:
                if not cutover_at:
                    return list(ledger.get("history", []))[:HISTORY_LIMIT]
                retired = {
                    **current,
                    "retired_at": cutover_at,
                    "replacement_reason": "MODEL_CUTOVER_STABLE",
                    "lifecycle_evidence": "observed_atomic_runtime_generation_transition",
                }
                history = list(ledger.get("history", []))
                if current.get("activated_at"):
                    history = [retired] + [item for item in history
                                           if item.get("release_sha256") != retired.get("release_sha256")]
                ledger["history"] = history[:HISTORY_LIMIT]
                ledger["current"] = {
                    "release_sha256": release, "model_sha256": model,
                    "runtime_generation": generation,
                    "model_week": active.get("model_week"), "week_start": active.get("week_start"),
                    "week_end": active.get("week_end"),
                    "activated_at": cutover_at,
                    "first_observed_at": now.astimezone(timezone.utc).isoformat(),
                    "activation_evidence": "MODEL_CUTOVER_STABLE",
                }
            else:
                current.update({"model_week": active.get("model_week"),
                                "week_start": active.get("week_start"),
                                "week_end": active.get("week_end")})
                if not current.get("activated_at") and cutover_at:
                    current.update({"activated_at": cutover_at,
                                    "activation_evidence": "MODEL_CUTOVER_STABLE"})
                ledger["current"] = current
            _atomic_json(path, ledger)
        return list(ledger.get("history", []))[:HISTORY_LIMIT]

    def _models(self, robots: list[Mapping[str, Any]], *, now: datetime) -> dict[str, Any]:
        gate = _json(self.grid_state / "xgboost_risk_gate.json")
        active = {
            key: gate.get(key) for key in (
                "package_id", "model_version", "runtime_generation", "release_sha256",
                "predecessor_release_sha256", "model_sha256", "feature_schema_sha256",
                "strategy_sha256", "data_sha256", "generated_at", "valid_until",
                "cutover_phase", "fold_boundary", "source_healthy", "reason",
            ) if key in gate
        }
        active["pairs"] = gate.get("pairs", {})
        weeks = [row for row in active["pairs"].values() if isinstance(row, Mapping)]
        active["model_week"] = next((row.get("model_week") for row in weeks
                                     if row.get("model_week") is not None), None)
        active["week_start"] = max((int(row["week_start"]) for row in weeks
                                    if row.get("week_start") is not None), default=None)
        active["week_end"] = min((int(row["week_end"]) for row in weeks
                                  if row.get("week_end") is not None), default=None)
        active["system_healthy"] = bool(active.get("source_healthy"))
        active["trading"] = {
            f"{robot.get('strategy')}:{robot.get('pair')}": {
                "trading_normal": robot.get("trading_normal"),
                "final_permissions": robot.get("final_permissions", {}),
            } for robot in robots
        }
        candidates = self._approval_candidates()
        for item in candidates:
            if item.get("model_week") is None and active.get("model_week") is not None:
                item["model_week"] = int(active["model_week"]) + 1
        return {
            "active": active,
            "candidate": candidates,
            "history": self._model_lifecycle(active, now=now),
        }

    def _publish_evidence(self, models: Mapping[str, Any]) -> dict[str, Any]:
        sets = []
        audits = self.release_root / "audits"
        manifests = list(audits.glob("*/manifest.json")) if audits.is_dir() else []
        generated = self.output.parent / "parameters"
        if generated.is_dir():
            manifests.extend(generated.glob("*/manifest.json"))
        manifests = sorted(set(manifests), key=lambda item: item.stat().st_mtime, reverse=True)
        active_model = str(models.get("active", {}).get("model_sha256") or "")
        active_release = str(models.get("active", {}).get("release_sha256") or "")
        for manifest_path in manifests:
            manifest = _json(manifest_path)
            images = manifest.get("images", [])
            if not isinstance(images, list):
                continue
            set_id = _canonical_sha({
                "manifest": _file_sha(manifest_path),
                "model": manifest.get("evidence_model_sha256"),
            })[:24]
            destination = self.evidence_root / set_id
            verified = []
            for item in images:
                if not isinstance(item, dict):
                    continue
                portable_name = Path(str(item.get("path", "")).replace("\\", "/")).name
                source = manifest_path.parent / portable_name
                expected = str(item.get("sha256") or "")
                if not source.is_file() or not expected or _file_sha(source) != expected:
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / source.name
                if not target.is_file() or _file_sha(target) != expected:
                    shutil.copy2(source, target)
                verified.append({
                    "strategy": item.get("strategy"), "pair": item.get("pair"),
                    "window": item.get("window"), "sha256": expected,
                    "size": target.stat().st_size, "width": item.get("width"),
                    "height": item.get("height"),
                    "relative_path": str(target.relative_to(self.output.parent)).replace("\\", "/"),
                })
            evidence_model = str(manifest.get("evidence_model_sha256") or "")
            production_model = str(manifest.get("production_model_sha256") or "")
            evidence_release = str(manifest.get("release_sha256") or "")
            if not evidence_release:
                relation = "UNBOUND_LEGACY"
            elif evidence_release == active_release and production_model == active_model:
                relation = "EXACT"
            else:
                relation = "BOUND_OTHER_MODEL"
            sets.append({
                "evidence_set_id": set_id, "generated_at": manifest.get("generated_at"),
                "execution_policy": manifest.get("execution_policy"),
                "production_model_sha256": production_model,
                "evidence_model_sha256": evidence_model,
                "release_sha256": evidence_release or None,
                "relation_to_active": relation,
                "notice": None if relation == "EXACT" else (
                    "旧证据未同时绑定发布版本和模型，已保留审计但不提供导航"
                    if relation == "UNBOUND_LEGACY" else "该证据属于其他已绑定模型"
                ),
                "attachments": verified,
                "manifest_sha256": _file_sha(manifest_path),
            })
            if len(sets) == HISTORY_LIMIT + 2:
                break
        catalog = {
            "schema": EVIDENCE_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sets": sets,
        }
        _atomic_json(self.output.parent / "model_evidence_catalog.json", catalog)
        return catalog

    @staticmethod
    def _version_snapshot(catalog: Mapping[str, Any]) -> dict[str, Any]:
        """Return only fields whose change creates a parameter/model version."""
        risks = catalog.get("risks", {})
        models = catalog.get("models", {})
        active = models.get("active", {})
        stable_active = {
            key: active.get(key) for key in (
                "package_id", "model_version", "runtime_generation", "release_sha256",
                "predecessor_release_sha256", "model_sha256", "feature_schema_sha256",
                "strategy_sha256", "data_sha256",
            ) if key in active
        }
        stable_pairs = {}
        for pair, value in active.get("pairs", {}).items():
            if not isinstance(value, Mapping):
                continue
            stable_pairs[pair] = {
                key: value.get(key) for key in (
                    "pair", "source_pair", "model_week", "week_start", "week_end",
                    "week_model_sha256", "entry_threshold",
                ) if key in value
            }
        stable_active["pairs"] = stable_pairs
        return {
            "grid": catalog.get("grid", {}),
            "dca": catalog.get("dca", {}),
            "risks": {
                strategy: risks.get(strategy, {}) for strategy in ("grid", "dca")
            },
            "models": {"active": stable_active},
        }

    def _record_history(self, catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
        snapshot = self._version_snapshot(catalog)
        digest = _canonical_sha(snapshot)
        path = self.history_root / f"{digest}.json"
        if not path.is_file():
            _atomic_json(path, {
                "schema": HISTORY_SCHEMA,
                "recorded_at": catalog["generated_at"],
                "catalog_sha256": digest,
                **snapshot,
            })
        else:
            os.utime(path, None)
        paths = []
        for candidate in self.history_root.glob("*.json"):
            if _json(candidate).get("schema") != HISTORY_SCHEMA:
                candidate.unlink(missing_ok=True)
                continue
            paths.append(candidate)
        paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        for stale in paths[HISTORY_LIMIT:]:
            stale.unlink(missing_ok=True)
        return [{
            "catalog_sha256": path.stem,
            "recorded_at": _json(path).get("recorded_at"),
        } for path in paths[:HISTORY_LIMIT]]

    def publish(self, robots: list[Mapping[str, Any]], *, now: datetime) -> dict[str, Any]:
        catalog: dict[str, Any] = {
            "schema": SCHEMA,
            "generated_at": now.astimezone(timezone.utc).isoformat(),
            "history_limit": HISTORY_LIMIT,
            "read_only": True,
            "grid": self._grid(),
            "dca": self._dca(),
            "risks": self._risks(robots),
            "models": self._models(robots, now=now),
        }
        catalog["catalog_sha256"] = _canonical_sha(self._version_snapshot(catalog))
        evidence = self._publish_evidence(catalog["models"])
        def evidence_count(subject: Mapping[str, Any]) -> int:
            identities = {
                (row.get("strategy"), row.get("pair"))
                for item in evidence["sets"]
                if item.get("release_sha256") == subject.get("release_sha256")
                and item.get("production_model_sha256") == subject.get("model_sha256")
                for row in item.get("attachments", []) if row.get("window") == "360d"
            }
            return len(identities)
        catalog["models"]["active"]["exact_evidence_count"] = evidence_count(
            catalog["models"]["active"])
        for group in ("candidate", "history"):
            for item in catalog["models"].get(group, []):
                item["exact_evidence_count"] = evidence_count(item)
        catalog["evidence"] = [{
            "evidence_set_id": item["evidence_set_id"],
            "relation_to_active": item["relation_to_active"],
            "attachment_count": len(item["attachments"]),
        } for item in evidence["sets"]]
        catalog["history"] = self._record_history(catalog)
        _atomic_json(self.output.parent / "management_parameter_catalog.json", catalog)
        return catalog
