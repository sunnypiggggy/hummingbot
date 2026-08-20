"""Weekly parameter selection and official Hummingbot API orchestration."""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml

try:
    from portfolio_grid_core import default_search_space, select_params_parallel, simulate_portfolio
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from portfolio_grid_core import default_search_space, select_params_parallel, simulate_portfolio


LOG = logging.getLogger("portfolio-grid-scheduler")
SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "TRX-USDT"]
CONFIG_NAME = "walk_forward_portfolio_grid.yml"
DEFAULT_PAPER_BALANCES = {"BTC": 1.0, "ETH": 20.0, "SOL": 100.0, "BNB": 100.0, "XRP": 100000.0,
                          "DOGE": 1000000.0, "ADA": 100000.0, "LINK": 10000.0, "AVAX": 10000.0,
                          "TRX": 1000000.0, "USDT": 100000.0}


class Scheduler:
    def __init__(self):
        self.bots = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
        self.root = Path(os.getenv("SCHEDULER_STATE_PATH", "/workspace/state"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "candles"
        self.reports = self.root / "selections"
        self.state_path = self.root / "scheduler_state.json"
        self.api_url = os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/")
        self.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])
        self.instance_prefix = os.getenv("PORTFOLIO_INSTANCE_PREFIX", "walk-forward-portfolio-grid")
        self.instance_name = os.getenv("PORTFOLIO_INSTANCE_NAME", self.instance_prefix)
        self.profile = os.getenv("PORTFOLIO_CREDENTIALS_PROFILE", "master_account")
        self.config_name = os.getenv("PORTFOLIO_CONFIG_NAME", CONFIG_NAME)
        self.quote_asset = os.getenv("PORTFOLIO_QUOTE_ASSET", "USDT")
        self.image = os.getenv("PORTFOLIO_RUNTIME_IMAGE", "hummingbot/walk-forward-runtime:local")
        self.interval = os.getenv("PORTFOLIO_INTERVAL", "5m")
        self.initial_quote = float(os.getenv("PORTFOLIO_INITIAL_QUOTE", "10000"))
        self.workers = int(os.getenv("PORTFOLIO_OPTIMIZATION_WORKERS", "4"))
        self.initial_train_days = int(os.getenv("PORTFOLIO_INITIAL_TRAIN_DAYS", "30"))
        self.train_days = int(os.getenv("PORTFOLIO_TRAIN_DAYS", "7"))
        self.validation_days = int(os.getenv("PORTFOLIO_VALIDATION_DAYS", "7"))
        self.maker_fee_rate = float(os.getenv("PORTFOLIO_MAKER_FEE_RATE", "0.0002"))
        self.taker_fee_rate = float(os.getenv("PORTFOLIO_TAKER_FEE_RATE", "0.0002"))
        self.min_validation_return = float(os.getenv("PORTFOLIO_MIN_VALIDATION_RETURN", "0"))
        self.min_validation_score = float(os.getenv("PORTFOLIO_MIN_VALIDATION_SCORE", "0"))
        self.max_validation_drawdown = float(os.getenv("PORTFOLIO_MAX_VALIDATION_DRAWDOWN", "0.04"))
        self.min_validation_cycles = int(os.getenv("PORTFOLIO_MIN_VALIDATION_CYCLES", "10"))
        self.deploy_retries = int(os.getenv("PORTFOLIO_DEPLOY_RETRIES", "5"))
        self.deploy_retry_seconds = float(os.getenv("PORTFOLIO_DEPLOY_RETRY_SECONDS", "5"))
        self.archive_timeout = float(os.getenv("PORTFOLIO_ARCHIVE_TIMEOUT", "90"))
        configured_pairs = [pair.strip().upper() for pair in os.getenv(
            "PORTFOLIO_PAIRS", ",".join(DEFAULT_PAIRS),
        ).split(",") if pair.strip()]
        self.pairs = configured_pairs[:int(os.getenv("PORTFOLIO_PAIR_LIMIT", str(len(configured_pairs))))]
        self.paper_balances = json.loads(os.getenv(
            "PORTFOLIO_PAPER_BALANCES_JSON", json.dumps(DEFAULT_PAPER_BALANCES),
        ))
        self.adopt_instance = os.getenv("PORTFOLIO_ADOPT_INSTANCE")
        self.adopt_config = os.getenv("PORTFOLIO_ADOPT_CONFIG")
        self.session = requests.Session()

    def run_forever(self):
        while True:
            try:
                self.reconcile()
            except Exception:
                LOG.exception("Scheduler cycle failed; keeping the current bot unchanged.")
            time.sleep(60)

    def reconcile(self):
        self.ensure_api_staging()
        now = datetime.now(SHANGHAI)
        state = self.load_json(self.state_path, {})
        state = self.adopt_existing_instance(state)
        state = self.migrate_legacy_config(state)
        period, train_days, validation_days, validation_end = self.target_period(now, state)
        if state.get("period") != period and state.get("evaluated_period") != period:
            self.select_and_deploy(period, train_days, validation_days, validation_end, state)
            return
        active = state.get("active_instance")
        if active and not self.bot_is_running(active):
            LOG.warning("Current bot %s is unavailable; redeploying the current parameter version.", active)
            self.deploy(state["config_file"], state, replace=False)

    def target_period(self, now: datetime, state: Dict) -> Tuple[str, int, int, datetime]:
        monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        switch_at = monday.replace(minute=10)
        if not state:
            return (f"initial-{now.strftime('%Y%m%d-%H%M')}", self.initial_train_days,
                    self.validation_days, now.replace(second=0, microsecond=0))
        if str(state.get("period", "")).startswith("initial-"):
            deployed_at = self.parse_datetime(state.get("deployed_at"))
            if now < switch_at or (deployed_at is not None and deployed_at.astimezone(SHANGHAI) >= switch_at):
                return (state["period"], self.initial_train_days, self.validation_days,
                        now.replace(second=0, microsecond=0))
        if now >= switch_at:
            return monday.strftime("%Y-%m-%d"), self.train_days, self.validation_days, monday
        previous = monday - timedelta(days=7)
        return previous.strftime("%Y-%m-%d"), self.train_days, self.validation_days, monday

    def ensure_api_staging(self):
        scripts = self.bots / "scripts"
        profile = self.bots / "credentials" / self.profile
        configs = self.bots / "conf" / "scripts"
        for directory in (scripts, profile, configs):
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2("/app/walk_forward_portfolio_grid.py", scripts / "walk_forward_portfolio_grid.py")
        if not (profile / "conf_client.yml").exists():
            conf = {"instance_id": "api-managed", "log_level": "INFO", "db_mode": {"db_engine": "sqlite"},
                    "paper_trade": {"paper_trade_exchanges": ["binance"], "paper_trade_account_balance": self.paper_balances},
                    "mqtt_bridge": {"mqtt_host": "127.0.0.1", "mqtt_port": 1883, "mqtt_namespace": "hbot",
                                    "mqtt_ssl": False, "mqtt_logger": True, "mqtt_notifier": True,
                                    "mqtt_commands": True, "mqtt_events": True, "mqtt_external_events": True,
                                    "mqtt_autostart": False}}
            (profile / "conf_client.yml").write_text(yaml.safe_dump(conf, sort_keys=False))
            (profile / "conf_fee_overrides.yml").write_text(
                "template_version: 14\n"
                f"binance_maker_percent_fee: {self.maker_fee_rate * 100:g}\n"
                f"binance_taker_percent_fee: {self.taker_fee_rate * 100:g}\n"
            )

    def adopt_existing_instance(self, state: Dict) -> Dict:
        if state or not self.adopt_instance:
            return state
        if not self.bot_is_running(self.adopt_instance):
            raise RuntimeError(f"Configured adoption instance is not running: {self.adopt_instance}")
        adopted = {
            "period": "adopted-manual",
            "config_file": self.adopt_config or self.config_name,
            "active_instance": self.adopt_instance,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_json(self.state_path, adopted)
        LOG.info("Adopted existing Grid instance %s", self.adopt_instance)
        return adopted

    def migrate_legacy_config(self, state: Dict) -> Dict:
        legacy_name = state.get("config_file")
        if not legacy_name or legacy_name == self.config_name:
            return state
        config_dir = self.bots / "conf" / "scripts"
        source = config_dir / legacy_name
        if not source.exists():
            raise FileNotFoundError(f"Legacy Grid config is missing: {source}")
        shutil.copy2(source, config_dir / self.config_name)
        migrated = {**state, "config_file": self.config_name}
        if isinstance(state.get("selection"), dict):
            migrated["selection"] = {**state["selection"], "config_file": self.config_name}
        self.save_json(self.state_path, migrated)
        LOG.info("Migrated Grid config name from %s to %s", legacy_name, self.config_name)
        return migrated

    def select_and_deploy(self, period: str, train_days: int, validation_days: int,
                          validation_end: datetime, state: Dict):
        validation_start = validation_end - timedelta(days=validation_days)
        train_end = validation_start
        train_start = train_end - timedelta(days=train_days)
        training_candles = self.load_candles(train_start, train_end)
        validation_candles = self.load_candles(validation_start, validation_end)
        params, training_result, training_score, candidates = select_params_parallel(
            training_candles, default_search_space(), self.initial_quote, self.maker_fee_rate,
            0.08, 24, self.workers, maker_fee_rate=self.maker_fee_rate,
            taker_fee_rate=self.taker_fee_rate,
        )
        validation_result = simulate_portfolio(
            validation_candles, params, self.initial_quote, self.maker_fee_rate, 0.08, 24,
            maker_fee_rate=self.maker_fee_rate, taker_fee_rate=self.taker_fee_rate,
        )
        validation_score = self.score(validation_result)
        rejection_reasons = self.validation_rejection_reasons(validation_result, validation_score)
        report_dir = self.reports / period
        report_dir.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(report_dir / "candidate_evaluations.csv", index=False)
        evaluation = {
            "period": period,
            "train_start": train_start.isoformat(), "train_end": train_end.isoformat(),
            "validation_start": validation_start.isoformat(), "validation_end": validation_end.isoformat(),
            "params": {key: getattr(params, key) for key in params.__dataclass_fields__},
            "training_result": training_result, "training_score": training_score,
            "validation_result": validation_result, "validation_score": validation_score,
            "qualified": not rejection_reasons, "rejection_reasons": rejection_reasons,
            "maker_fee_rate": self.maker_fee_rate, "taker_fee_rate": self.taker_fee_rate,
            "config_file": self.config_name,
        }
        (report_dir / "selection.json").write_text(json.dumps(evaluation, indent=2, default=str))
        if rejection_reasons:
            retained = {
                **state, "evaluated_period": period, "last_evaluation": evaluation,
                "last_rejected_at": datetime.now(timezone.utc).isoformat(),
            }
            self.save_json(self.state_path, retained)
            LOG.warning("Rejected %s parameters; keeping %s: %s", period, state.get("active_instance"),
                        "; ".join(rejection_reasons))
            return

        runtime_state = self.load_runtime_state(state.get("active_instance"))
        config_name = self.config_name
        config = {
            "script_file_name": "walk_forward_portfolio_grid.py", "controllers_config": [],
            "parameter_version": period, "exchange": "binance_paper_trade", "trading_pairs": self.pairs,
            "quote_asset": self.quote_asset, "grid_range": params.grid_range, "grid_levels": params.grid_levels,
            "order_quote_pct": params.order_quote_pct, "take_profit": params.take_profit,
            "move_threshold": params.move_threshold, "portfolio_stop_loss": 0.08,
            "order_refresh_time": 60, "min_grid_move_seconds": 0, "cooldown_seconds": 86400,
            "min_order_quote": 10, "initial_peak_equity": runtime_state.get("peak_equity", 0),
            "initial_cooldown_until": runtime_state.get("cooldown_until", 0),
            "initial_grid_states": runtime_state.get("grid_states", {}),
        }
        config_path = self.bots / "conf" / "scripts" / config_name
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        if state.get("active_instance"):
            self.stop_and_archive(state["active_instance"])
            time.sleep(20)
        new_state = {
            **state, "period": period, "evaluated_period": period, "config_file": config_name,
            "selection": evaluation, "active_instance": None,
        }
        self.deploy(config_name, new_state, replace=True)

    @staticmethod
    def score(result: Dict[str, float]) -> float:
        value = result["net_pnl_pct"] - abs(result["max_drawdown_pct"]) * 1.5
        return value - 0.25 if result["liquidated"] else value

    def validation_rejection_reasons(self, result: Dict[str, float], score: float) -> list[str]:
        reasons = []
        if result["net_pnl_pct"] <= self.min_validation_return:
            reasons.append(
                f"validation return {result['net_pnl_pct']:.2%} <= {self.min_validation_return:.2%}"
            )
        if score <= self.min_validation_score:
            reasons.append(f"validation score {score:.4f} <= {self.min_validation_score:.4f}")
        if result["max_drawdown_pct"] < -self.max_validation_drawdown:
            reasons.append(
                f"validation drawdown {result['max_drawdown_pct']:.2%} < {-self.max_validation_drawdown:.2%}"
            )
        if result["liquidated"]:
            reasons.append("validation triggered portfolio liquidation")
        if result["completed_cycles"] < self.min_validation_cycles:
            reasons.append(
                f"validation cycles {result['completed_cycles']} < {self.min_validation_cycles}"
            )
        return reasons

    def deploy(self, config_name: str, state: Dict, replace: bool):
        payload = {
            "instance_name": self.instance_name, "credentials_profile": self.profile, "image": self.image,
            "script": "walk_forward_portfolio_grid", "script_config": config_name, "headless": True,
        }
        instance = None
        last_error = None
        for attempt in range(1, self.deploy_retries + 1):
            try:
                response = self.api(
                    "POST", "/bot-orchestration/deploy-v2-script?use_timestamp=false", payload,
                )
                candidate = response.get("unique_instance_name")
                if response.get("success") and candidate == self.instance_name:
                    instance = candidate
                    break
                last_error = RuntimeError(f"API deployment failed: {response}")
            except requests.RequestException as error:
                last_error = error
                LOG.warning("Deploy request attempt %d failed: %s", attempt, error)

            # The API can time out after Docker has already created the bot.
            # Reconcile the fixed name before retrying; retries cannot create a
            # timestamped sibling because the exact name is reserved.
            try:
                if self.bot_is_running(self.instance_name):
                    LOG.warning("Reconciled timed-out deployment as running instance %s", self.instance_name)
                    instance = self.instance_name
                    break
            except requests.RequestException as error:
                last_error = error

            if attempt < self.deploy_retries:
                time.sleep(self.deploy_retry_seconds)

        if instance is None:
            raise RuntimeError(f"Unable to deploy fixed Grid instance {self.instance_name}") from last_error
        state["active_instance"] = instance
        state["deployed_at"] = datetime.now(timezone.utc).isoformat()
        self.save_json(self.state_path, state)
        LOG.info("Deployed %s using %s", instance, config_name)

    def stop_and_archive(self, instance: str):
        self.api(
            "POST",
            f"/bot-orchestration/stop-and-archive-bot/{instance}"
            "?skip_order_cancellation=false&archive_locally=true",
        )
        deadline = time.monotonic() + self.archive_timeout
        while time.monotonic() < deadline:
            if not self.instance_exists(instance):
                return
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for {instance} to stop and archive")

    def instance_exists(self, instance: str) -> bool:
        active = self.api("GET", "/docker/active-containers")
        exited = self.api("GET", "/docker/exited-containers")
        return any(container.get("name") == instance for container in [*active, *exited])

    def bot_is_running(self, instance: str) -> bool:
        containers = self.api("GET", "/docker/active-containers")
        return any(container.get("name") == instance and container.get("status") == "running"
                   for container in containers)

    def load_candles(self, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        return {pair: self.load_pair_candles(pair, start, end) for pair in self.pairs}

    def load_pair_candles(self, pair: str, start: datetime, end: datetime) -> pd.DataFrame:
        path = self.cache / pair / f"{self.interval}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        missing_start = start_ms if existing.empty else min(start_ms, int(existing["timestamp"].min() * 1000))
        missing_end = end_ms if existing.empty else max(end_ms, int(existing["timestamp"].max() * 1000) + 300000)
        if existing.empty or missing_start < int(existing["timestamp"].min() * 1000) or missing_end > int(existing["timestamp"].max() * 1000) + 300000:
            rows = self.download_klines(pair, missing_start, missing_end)
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            existing = existing.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            existing.to_parquet(path, index=False)
        window = existing[(existing["timestamp"] >= start.timestamp()) & (existing["timestamp"] < end.timestamp())].copy()
        if len(window) < 20:
            raise RuntimeError(f"Insufficient cached candles for {pair}: {len(window)}")
        return window.reset_index(drop=True)

    def download_klines(self, pair: str, start_ms: int, end_ms: int):
        rows, cursor = [], start_ms
        symbol = pair.replace("-", "")
        while cursor < end_ms:
            response = self.session.get("https://api.binance.com/api/v3/klines", params={
                "symbol": symbol, "interval": self.interval, "startTime": cursor, "endTime": end_ms, "limit": 1000,
            }, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break
            rows.extend({"timestamp": item[0] / 1000, "open": float(item[1]), "high": float(item[2]),
                         "low": float(item[3]), "close": float(item[4]), "volume": float(item[5])} for item in payload)
            cursor = int(payload[-1][0]) + 300000
            time.sleep(0.05)
        return rows

    def load_runtime_state(self, instance: str | None) -> Dict:
        if not instance:
            return {}
        return self.load_json(self.bots / "instances" / instance / "data" / "runtime_state.json", {})

    def api(self, method: str, path: str, payload: Dict | None = None) -> Dict:
        response = self.session.request(method, self.api_url + path, auth=self.auth, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def parse_datetime(value) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def load_json(path: Path, default: Dict) -> Dict:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def save_json(path: Path, value: Dict):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, default=str))
        temporary.replace(path)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    Scheduler().run_forever()
