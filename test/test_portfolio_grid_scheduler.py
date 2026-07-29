import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scheduler.portfolio_grid_scheduler import CONFIG_NAME, Scheduler
from scripts.portfolio_grid_core import GridParams


class PortfolioGridSchedulerTest(unittest.TestCase):
    def scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.instance_name = "walk-forward-portfolio-grid"
        scheduler.profile = "master_account"
        scheduler.config_name = CONFIG_NAME
        scheduler.quote_asset = "USDT"
        scheduler.image = "hummingbot/hummingbot:latest"
        scheduler.initial_quote = 10000
        scheduler.workers = 1
        scheduler.maker_fee_rate = 0
        scheduler.taker_fee_rate = 0.0002
        scheduler.pairs = ["BTC-FDUSD"]
        scheduler.initial_train_days = 30
        scheduler.train_days = 7
        scheduler.validation_days = 7
        scheduler.min_validation_return = 0
        scheduler.min_validation_score = 0
        scheduler.max_validation_drawdown = 0.04
        scheduler.min_validation_cycles = 10
        scheduler.deploy_retries = 3
        scheduler.deploy_retry_seconds = 0
        scheduler.archive_timeout = 10
        return scheduler

    def test_config_name_is_stable(self):
        self.assertEqual("walk_forward_portfolio_grid.yml", CONFIG_NAME)

    def test_initial_period_transitions_at_first_following_monday(self):
        scheduler = self.scheduler()
        shanghai = ZoneInfo("Asia/Shanghai")
        state = {
            "period": "initial-20260710-2327",
            "deployed_at": "2026-07-11T02:07:44+00:00",
        }
        period, train_days, validation_days, end = scheduler.target_period(
            datetime(2026, 7, 13, 0, 11, tzinfo=shanghai), state,
        )
        self.assertEqual("2026-07-13", period)
        self.assertEqual((7, 7), (train_days, validation_days))
        self.assertEqual(datetime(2026, 7, 13, 0, 0, tzinfo=shanghai), end)

    def test_initial_deployed_after_switch_waits_until_next_week(self):
        scheduler = self.scheduler()
        shanghai = ZoneInfo("Asia/Shanghai")
        state = {
            "period": "initial-20260713-1200",
            "deployed_at": "2026-07-13T04:00:00+00:00",
        }
        period, train_days, validation_days, _ = scheduler.target_period(
            datetime(2026, 7, 13, 13, 0, tzinfo=shanghai), state,
        )
        self.assertEqual("initial-20260713-1200", period)
        self.assertEqual((30, 7), (train_days, validation_days))

    def test_validation_gate_rejects_unprofitable_or_risky_candidate(self):
        scheduler = self.scheduler()
        result = {
            "net_pnl_pct": -0.01,
            "max_drawdown_pct": -0.05,
            "liquidated": True,
            "completed_cycles": 2,
        }
        reasons = scheduler.validation_rejection_reasons(result, scheduler.score(result))
        self.assertEqual(5, len(reasons))

    def test_validation_gate_accepts_profitable_low_drawdown_candidate(self):
        scheduler = self.scheduler()
        result = {
            "net_pnl_pct": 0.03,
            "max_drawdown_pct": -0.01,
            "liquidated": False,
            "completed_cycles": 20,
        }
        self.assertEqual([], scheduler.validation_rejection_reasons(result, scheduler.score(result)))

    @patch("scheduler.portfolio_grid_scheduler.simulate_portfolio")
    @patch("scheduler.portfolio_grid_scheduler.select_params_parallel")
    def test_rejected_validation_keeps_current_bot(self, select_params, simulate):
        scheduler = self.scheduler()
        params = GridParams(0.08, 24, 0.02, 0.003, 0.005)
        training_result = {
            "net_pnl_pct": 0.05, "max_drawdown_pct": -0.01, "liquidated": False,
            "completed_cycles": 50,
        }
        validation_result = {
            "net_pnl_pct": -0.01, "max_drawdown_pct": -0.02, "liquidated": False,
            "completed_cycles": 20,
        }
        select_params.return_value = (params, training_result, 0.035, pd.DataFrame([{"candidate": 1}]))
        simulate.return_value = validation_result
        scheduler.load_candles = Mock(return_value={"BTC-FDUSD": pd.DataFrame()})
        scheduler.stop_and_archive = Mock()
        scheduler.deploy = Mock()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler.reports = root / "reports"
            scheduler.state_path = root / "scheduler_state.json"
            scheduler.bots = root / "bots"
            scheduler.select_and_deploy(
                "2026-07-27", 7, 7,
                datetime(2026, 7, 27, tzinfo=ZoneInfo("Asia/Shanghai")),
                {"active_instance": "existing-fdusd-grid", "period": "adopted-manual"},
            )
            saved = scheduler.load_json(scheduler.state_path, {})

        self.assertEqual("2026-07-27", saved["evaluated_period"])
        self.assertFalse(saved["last_evaluation"]["qualified"])
        scheduler.stop_and_archive.assert_not_called()
        scheduler.deploy.assert_not_called()

    def test_migrate_legacy_config_copies_yaml_and_updates_state(self):
        scheduler = self.scheduler()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler.bots = root / "bots"
            scheduler.state_path = root / "scheduler_state.json"
            config_dir = scheduler.bots / "conf" / "scripts"
            config_dir.mkdir(parents=True)
            legacy = config_dir / "walk_forward_portfolio_grid_2026-07-20.yml"
            legacy.write_text("grid_range: 0.08\n", encoding="utf-8")
            state = {
                "config_file": legacy.name,
                "selection": {"config_file": legacy.name},
            }

            migrated = scheduler.migrate_legacy_config(state)

            self.assertEqual(CONFIG_NAME, migrated["config_file"])
            self.assertEqual(CONFIG_NAME, migrated["selection"]["config_file"])
            self.assertEqual("grid_range: 0.08\n", (config_dir / CONFIG_NAME).read_text(encoding="utf-8"))

    def test_deploy_uses_exact_name_and_persists_it(self):
        scheduler = self.scheduler()
        scheduler.api = Mock(return_value={
            "success": True,
            "unique_instance_name": scheduler.instance_name,
        })
        with TemporaryDirectory() as directory:
            scheduler.state_path = Path(directory) / "scheduler_state.json"
            state = {"period": "2026-07-27"}
            scheduler.deploy(CONFIG_NAME, state, replace=True)

        scheduler.api.assert_called_once()
        method, path, payload = scheduler.api.call_args.args
        self.assertEqual("POST", method)
        self.assertEqual("/bot-orchestration/deploy-v2-script?use_timestamp=false", path)
        self.assertEqual(scheduler.instance_name, payload["instance_name"])
        self.assertEqual(scheduler.instance_name, state["active_instance"])

    def test_deploy_reconciles_timeout_after_container_was_created(self):
        scheduler = self.scheduler()
        scheduler.api = Mock(side_effect=requests.ReadTimeout("late response"))
        scheduler.bot_is_running = Mock(return_value=True)
        with TemporaryDirectory() as directory:
            scheduler.state_path = Path(directory) / "scheduler_state.json"
            state = {}
            scheduler.deploy(CONFIG_NAME, state, replace=False)

        self.assertEqual(scheduler.instance_name, state["active_instance"])
        scheduler.bot_is_running.assert_called_once_with(scheduler.instance_name)

    def test_bot_status_failure_is_not_treated_as_missing(self):
        scheduler = self.scheduler()
        scheduler.api = Mock(side_effect=requests.ConnectionError("api unavailable"))
        with self.assertRaises(requests.ConnectionError):
            scheduler.bot_is_running(scheduler.instance_name)

    @patch("scheduler.portfolio_grid_scheduler.time.sleep")
    def test_stop_and_archive_waits_until_container_is_removed(self, sleep):
        scheduler = self.scheduler()
        scheduler.api = Mock(return_value={"status": "success"})
        scheduler.instance_exists = Mock(side_effect=[True, False])
        scheduler.stop_and_archive("legacy-grid-20260727-020414")

        self.assertIn("/bot-orchestration/stop-and-archive-bot/legacy-grid-20260727-020414", scheduler.api.call_args.args[1])
        self.assertEqual(2, scheduler.instance_exists.call_count)
        sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
