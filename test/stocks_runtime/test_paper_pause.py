import asyncio
import os
import types
import unittest
from unittest.mock import AsyncMock, patch

from stocks_runtime.ensure_database import main
from stocks_runtime.settings import validate_paper_startup_policy


class PaperPauseTest(unittest.TestCase):
    def test_paused_startup_requires_no_database_or_driver(self):
        with patch.dict(os.environ, {"BINANCE_STOCKS_PAPER_PAUSED": "true"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "paused"):
                asyncio.run(main())

    def test_malformed_pause_flag_fails_closed(self):
        with patch.dict(os.environ, {"BINANCE_STOCKS_PAPER_PAUSED": "treu"}, clear=True):
            with self.assertRaises(ValueError):
                validate_paper_startup_policy()

    def test_paper_pause_does_not_change_live_authorization(self):
        with patch.dict(os.environ, {
            "BINANCE_STOCKS_RUNTIME_MODE": "LIVE", "BINANCE_STOCKS_PAPER_PAUSED": "true",
        }, clear=True):
            validate_paper_startup_policy()

    def _bootstrap(self, exists, create_enabled=None):
        connection = types.SimpleNamespace(
            fetchval=AsyncMock(return_value=exists), execute=AsyncMock(), close=AsyncMock(),
        )
        driver = types.SimpleNamespace(connect=AsyncMock(return_value=connection))
        values = {"DATABASE_URL": "postgresql://fake:fake@localhost/hummingbot_api"}
        if create_enabled is not None:
            values["BINANCE_STOCKS_DATABASE_CREATE_ENABLED"] = create_enabled
        with patch.dict(os.environ, values, clear=True), patch.dict("sys.modules", {"asyncpg": driver}):
            if not exists and create_enabled != "true":
                with self.assertRaisesRegex(RuntimeError, "creation authorization"):
                    asyncio.run(main())
            else:
                asyncio.run(main())
        connection.close.assert_awaited_once()
        return connection, driver

    def test_missing_database_is_not_recreated_by_restart(self):
        for value in (None, "false", "invalid"):
            connection, _ = self._bootstrap(None, value)
            connection.execute.assert_not_awaited()

    def test_existing_database_does_not_require_creation_authorization(self):
        connection, driver = self._bootstrap(1)
        connection.execute.assert_not_awaited()
        self.assertTrue(driver.connect.call_args.args[0].endswith("/postgres"))
        connection.fetchval.assert_awaited_once_with(
            "SELECT 1 FROM pg_database WHERE datname=$1", "hummingbot_stocks"
        )

    def test_explicit_create_targets_only_dedicated_stock_database(self):
        connection, _ = self._bootstrap(None, "true")
        connection.execute.assert_awaited_once_with('CREATE DATABASE "hummingbot_stocks"')
