from datetime import datetime, timezone
from unittest import TestCase

from stocks_runtime.market_calendar import (
    persisted_market_state_is_current,
    phases_conflict,
    xnys_market_state,
)


class XnysMarketCalendarTests(TestCase):
    def test_dst_regular_session(self):
        state = xnys_market_state(datetime(2026, 3, 9, 14, 0, tzinfo=timezone.utc))
        self.assertEqual("MARKET_OPEN", state.phase)

    def test_weekend_and_holiday_are_closed(self):
        self.assertEqual(
            "MARKET_CLOSED",
            xnys_market_state(datetime(2026, 8, 23, 15, 0, tzinfo=timezone.utc)).phase,
        )
        self.assertEqual(
            "MARKET_CLOSED",
            xnys_market_state(datetime(2026, 7, 3, 15, 0, tzinfo=timezone.utc)).phase,
        )

    def test_early_close_enters_post_market(self):
        state = xnys_market_state(datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc))
        self.assertEqual("POST_MARKET", state.phase)

    def test_overnight_and_closed_are_semantically_compatible(self):
        self.assertFalse(phases_conflict("OVERNIGHT", "MARKET_CLOSED"))
        self.assertTrue(phases_conflict("MARKET_OPEN", "PRE_MARKET"))

    def test_persisted_state_must_match_current_new_york_date(self):
        local = xnys_market_state(datetime(2026, 8, 25, 15, 0, tzinfo=timezone.utc))
        self.assertTrue(persisted_market_state_is_current(
            trading_date="2026-08-25", valid_until=local.valid_until, local=local,
        ))
        self.assertFalse(persisted_market_state_is_current(
            trading_date="2026-08-24", valid_until=local.valid_until, local=local,
        ))
