import time
from decimal import Decimal
from unittest import TestCase

from stocks_runtime.paper_broker import PaperQuote, PostgresPaperBroker, _checkpoint_json, _decode_checkpoint_row


class _Ledger:
    SCHEMA = "binance_stocks_paper"


class PaperBrokerPureTests(TestCase):
    def test_checkpoint_jsonb_text_is_decoded_for_restart_recovery(self):
        row = {
            "executor_id": "paper-executor-1",
            "config": '{"id":"paper-executor-1"}',
            "metadata": '{"account_name":"stocks_managed"}',
            "state": '{"phase":"OPEN"}',
        }
        decoded = _decode_checkpoint_row(row)
        self.assertEqual("paper-executor-1", decoded["config"]["id"])
        self.assertEqual("stocks_managed", decoded["metadata"]["account_name"])
        self.assertEqual("OPEN", decoded["state"]["phase"])

    def test_quote_parses_bbo_sizes_and_has_stable_event_id(self):
        payload = {
            "s": "AAPL",
            "bp": "200.10",
            "ap": "200.12",
            "bs": "0.75",
            "as": "0.50",
            "T": 1_786_000_000_000,
        }
        first = PaperQuote.from_payload(payload)
        second = PaperQuote.from_payload(dict(payload))
        self.assertTrue(first.valid)
        self.assertEqual(Decimal("0.50"), first.ask_size)
        self.assertEqual(first.event_id, second.event_id)

    def test_quote_without_displayed_size_is_not_fabricated(self):
        quote = PaperQuote.from_payload({
            "symbol": "AAPL", "bidPrice": "200", "askPrice": "201", "time": time.time()
        })
        self.assertTrue(quote.valid)
        self.assertEqual(Decimal("0"), quote.bid_size)
        self.assertEqual(Decimal("0"), quote.ask_size)

    def test_rest_event_time_is_stable_across_replay(self):
        payload = {
            "symbol": "AAPL",
            "bid": "100",
            "ask": "101",
            "bidQty": "1",
            "askQty": "0.5",
            "eventTime": 1_786_700_001_000,
        }
        first = PaperQuote.from_payload(payload, now=1)
        second = PaperQuote.from_payload(payload, now=999)
        self.assertEqual(1_786_700_001, first.event_time)
        self.assertEqual(first.event_id, second.event_id)

    def test_fee_is_cumulative_per_order_not_per_partial_fill(self):
        self.assertEqual(Decimal("0"), PostgresPaperBroker.cumulative_fee(Decimal("0")))
        self.assertEqual(Decimal("0.35"), PostgresPaperBroker.cumulative_fee(Decimal("100")))
        self.assertEqual(Decimal("0.35"), PostgresPaperBroker.cumulative_fee(Decimal("350")))
        self.assertEqual(Decimal("0.351"), PostgresPaperBroker.cumulative_fee(Decimal("351")))
        first = PostgresPaperBroker.cumulative_fee(Decimal("50"))
        second = PostgresPaperBroker.cumulative_fee(Decimal("100"))
        self.assertEqual(Decimal("0"), second - first)

    def test_market_direction_is_fail_closed_until_state_arrives(self):
        broker = PostgresPaperBroker(_Ledger())
        self.assertFalse(broker._direction_allowed("AAPL", "BUY"))
        broker.update_market_state("MARKET_OPEN", {"AAPL": "TRADING"}, {"AAPL": "BUY"})
        self.assertTrue(broker._direction_allowed("AAPL", "BUY"))
        self.assertFalse(broker._direction_allowed("AAPL", "SELL"))

    def test_checkpoint_json_quotes_non_finite_values_for_postgres(self):
        payload = _checkpoint_json({"price": float("nan"), "decimal": Decimal("NaN")})
        self.assertNotIn(": NaN", payload)
        self.assertIn('"NaN"', payload)
