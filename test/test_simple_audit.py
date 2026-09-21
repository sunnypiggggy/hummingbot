import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, Mock

from live_guard.simple_audit import (ASSETS, ROBOTS, SimpleAudit, balance_snapshot,
                                    collect, comparison_value, history_rows, render, segments)
from management_bot import simple_audit_view

NOW = 1789873200


def inputs(now=NOW):
    balance = balance_snapshot({a: {"free": "1", "locked": "1", "total": "2"} for a in ASSETS}, "test-account", now)
    grid = {"bots": {"grid-live-fdusd-400": {"latest": {
        "observed_at": now, "equity": "40", "pairs": {
            p: {"owned_base": ".5", "quote_balance": "1", "equity": "10"}
            for s,p in ROBOTS if s == "grid"}}}}}
    dca = {"simple_audit_balance": balance, "bots": {
        "dca-live-"+p.replace("-", "").lower()+"-200": {"latest": {
            "owned_base": ".5", "quote_balance": "1", "observed_at": now}}
        for s,p in ROBOTS if s == "dca"}}
    inventory = {"account_fingerprint": "test-account", "sources_healthy": True,
                 "generated_at": now, "assets": {a: {"owned_total": "1", "exchange": {"total": "2"}} for a in ("BTC", "ETH")}}
    quotes = {"observed_at": now, "prices": {"BTC": "100", "ETH": "10", "FDUSD": ".9", "USDT": "1"}}
    return balance, grid, dca, inventory, quotes


class SimpleAuditTests(TestCase):
    def test_totals_free_locked_reserve_and_non_parity(self):
        row = collect(*inputs(), NOW)
        self.assertEqual("223.8", row["account_total"])
        self.assertEqual("20", row["reserve"])
        self.assertEqual("131.8", row["robot_total"])
        self.assertEqual("92.0", row["difference"])
        self.assertEqual([], row["issues"])

    def test_invalid_account_price_or_stale_never_zero(self):
        for mutator in (
            lambda x: x[0].update(observed_at=NOW-61),
            lambda x: x[4]["prices"].update(FDUSD="NaN"),
            lambda x: x[0]["assets"]["BTC"].update(total="8"),
            lambda x: x[4].update(observed_at=NOW-61),
        ):
            data = inputs();mutator(data)
            row = collect(*data, NOW)
            self.assertIsNone(row["account_total"])
            self.assertIsNone(row["difference"])

    def test_missing_negative_stale_or_mixed_ownership_blocks_comparison(self):
        for mutator in (
            lambda x: x[3].update(account_fingerprint="other"),
            lambda x: x[1]["bots"]["grid-live-fdusd-400"]["latest"]["pairs"]["BTC-FDUSD"].update(owned_base="-1"),
            lambda x: x[1]["bots"]["grid-live-fdusd-400"]["latest"].update(observed_at=NOW-61),
            lambda x: x[3]["assets"]["BTC"].update(owned_total="2"),
            lambda x: x[3]["assets"]["BTC"]["exchange"].update(total="3"),
        ):
            data = inputs();mutator(data)
            row = collect(*data, NOW)
            self.assertIsNotNone(row["account_total"])
            self.assertIsNone(row["robot_total"])
            self.assertIsNone(row["difference"])
            self.assertTrue(row["issues"])

    def test_base_deficit_not_hidden_by_positive_quote_surplus(self):
        balance,g,d,i,q=inputs()
        balance['assets']['ETH']={'free':'.9','locked':'0','total':'.9'}
        i['assets']['ETH']['exchange']['total']='.9'
        row=collect(balance,g,d,i,q,NOW)
        self.assertIsNone(row['difference'])
        self.assertTrue(any('ETH账面归属超出' in s for s in row['issues']))
        self.assertIsNone(comparison_value(row))
        row.update(issues=[],robot_total='100',difference='100')
        self.assertIsNone(comparison_value(row))

    def test_comparison_uses_same_timestamp_currency_and_does_not_fake_history(self):
        row=collect(*inputs(),NOW)
        self.assertAlmostEqual(131.8,comparison_value(row))
        row['issues'].append('过期')
        self.assertIsNone(comparison_value(row))

    @patch("live_guard.telegram_notifications.TelegramOutbox.enqueue")
    def test_hourly_restart_dedup_failed_slot_retry_and_retention(self, enqueue):
        with TemporaryDirectory() as raw:
            audit=SimpleAudit(raw)
            history=sqlite3.connect(":memory:")
            history.execute("CREATE TABLE profit_snapshot(strategy,pair,observed_at,equity)")
            def cycle(t, fail=False):
                _,g,d,i,q=inputs(t)
                if fail:q["prices"]["BTC"]="0"
                audit.cycle(grid=g,dca=d,inventory=i,quotes=q,history_connection=history,api_base="unused",now=t)
            cycle(NOW, True)
            self.assertEqual(0,audit.db.execute("SELECT count(*) FROM hours").fetchone()[0])
            cycle(NOW)
            first=(audit.root/"summary.json").read_bytes()
            audit.db.close();audit=SimpleAudit(raw)
            cycle(NOW+1)
            self.assertEqual(first,(audit.root/"summary.json").read_bytes())
            cycle(NOW+3*3600)
            self.assertEqual(2,audit.db.execute("SELECT count(*) FROM hours").fetchone()[0])
            from PIL import Image
            with Image.open(audit.root/"robots.png") as im:self.assertEqual((1440,3200),im.size)
            with Image.open(audit.root/"account.png") as im:self.assertEqual((1440,1800),im.size)
            page,_=simple_audit_view.page(raw,now=NOW+3*3600)
            self.assertIn("账户总额", page)
            self.assertNotIn("全部可信", page)
            self.assertEqual("robots.png",simple_audit_view.chart(raw,"robots").name)
            audit.db.close();history.close()
            enqueue.assert_not_called()

    def test_gaps_nulls_and_original_history_not_mixed(self):
        db=sqlite3.connect(":memory:");db.execute("CREATE TABLE profit_snapshot(strategy,pair,observed_at,equity)")
        db.executemany("INSERT INTO profit_snapshot VALUES(?,?,?,?)",[
            ("grid","BTC-FDUSD",NOW-60,200),("grid@epoch","BTC-FDUSD",NOW,999),
            ("grid","BTC-FDUSD",NOW,201)])
        rows=history_rows(db,NOW)
        self.assertEqual([200,201],[v for _,v in rows['grid:BTC-FDUSD']])
        self.assertEqual(2,len(segments([(NOW-180,NOW),(NOW-120,None),(NOW,201)],NOW)))
        self.assertEqual(2,len(segments([(NOW-7200,200),(NOW,201)],NOW,hourly=True)))
        db.close()

    def test_attachment_path_hash_and_stale_display(self):
        with TemporaryDirectory() as raw:
            folder=Path(raw)/"simple_audit";folder.mkdir()
            data=collect(*inputs(),NOW)
            data['charts']={'account':{'file':'../escape.png','sha256':'bad'}}
            (folder/'summary.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):simple_audit_view.chart(raw,'account')
            text,_=simple_audit_view.page(raw,now=NOW+7200)
            self.assertIn('结果已过期',text)
            self.assertNotIn('账户减机器人',text)

    def test_menu_last_and_navigation_readonly(self):
        from management_bot.app import HOME_ROWS
        self.assertEqual([('🔎 简单稽核','m:simple_audit')],HOME_ROWS[-1])
        callbacks=[c for row in simple_audit_view.ROWS for _,c in row]
        self.assertEqual(['sa:account','sa:robots','m:simple_audit','m:home'],callbacks)

    def test_quote_client_only_public_read(self):
        from live_guard.simple_audit import quote_snapshot
        response=Mock()
        response.json.return_value=[{'symbol':a+'USDT','bidPrice':'2'} for a in ASSETS[:-1]]
        with patch('requests.get',return_value=response) as get:
            result=quote_snapshot('https://example.invalid')
        self.assertEqual('1',result['prices']['USDT'])
        self.assertEqual('https://example.invalid/api/v3/ticker/bookTicker',get.call_args.args[0])
        self.assertNotIn('headers',get.call_args.kwargs)
        self.assertEqual('["BTCUSDT","ETHUSDT","FDUSDUSDT"]',get.call_args.kwargs['params']['symbols'])

    def test_callback_sends_one_picture_and_stays_on_page(self):
        from test.test_trading_management_bot import TelegramFlowTests
        with TemporaryDirectory() as raw:
            bot=TelegramFlowTests()._bot(Path(raw))
            try:
                bot.hummingbot=Mock(side_effect=AssertionError('no control calls'))
                callback={'update_id':851,'callback_query':{'id':'test','from':{'id':7},
                    'message':{'message_id':11,'chat':{'id':7,'type':'private'}},'data':'sa:robots'}}
                with patch('management_bot.simple_audit_view.chart',return_value=Path(raw)/'robots.png'), patch(
                        'management_bot.simple_audit_view.page',return_value=('报告',simple_audit_view.ROWS)):
                    bot.handle_update(callback)
                    bot.handle_update(callback)
                pictures=[s for s in bot.telegram.sent if isinstance(s[2],str) and s[2].endswith('.png')]
                self.assertEqual(1,len(pictures))
                self.assertEqual('',pictures[0][1])
                self.assertEqual(simple_audit_view.ROWS,bot.telegram.edited[-1][3])
            finally:bot.store.close()
