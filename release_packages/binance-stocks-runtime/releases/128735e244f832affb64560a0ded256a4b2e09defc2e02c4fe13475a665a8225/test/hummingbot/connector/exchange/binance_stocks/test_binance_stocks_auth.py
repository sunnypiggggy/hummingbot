import hashlib
import hmac
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock
from urllib.parse import urlencode

from hummingbot.connector.exchange.binance_stocks.binance_stocks_auth import BinanceStocksAuth
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest


class BinanceStocksAuthTests(IsolatedAsyncioTestCase):
    async def test_hmac_auth_adds_recv_window_timestamp_signature_and_api_key(self):
        clock = MagicMock()
        clock.time.return_value = 1_700_000_000.123
        auth = BinanceStocksAuth("key", "secret", clock)
        request = RESTRequest(
            method=RESTMethod.GET,
            params={"symbol": "AAPL", "clientOrderId": "abc"},
            is_auth_required=True,
        )

        authenticated = await auth.rest_authenticate(request)

        expected_without_signature = {
            "symbol": "AAPL",
            "clientOrderId": "abc",
            "recvWindow": 5000,
            "timestamp": 1_700_000_000_123,
        }
        expected_signature = hmac.new(
            b"secret", urlencode(expected_without_signature).encode(), hashlib.sha256
        ).hexdigest()
        self.assertEqual(expected_signature, authenticated.params["signature"])
        self.assertEqual("key", authenticated.headers["X-MBX-APIKEY"])
