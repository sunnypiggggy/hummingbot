import hashlib
import hmac
import json
from collections import OrderedDict
from typing import Any, Dict
from urllib.parse import urlencode

from hummingbot.connector.time_synchronizer import TimeSynchronizer
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest, WSRequest


class BinanceStocksAuth(AuthBase):
    """HMAC-SHA256 authentication for Binance SAPI equity endpoints."""

    def __init__(self, api_key: str, secret_key: str, time_provider: TimeSynchronizer):
        self.api_key = api_key
        self.secret_key = secret_key
        self.time_provider = time_provider

    async def rest_authenticate(self, request: RESTRequest) -> RESTRequest:
        if request.method == RESTMethod.POST:
            raw_data = json.loads(request.data) if isinstance(request.data, str) else (request.data or {})
            # Stocks examples sign the form/query representation. Keep the
            # transmitted body byte-for-byte equivalent to the signed payload.
            request.data = urlencode(self.add_auth_to_params(raw_data))
        else:
            request.params = self.add_auth_to_params(request.params or {})
        headers = dict(request.headers or {})
        headers["X-MBX-APIKEY"] = self.api_key
        if request.method == RESTMethod.POST:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request.headers = headers
        return request

    async def ws_authenticate(self, request: WSRequest) -> WSRequest:
        return request

    def add_auth_to_params(self, params: Dict[str, Any]) -> OrderedDict:
        signed = OrderedDict(params or {})
        signed.setdefault("recvWindow", 5000)
        signed["timestamp"] = int(self.time_provider.time() * 1e3)
        signed["signature"] = self._generate_signature(signed)
        return signed

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        encoded = urlencode(params)
        return hmac.new(self.secret_key.encode(), encoded.encode(), hashlib.sha256).hexdigest()
