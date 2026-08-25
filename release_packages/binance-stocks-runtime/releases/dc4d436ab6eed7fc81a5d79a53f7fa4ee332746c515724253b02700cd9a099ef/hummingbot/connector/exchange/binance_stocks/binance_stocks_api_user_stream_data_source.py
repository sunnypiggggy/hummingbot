import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant

from . import binance_stocks_constants as CONSTANTS, binance_stocks_web_utils as web_utils
from .binance_stocks_auth import BinanceStocksAuth
from .binance_stocks_utils import extract_payload

if TYPE_CHECKING:
    from .binance_stocks_exchange import BinanceStocksExchange


class BinanceStocksAPIUserStreamDataSource(UserStreamTrackerDataSource):
    def __init__(
        self,
        auth: BinanceStocksAuth,
        trading_pairs: List[str],
        connector: "BinanceStocksExchange",
        api_factory: WebAssistantsFactory,
        domain: str = CONSTANTS.DEFAULT_DOMAIN,
    ):
        super().__init__()
        self._auth = auth
        self._connector = connector
        self._api_factory = api_factory
        self._domain = domain
        self._current_listen_key: Optional[str] = None
        self._listen_key_initialized_event = asyncio.Event()
        self._manage_listen_key_task: Optional[asyncio.Task] = None
        self._last_listen_key_refresh = 0.0

    async def _get_listen_key(self, listen_key: Optional[str] = None) -> str:
        assistant = await self._api_factory.get_rest_assistant()
        data = {"listenKey": listen_key} if listen_key else {}
        response = await assistant.execute_request(
            url=web_utils.private_rest_url(CONSTANTS.LISTEN_KEY_PATH_URL, self._domain),
            data=data,
            method=RESTMethod.POST,
            is_auth_required=True,
            throttler_limit_id=CONSTANTS.LISTEN_KEY_PATH_URL,
        )
        payload = extract_payload(response)
        key = payload.get("listenKey") if isinstance(payload, dict) else None
        if not key:
            raise IOError(f"Binance Stocks listenKey response is invalid: {response}")
        return str(key)

    async def _manage_listen_key_loop(self):
        try:
            while True:
                if self._current_listen_key is None:
                    self._current_listen_key = await self._get_listen_key()
                    self._last_listen_key_refresh = self._time()
                    self._listen_key_initialized_event.set()
                    self._connector.set_account_authorized(True)
                elif self._time() - self._last_listen_key_refresh >= CONSTANTS.LISTEN_KEY_KEEP_ALIVE_SECONDS:
                    renewed = await self._get_listen_key(self._current_listen_key)
                    if renewed != self._current_listen_key:
                        raise IOError("Binance Stocks listenKey changed during renewal")
                    self._last_listen_key_refresh = self._time()
                await self._sleep(5.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._connector.set_account_authorized(False)
            self._current_listen_key = None
            self._listen_key_initialized_event.clear()
            raise

    async def _ensure_listen_key_task(self):
        if self._manage_listen_key_task is None or self._manage_listen_key_task.done():
            self._manage_listen_key_task = safe_ensure_future(self._manage_listen_key_loop())

    async def _connected_websocket_assistant(self) -> WSAssistant:
        await self._ensure_listen_key_task()
        await self._listen_key_initialized_event.wait()
        websocket = await self._api_factory.get_ws_assistant()
        await websocket.connect(
            ws_url=f"{CONSTANTS.WS_URL}/ws/{self._current_listen_key}@orderReport",
            ping_timeout=CONSTANTS.WS_HEARTBEAT_SECONDS,
        )
        return websocket

    async def _subscribe_channels(self, websocket_assistant: WSAssistant):
        self.logger().info("Connected to Binance Stocks orderReport stream.")

    async def _process_event_message(self, event_message: Dict[str, Any], queue: asyncio.Queue):
        if not isinstance(event_message, dict):
            return
        payload = event_message.get("data", event_message)
        if payload.get("e") == "orderReport" or payload.get("eventType") == "orderReport":
            queue.put_nowait(payload)

    async def _on_user_stream_interruption(self, websocket_assistant: Optional[WSAssistant]):
        if websocket_assistant is not None:
            await websocket_assistant.disconnect()

    async def stop(self):
        if self._manage_listen_key_task is not None:
            self._manage_listen_key_task.cancel()
            try:
                await self._manage_listen_key_task
            except asyncio.CancelledError:
                pass
            self._manage_listen_key_task = None
        self._current_listen_key = None
        self._listen_key_initialized_event.clear()
        await super().stop()
