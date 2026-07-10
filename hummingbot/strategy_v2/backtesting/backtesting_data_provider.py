import logging
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from hummingbot import data_path
from hummingbot.client.config.config_helpers import get_connector_class
from hummingbot.client.settings import AllConnectorSettings, ConnectorType
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import LazyDict, PriceType
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HistoricalCandlesConfig
from hummingbot.data_feed.market_data_provider import MarketDataProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingDataProvider(MarketDataProvider):
    CONNECTOR_TYPES = [ConnectorType.CLOB_SPOT, ConnectorType.CLOB_PERP, ConnectorType.Exchange,
                       ConnectorType.Derivative]
    EXCLUDED_CONNECTORS = ["hyperliquid_perpetual", "dydx_perpetual", "cube", "vertex",
                           "coinbase_advanced_trade", "kraken", "dydx_v4_perpetual", "hitbtc",
                           "hyperliquid", "injective_v2_perpetual", "injective_v2"]
    CANDLES_CACHE_DIR = Path(data_path()) / "backtesting_candles"

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.start_time = None
        self.end_time = None
        self.prices = {}
        self._time = None
        self.trading_rules = {}
        self.conn_settings = AllConnectorSettings.get_connector_settings()
        self.connectors = LazyDict[str, Optional[ConnectorBase]](
            lambda name: self.get_connector(name) if (
                self.conn_settings[name].type in self.CONNECTOR_TYPES and
                name not in self.EXCLUDED_CONNECTORS and
                "testnet" not in name
            ) else None
        )

    def get_connector(self, connector_name: str):
        conn_setting = self.conn_settings.get(connector_name)
        if conn_setting is None:
            logger.error(f"Connector {connector_name} not found")
            raise ValueError(f"Connector {connector_name} not found")

        init_params = conn_setting.conn_init_parameters(
            trading_pairs=[],
            trading_required=False,
            api_keys=MarketDataProvider.get_connector_config_map(connector_name),
        )
        connector_class = get_connector_class(connector_name)
        connector = connector_class(**init_params)
        return connector

    def get_trading_rules(self, connector_name: str, trading_pair: str):
        """
        Retrieves the trading rules from the specified connector.
        :param connector_name: str
        :return: Trading rules.
        """
        return self.trading_rules[connector_name][trading_pair]

    def time(self):
        return self._time

    async def initialize_trading_rules(self, connector_name: str):
        if len(self.trading_rules.get(connector_name, {})) == 0:
            connector = self.connectors.get(connector_name)
            await connector._update_trading_rules()
            self.trading_rules[connector_name] = connector.trading_rules

    async def initialize_candles_feed(self, config: CandlesConfig):
        await self.get_candles_feed(config)

    def update_backtesting_time(self, start_time: int, end_time: int):
        self.start_time = start_time
        self.end_time = end_time
        self._time = start_time

    async def get_candles_feed(self, config: CandlesConfig):
        """
        Retrieves or creates and starts a candle feed based on the given configuration.
        If an existing feed has a higher or equal max_records, it is reused.
        :param config: CandlesConfig
        :return: Candle feed instance.
        """
        key = self._generate_candle_feed_key(config)
        existing_feed = self.candles_feeds.get(key, pd.DataFrame())
        # existing_feed = self.ensure_epoch_index(existing_feed)

        if not existing_feed.empty:
            existing_feed_start_time = existing_feed["timestamp"].min()
            existing_feed_end_time = existing_feed["timestamp"].max()
            if existing_feed_start_time <= self.start_time and existing_feed_end_time >= self.end_time:
                return existing_feed
        # Create a new feed or restart the existing one with updated max_records
        candle_feed = CandlesFactory.get_candle(config)
        candles_buffer = config.max_records * CandlesBase.interval_to_seconds[config.interval]
        requested_start_time = self.start_time - candles_buffer
        requested_end_time = self.end_time
        cached_candles_df = self._load_cached_candles(config)
        interval_seconds = CandlesBase.interval_to_seconds[config.interval]
        cached_end_time = self._round_timestamp_to_interval(requested_end_time, interval_seconds)
        if self._covers_time_range(cached_candles_df, requested_start_time, cached_end_time):
            logger.info(f"Using cached candles for {key}")
            candles_df = cached_candles_df[
                (cached_candles_df["timestamp"] >= requested_start_time)
                & (cached_candles_df["timestamp"] <= requested_end_time)
            ].copy()
            self.candles_feeds[key] = candles_df
            return candles_df

        candles_df = await self._fetch_missing_candles(
            candle_feed=candle_feed,
            config=config,
            cached_candles_df=cached_candles_df,
            requested_start_time=requested_start_time,
            requested_end_time=requested_end_time,
            interval_seconds=interval_seconds,
        )
        candles_df = self._merge_and_save_cached_candles(config, cached_candles_df, candles_df)
        candles_df = candles_df[
            (candles_df["timestamp"] >= requested_start_time)
            & (candles_df["timestamp"] <= requested_end_time)
        ].copy()
        # TODO: fix pandas-ta improper float index slicing to allow us to use float indexes
        # candles_df = self.ensure_epoch_index(candles_df)
        self.candles_feeds[key] = candles_df
        return candles_df

    @classmethod
    def _cache_file_path(cls, config: CandlesConfig) -> Path:
        safe_trading_pair = config.trading_pair.replace("/", "-").replace(":", "-")
        file_name = f"{config.connector}_{safe_trading_pair}_{config.interval}.csv"
        return cls.CANDLES_CACHE_DIR / file_name

    @classmethod
    def _load_cached_candles(cls, config: CandlesConfig) -> pd.DataFrame:
        cache_file = cls._cache_file_path(config)
        if not cache_file.exists():
            return pd.DataFrame()
        try:
            candles_df = pd.read_csv(cache_file)
            if candles_df.empty or "timestamp" not in candles_df.columns:
                return pd.DataFrame()
            candles_df.drop_duplicates(subset=["timestamp"], inplace=True)
            candles_df.sort_values(by="timestamp", inplace=True)
            candles_df.reset_index(drop=True, inplace=True)
            return candles_df
        except Exception as exception:
            logger.warning(f"Unable to load backtesting candles cache {cache_file}: {exception}")
            return pd.DataFrame()

    @staticmethod
    def _covers_time_range(candles_df: pd.DataFrame, start_time: int, end_time: int) -> bool:
        if candles_df.empty or "timestamp" not in candles_df.columns:
            return False
        return candles_df["timestamp"].min() <= start_time and candles_df["timestamp"].max() >= end_time

    @staticmethod
    def _round_timestamp_to_interval(timestamp: int, interval_seconds: int) -> int:
        return timestamp - (timestamp % interval_seconds)

    async def _fetch_missing_candles(
        self,
        candle_feed: CandlesBase,
        config: CandlesConfig,
        cached_candles_df: pd.DataFrame,
        requested_start_time: int,
        requested_end_time: int,
        interval_seconds: int,
    ) -> pd.DataFrame:
        if cached_candles_df.empty:
            logger.info(
                f"Fetching candles for {config.connector}_{config.trading_pair}_{config.interval}: "
                f"{requested_start_time} -> {requested_end_time}"
            )
            return await self._fetch_candles_range(candle_feed, config, requested_start_time, requested_end_time)

        fetched_dfs = []
        cached_start_time = int(cached_candles_df["timestamp"].min())
        cached_end_time = int(cached_candles_df["timestamp"].max())
        if requested_start_time < cached_start_time:
            logger.info(
                f"Fetching missing leading candles for {config.connector}_{config.trading_pair}_{config.interval}: "
                f"{requested_start_time} -> {cached_start_time - interval_seconds}"
            )
            fetched_dfs.append(
                await self._fetch_candles_range(
                    candle_feed, config, requested_start_time, cached_start_time - interval_seconds
                )
            )
        if self._round_timestamp_to_interval(requested_end_time, interval_seconds) > cached_end_time:
            logger.info(
                f"Fetching missing trailing candles for {config.connector}_{config.trading_pair}_{config.interval}: "
                f"{cached_end_time + interval_seconds} -> {requested_end_time}"
            )
            fetched_dfs.append(
                await self._fetch_candles_range(
                    candle_feed, config, cached_end_time + interval_seconds, requested_end_time
                )
            )
        return pd.concat(fetched_dfs, ignore_index=True) if fetched_dfs else pd.DataFrame()

    @staticmethod
    async def _fetch_candles_range(
        candle_feed: CandlesBase,
        config: CandlesConfig,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        if start_time > end_time:
            return pd.DataFrame()
        return await candle_feed.get_historical_candles(config=HistoricalCandlesConfig(
            connector_name=config.connector,
            trading_pair=config.trading_pair,
            interval=config.interval,
            start_time=start_time,
            end_time=end_time,
        ))

    @classmethod
    def _merge_and_save_cached_candles(
        cls,
        config: CandlesConfig,
        cached_candles_df: pd.DataFrame,
        fetched_candles_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if fetched_candles_df.empty:
            return cached_candles_df
        candles_df = pd.concat([cached_candles_df, fetched_candles_df], ignore_index=True)
        candles_df.drop_duplicates(subset=["timestamp"], inplace=True)
        candles_df.sort_values(by="timestamp", inplace=True)
        candles_df.reset_index(drop=True, inplace=True)
        cache_file = cls._cache_file_path(config)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        candles_df.to_csv(cache_file, index=False)
        return candles_df

    def get_candles_df(self, connector_name: str, trading_pair: str, interval: str, max_records: int = 500):
        """
        Retrieves the candles for a trading pair from the specified connector.
        :param connector_name: str
        :param trading_pair: str
        :param interval: str
        :param max_records: int
        :return: Candles dataframe.
        """
        candles_df = self.candles_feeds.get(f"{connector_name}_{trading_pair}_{interval}")
        return candles_df[(candles_df["timestamp"] >= self.start_time) & (candles_df["timestamp"] <= self.end_time)]

    def get_price_by_type(self, connector_name: str, trading_pair: str, price_type: PriceType):
        """
        Retrieves the price for a trading pair from the specified connector based on the price type.
        :param connector_name: str
        :param trading_pair: str
        :param price_type: PriceType
        :return: Price.
        """
        return self.prices.get(f"{connector_name}_{trading_pair}", Decimal("1"))

    def quantize_order_amount(self, connector_name: str, trading_pair: str, amount: Decimal):
        """
        Quantizes the order amount based on the trading pair's minimum order size.
        :param connector_name: str
        :param trading_pair: str
        :param amount: Decimal
        :return: Quantized amount.
        """
        trading_rules = self.get_trading_rules(connector_name, trading_pair)
        order_size_quantum = trading_rules.min_base_amount_increment
        return (amount // order_size_quantum) * order_size_quantum

    def quantize_order_price(self, connector_name: str, trading_pair: str, price: Decimal):
        """
        Quantizes the order price based on the trading pair's minimum price increment.
        :param connector_name: str
        :param trading_pair: str
        :param price: Decimal
        :return: Quantized price.
        """
        trading_rules = self.get_trading_rules(connector_name, trading_pair)
        price_quantum = trading_rules.min_price_increment
        return (price // price_quantum) * price_quantum

    # TODO: enable copy-on-write and allow specification of inplace
    @staticmethod
    def ensure_epoch_index(df: pd.DataFrame, timestamp_column: str = "timestamp",
                           keep_original: bool = True, index_name: str = "epoch_seconds") -> pd.DataFrame:
        """Ensures DataFrame has numeric monotonic increasing timestamp index in seconds since epoch."""
        # Skip if already numeric index but not RangeIndex as that generally means the index was dropped
        if df.index.name == index_name or df.empty:
            return df

        # DatetimeIndex → convert to seconds
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.map(pd.Timestamp.timestamp)
        # Has timestamp column → use as index
        elif timestamp_column in df.columns:
            df = df.set_index(timestamp_column, drop=not keep_original)
            # Convert non-numeric indices to seconds
            if not pd.api.types.is_numeric_dtype(df.index):
                df.index = pd.to_datetime(df.index).map(pd.Timestamp.timestamp)
        else:
            raise ValueError(f"Cannot create timestamp index: no '{timestamp_column}' column found and index isn't convertible")
        df.sort_index(inplace=True)
        df.index.name = index_name
        return df
