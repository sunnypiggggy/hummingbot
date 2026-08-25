from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, Optional

from stocks_runtime.ledger import LedgerLimitExceeded, PostgresManagedLedger
from stocks_runtime.executor_config import order_type_name, trade_side_name


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SAFE_TICKER = re.compile(r"^[A-Z][A-Z0-9.]{0,14}$")


class PolicyViolation(LedgerLimitExceeded):
    def __init__(self, code: str, message: str, **context: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context

    def payload(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.context}


class StocksExecutorPolicy:
    def __init__(self, ledger: PostgresManagedLedger, mode: str, live_authorized: bool):
        self.ledger = ledger
        self.mode = mode
        self.live_authorized = live_authorized
        self.symbols: set[str] = set()
        self.prices: Dict[str, Decimal] = {}
        self.trading_date: Optional[str] = None

    def update_market(self, symbols: set[str], prices: Dict[str, Decimal], trading_date: Optional[str] = None) -> None:
        self.symbols = symbols
        self.prices = prices
        self.trading_date = trading_date

    @staticmethod
    def _enum(value: Any) -> str:
        return str(getattr(value, "value", value)).upper()

    async def validate_and_reserve(
        self, executor_config: Dict[str, Any], account_name: Optional[str], controller_id: Optional[str],
        *, schedule: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self._validate(
            executor_config, account_name, controller_id, reserve=True, schedule=schedule
        )

    async def preview(
        self, executor_config: Dict[str, Any], account_name: Optional[str], controller_id: Optional[str]
    ) -> Dict[str, Any]:
        """Run the authoritative policy without reserving funds or inventory."""
        return await self._validate(executor_config, account_name, controller_id, reserve=False)

    async def revalidate_reserved(
        self, executor_config: Dict[str, Any], account_name: Optional[str], controller_id: Optional[str]
    ) -> Dict[str, Any]:
        """Repeat all dynamic checks without counting or replacing this request's reservation."""
        return await self._validate(
            executor_config,
            account_name,
            controller_id,
            reserve=False,
            exclude_executor_id=str(executor_config.get("id", "")),
        )

    async def _validate(
        self,
        executor_config: Dict[str, Any],
        account_name: Optional[str],
        controller_id: Optional[str],
        *,
        reserve: bool,
        schedule: Optional[Dict[str, Any]] = None,
        exclude_executor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        config = dict(executor_config)
        executor_id = str(config.get("id", ""))
        if not SAFE_ID.fullmatch(executor_id):
            raise ValueError("executor_config.id must be a stable 8-64 character identifier")
        executor_type = str(config.get("type", ""))
        if executor_type not in {"order_executor", "position_executor"}:
            raise ValueError("only order_executor and position_executor are allowed")
        if account_name not in {None, "stocks_managed"}:
            raise ValueError("the Stocks runtime only accepts account stocks_managed")
        if str(config.get("connector_name", "")) != "binance_stocks":
            raise ValueError("the Stocks runtime only accepts connector binance_stocks")
        pair = str(config.get("trading_pair", "")).upper()
        parts = pair.rsplit("-", 1)
        if len(parts) != 2 or parts[1] != "USDC" or not SAFE_TICKER.fullmatch(parts[0]):
            raise ValueError("trading_pair must be a direct TICKER-USDC equity pair")
        symbol = parts[0]
        if symbol not in self.symbols:
            raise ValueError(f"{symbol} is not present in the current direct-equity exchangeInfo")
        side = trade_side_name(config.get("side"))
        whitelist = await self.ledger.whitelist_entry(symbol)
        if side == "BUY" and (not whitelist or not bool(whitelist.get("enabled"))):
            raise LedgerLimitExceeded(f"{symbol} is not enabled in the operator whitelist")
        amount = Decimal(str(config.get("amount", "0")))
        if amount <= 0:
            raise ValueError("amount must be positive base shares")
        price = Decimal(str(config.get("price") or config.get("entry_price") or self.prices.get(symbol, 0)))
        if price <= 0:
            raise ValueError(f"no fresh executable price is available for {symbol}")
        if executor_type == "order_executor":
            strategy = order_type_name(config.get("execution_strategy"))
            if strategy not in {"LIMIT", "MARKET"}:
                raise ValueError("OrderExecutor supports LIMIT or MARKET only")
            if strategy == "LIMIT" and config.get("price") is None:
                raise ValueError("LIMIT requires price")
        else:
            if side != "BUY":
                raise ValueError("PositionExecutor is long-only")
            barriers = config.get("triple_barrier_config") or {}
            if Decimal(str(barriers.get("stop_loss") or 0)) <= 0:
                raise ValueError("PositionExecutor requires a positive stop_loss")
            if int(barriers.get("time_limit") or 0) <= 0:
                raise ValueError("PositionExecutor requires a positive time_limit")
            for field in ("open_order_type", "take_profit_order_type"):
                value = order_type_name(barriers.get(field, "LIMIT"))
                if value not in {"LIMIT", "MARKET"}:
                    raise ValueError(f"{field} supports LIMIT or MARKET only")
        limits = await self.ledger.active_limits()
        current_pnl = await self.ledger.managed_pnl(self.prices)
        if self.trading_date:
            await self.ledger.set_trading_date(self.trading_date, current_pnl)
        daily_pnl = await self.ledger.daily_pnl(current_pnl)
        if side == "BUY" and daily_pnl <= -limits.daily_loss_limit:
            raise PolicyViolation(
                "日亏损限额已触发", "当前交易日净亏损已达到运行限额，暂不允许新增BUY",
                requested="BUY", current=str(-daily_pnl), available="0",
                limit=str(limits.daily_loss_limit),
            )
        estimated_notional = amount * price
        if estimated_notional > limits.max_order_notional:
            raise PolicyViolation(
                "超过单笔本金限额", "请求本金超过当前单笔运行限额",
                requested=str(estimated_notional), current="0",
                available=str(limits.max_order_notional), limit=str(limits.max_order_notional),
            )
        fee_reserve = Decimal("0.35") if estimated_notional <= 350 else estimated_notional * Decimal("0.001")
        usage = await self.ledger.capital_usage(self.prices, exclude_executor_id=exclude_executor_id)
        exposure = Decimal(usage["principal_and_mtm"])
        if side == "BUY" and exposure + estimated_notional > limits.max_managed_exposure:
            raise PolicyViolation(
                "超过总本金限额", "持仓MTM与未成交BUY本金合计将超过总运行限额",
                requested=str(estimated_notional), current=str(exposure),
                available=str(max(Decimal("0"), limits.max_managed_exposure - exposure)),
                limit=str(limits.max_managed_exposure),
            )
        symbol_limit = min(
            limits.max_symbol_exposure,
            Decimal(str(whitelist.get("max_position_notional"))) if whitelist else limits.max_symbol_exposure,
        )
        symbol_exposure = Decimal(usage["by_symbol"].get(symbol, 0))
        if side == "BUY" and symbol_exposure + estimated_notional > symbol_limit:
            raise PolicyViolation(
                "超过单股票本金限额", f"{symbol}持仓MTM与未成交BUY本金将超过单股票限额",
                requested=str(estimated_notional), current=str(symbol_exposure),
                available=str(max(Decimal("0"), symbol_limit - symbol_exposure)), limit=str(symbol_limit),
            )
        capital_with_fees = exposure + estimated_notional + Decimal(usage["fee_reserve"]) + fee_reserve
        trusted_equity = Decimal(usage["quote_total"])
        if side == "BUY" and capital_with_fees > trusted_equity:
            raise PolicyViolation(
                "资金不足", "总风险本金与费用预留超过可信PAPER权益",
                requested=str(estimated_notional + fee_reserve),
                current=str(exposure + Decimal(usage["fee_reserve"])),
                available=str(max(Decimal("0"), trusted_equity - exposure - Decimal(usage["fee_reserve"]))),
                limit=str(trusted_equity),
            )
        source_owner = (
            str(config.get("managed_source_owner") or "unassigned")
            if executor_type == "order_executor" and side == "SELL" else None
        )
        if side == "SELL" and source_owner:
            available = await self.ledger.managed_available(source_owner, symbol)
            if available < amount:
                raise LedgerLimitExceeded(f"managed {symbol} inventory is insufficient for SELL")
        common = {
            "runtime_mode": self.mode,
            "live_authorized": self.live_authorized,
            # PAPER submits exclusively to the local persistent paper venue.
            # SHADOW remains validation-only; LIVE still requires authorization.
            "would_submit": self.mode == "PAPER" or (self.mode == "LIVE" and self.live_authorized),
            "controller_id": controller_id or "stocks-runtime",
            "estimated_notional": str(estimated_notional),
            "fee_reserve": str(fee_reserve),
            "symbol_limit": str(symbol_limit),
            "total_limit": str(limits.max_managed_exposure),
            "current_total_principal": str(exposure),
            "current_symbol_principal": str(symbol_exposure),
            "current_fee_reserve": str(usage["fee_reserve"]),
            "available_cash": str(usage["quote_available"]),
            "trusted_equity": str(usage["quote_total"]),
            "preflight_only": not reserve,
        }
        if not reserve:
            return {"allowed": True, "executor_id": executor_id, **common}
        reserve_args = dict(
            executor_id=executor_id,
            executor_type=executor_type,
            symbol=symbol,
            side=side,
            requested_base=amount,
            estimated_notional=estimated_notional,
            fee_reserve=fee_reserve,
            config=config,
            source_owner=source_owner,
            positions_mtm=Decimal(usage["positions_mtm"]),
            symbol_position_mtm=Decimal(usage["positions_by_symbol"].get(symbol, 0)),
            max_symbol_limit=symbol_limit,
        )
        if schedule is not None:
            reserve_args["schedule"] = schedule
        result = await self.ledger.reserve_intent(**reserve_args)
        result.update({"allowed": True, **common})
        return result
