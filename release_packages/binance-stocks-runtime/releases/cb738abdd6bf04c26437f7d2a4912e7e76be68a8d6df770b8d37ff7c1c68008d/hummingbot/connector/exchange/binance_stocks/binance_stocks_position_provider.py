from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional, Protocol


@dataclass(frozen=True)
class EquityPosition:
    total: Decimal
    available: Decimal


@dataclass(frozen=True)
class EquityAccountSnapshot:
    positions: Dict[str, EquityPosition] = field(default_factory=dict)
    quote_total: Decimal = Decimal("0")
    quote_available: Decimal = Decimal("0")
    source: str = "unknown"
    timestamp: float = 0.0


class EquityPositionProvider(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def get_snapshot(self) -> Optional[EquityAccountSnapshot]:
        raise NotImplementedError


class ManagedLedgerStore(Protocol):
    """Storage boundary used by the Stocks runtime.

    The store intentionally exposes only inventory owned by this runtime.  It is
    not an account-wide position service and must never import unrelated Binance
    equity orders into the managed balance.
    """

    async def managed_positions(self) -> Dict[str, EquityPosition]:
        ...

    async def quote_balances(self) -> tuple[Decimal, Decimal]:
        ...

    async def is_fresh(self) -> bool:
        ...

    async def register_order(self, **order) -> None:
        ...

    async def bind_exchange_order(self, client_order_id: str, exchange_order_id: str) -> None:
        ...

    async def record_cumulative_fill(self, **fill) -> None:
        ...


class ManagedLedgerEquityPositionProvider(EquityPositionProvider):
    """Expose the runtime-owned ledger through Hummingbot's balance interface.

    Binance Stocks does not currently publish an account-wide equity-position
    snapshot.  This provider therefore reports only fills whose deterministic
    client order IDs belong to the dedicated Stocks runtime.  Callers must keep
    ``external_positions_unknown`` visible in their API and operational reports.
    """

    def __init__(self, store: ManagedLedgerStore, source: str = "managed_ledger_non_authoritative"):
        self._store = store
        self._source = source

    @property
    def available(self) -> bool:
        return True

    async def get_snapshot(self) -> Optional[EquityAccountSnapshot]:
        if not await self._store.is_fresh():
            return None
        positions = await self._store.managed_positions()
        quote_total, quote_available = await self._store.quote_balances()
        import time
        return EquityAccountSnapshot(
            positions=positions,
            quote_total=quote_total,
            quote_available=quote_available,
            source=self._source,
            timestamp=time.time(),
        )

    async def register_order(self, **order) -> None:
        await self._store.register_order(**order)

    async def bind_exchange_order(self, client_order_id: str, exchange_order_id: str) -> None:
        await self._store.bind_exchange_order(client_order_id, exchange_order_id)

    async def record_cumulative_fill(self, **fill) -> None:
        await self._store.record_cumulative_fill(**fill)


class UnavailableEquityPositionProvider(EquityPositionProvider):
    """Default until Binance publishes an authoritative stock position API."""

    @property
    def available(self) -> bool:
        return False

    async def get_snapshot(self) -> Optional[EquityAccountSnapshot]:
        return None
