from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketCalendarState:
    phase: str
    trading_date: str
    observed_at: datetime
    valid_until: datetime
    source: str = "XNYS"


def xnys_market_state(now: datetime | None = None) -> MarketCalendarState:
    """Return the XNYS session phase including extended-hours boundaries.

    exchange_calendars owns holidays, DST and early closes.  The runtime image
    pins it explicitly so a missing calendar is an infrastructure condition,
    never a guessed weekday session.
    """
    import pandas as pd

    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local = instant.astimezone(NEW_YORK)
    label = pd.Timestamp(local.date())
    calendar = _xnys_calendar(local.year)
    is_session = bool(calendar.is_session(label))
    if is_session:
        market_open = calendar.session_open(label).to_pydatetime().astimezone(timezone.utc)
        market_close = calendar.session_close(label).to_pydatetime().astimezone(timezone.utc)
        pre_open = datetime.combine(local.date(), time(4, 0), NEW_YORK).astimezone(timezone.utc)
        post_close = datetime.combine(local.date(), time(20, 0), NEW_YORK).astimezone(timezone.utc)
        if instant < pre_open:
            phase, valid_until = "MARKET_CLOSED", pre_open
        elif instant < market_open:
            phase, valid_until = "PRE_MARKET", market_open
        elif instant < market_close:
            phase, valid_until = "MARKET_OPEN", market_close
        elif instant < post_close:
            phase, valid_until = "POST_MARKET", post_close
        else:
            phase, valid_until = "MARKET_CLOSED", _next_pre_open(calendar, label, after_session=True)
    else:
        phase, valid_until = "MARKET_CLOSED", _next_pre_open(calendar, label)
    return MarketCalendarState(
        phase=phase,
        trading_date=str(local.date()),
        observed_at=instant,
        valid_until=valid_until,
    )


def _next_pre_open(calendar, label, after_session: bool = False) -> datetime:
    next_label = calendar.next_session(label) if after_session else calendar.date_to_session(label, direction="next")
    next_date = next_label.date()
    return datetime.combine(next_date, time(4, 0), NEW_YORK).astimezone(timezone.utc)


@lru_cache(maxsize=4)
def _xnys_calendar(year: int):
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar

    return XNYSExchangeCalendar(start=f"{year - 1}-01-01", end=f"{year + 1}-12-31")


def phases_conflict(binance_phase: str, xnys_phase: str) -> bool:
    closed = {"MARKET_CLOSED", "OVERNIGHT"}
    left, right = str(binance_phase).upper(), str(xnys_phase).upper()
    return left != right and not ({left, right} <= closed)
