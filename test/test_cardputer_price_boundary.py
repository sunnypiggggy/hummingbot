"""Regression coverage for REST open-time filtering versus curve close times."""
import json
from unittest.mock import Mock

import pytest

from cardputer_monitor.collector import ROBOTS
from cardputer_monitor.history import PRICE_URL, PublicKlinePrices, sampled_curve


def candle(opened, price):
    return [opened * 1000, "100", "101", "99", str(price), "10",
            (opened + 300) * 1000 - 1, "1000", 5, "5", "500", "0"]


class OpenTimeSession:
    """A local fake that actually applies Binance's opening-time query bounds."""

    def __init__(self, rows):
        self.rows = rows
        self.headers = {}
        self.cookies = Mock()
        self.calls = []
        self.responses = []

    def get(self, url, **kwargs):
        params = kwargs["params"]
        rows = [row for row in self.rows
                if params["startTime"] <= row[0] <= params["endTime"]][:params["limit"]]
        body = json.dumps(rows).encode()
        response = Mock(status_code=200)
        response.iter_content.return_value = [body[i:i + 8192] for i in range(0, len(body), 8192)]
        self.calls.append((url, kwargs, rows))
        self.responses.append(response)
        return response


def market_rows(start, end):
    first = (start // 300 - 3) * 300
    last = (end // 300 + 1) * 300
    return [candle(opened, 100 + index)
            for index, opened in enumerate(range(first, last + 1, 300))]


@pytest.mark.parametrize("pair", [pair for _, pair, _, _ in ROBOTS])
@pytest.mark.parametrize("offset", [0, 1, 16, 299])
def test_open_time_filtered_market_has_all_73_closed_price_points(pair, offset):
    end = 1791167400 + offset
    start = end - 43200
    rows = market_rows(start, end)
    session = OpenTimeSession(rows)
    client = PublicKlinePrices(session=session)
    samples = client.samples(pair, start, end)
    ticks = list(range(start, end + 1, 600))
    curve = sampled_curve(samples, ticks)

    assert len(curve) == 73
    assert all(point["value"] is not None for point in curve)
    prior_close = (start // 300) * 300 - .001
    assert samples[0][0] == pytest.approx(prior_close, rel=0, abs=1e-6)
    assert 0 <= start - samples[0][0] <= 300
    assert all(start - 300 <= timestamp <= end for timestamp, _ in samples)

    # The fake includes the candle that opened before end and closes after it.
    # It must survive REST filtering but never enter the curve as a final price.
    url, call, returned = session.calls[0]
    unclosed = [row for row in returned if row[6] / 1000 > end]
    assert len(unclosed) == 1
    assert all(timestamp != unclosed[0][6] / 1000 for timestamp, _ in samples)
    assert len(returned) <= 150
    assert url == PRICE_URL
    assert call["params"]["symbol"] == pair.replace("-", "")
    assert call["params"]["interval"] == "5m"
    assert call["params"]["limit"] == 150
    assert call["verify"] is True and call["allow_redirects"] is False
    session.responses[0].close.assert_called_once()


def test_actual_oci_timestamp_retains_left_close_before_window_by_16_seconds():
    end, start = 1791167416, 1791124216
    opening, closing = 1791123900, 1791124199.999
    rows = market_rows(start, end)
    session = OpenTimeSession(rows)
    values = PublicKlinePrices(session=session).samples("BTC-FDUSD", start, end)
    assert opening * 1000 >= session.calls[0][1]["params"]["startTime"]
    assert values[0][0] == closing
    assert start - values[0][0] == pytest.approx(16.001, abs=1e-6)
    assert sampled_curve(values, [start])[0]["value"] is not None

    # The previous query excluded that candle by filtering its opening time.
    previously_returned = [row for row in rows if (start - 300) * 1000 <= row[0] <= end * 1000]
    assert all(row[0] != opening * 1000 for row in previously_returned)


def test_expanded_request_does_not_extend_300_second_freshness_or_fill_gaps():
    end = 1791167416
    start = end - 43200
    rows = market_rows(start, end)
    left_open = (start // 300 - 1) * 300
    # Keep the older bar that the wider request can return, remove the usable
    # left-edge bar and one interior bar. Neither stale value may fill a gap.
    interior_tick = start + 600
    interior_open = (interior_tick // 300 - 1) * 300
    rows = [row for row in rows if row[0] // 1000 not in (left_open, interior_open)]
    session = OpenTimeSession(rows)
    samples = PublicKlinePrices(session=session).samples("ETH-USDT", start, end)
    curve = sampled_curve(samples, [start, start + 600, start + 1200])
    assert curve[0]["value"] is None
    assert curve[1]["value"] is None
    assert curve[2]["value"] is not None
    assert all(timestamp >= start - 300 for timestamp, _ in samples)
