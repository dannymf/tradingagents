"""Alpaca market-data vendor (OHLCV bars + indicators computed from them).

Alpaca is a *market data* source: it serves daily OHLCV bars via the
``StockHistoricalDataClient``. This module exposes two vendor functions that
mirror the yfinance / alpha_vantage implementations so the routing layer in
``interface.py`` can treat ``alpaca`` like any other vendor:

* :func:`get_stock`     -> ``get_stock_data``    (raw OHLCV, CSV string)
* :func:`get_indicator` -> ``get_indicators``    (stockstats over Alpaca bars)

Credentials come from the same ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` env
vars the execution-layer broker uses; the data API is shared between paper and
live accounts, so no paper flag is needed. Fundamentals and news are not market
data and are intentionally left to other vendors.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .symbol_utils import NoMarketDataError


class AlpacaNotConfiguredError(Exception):
    """Raised when Alpaca data credentials are missing from the environment.

    Surfaced as a generic vendor failure so the routing layer falls back to
    another configured vendor instead of fabricating a price.
    """


# Technical indicators stockstats can compute from OHLCV, with the same
# descriptions the other vendors append so agent prompts read identically
# regardless of which vendor served the data.
_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.",
    "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.",
    "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.",
    "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.",
    "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.",
    "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.",
    "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.",
    "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.",
    "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.",
    "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.",
    "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.",
    "vwma": "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.",
    "mfi": "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals.",
}


def _get_client() -> StockHistoricalDataClient:
    """Build the historical-data client from ALPACA_* env vars."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise AlpacaNotConfiguredError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set to use the Alpaca data vendor."
        )
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _fetch_daily_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch split/dividend-adjusted daily bars as a tz-naive OHLCV frame.

    Returns columns ``Date, Open, High, Low, Close, Volume`` (chronological).
    Raises :class:`NoMarketDataError` when Alpaca returns no rows so the router
    emits its single "no data" sentinel instead of a vendor-specific blank.
    """
    canonical = symbol.strip().upper()
    request = StockBarsRequest(
        symbol_or_symbols=canonical,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
    )
    bars = _get_client().get_stock_bars(request)
    df = bars.df

    if df is None or df.empty:
        raise NoMarketDataError(
            symbol, canonical, f"Alpaca returned no bars between {start.date()} and {end.date()}"
        )

    # bars.df is MultiIndexed on (symbol, timestamp); flatten to a plain frame.
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date")


def get_stock(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return daily OHLCV for ``symbol`` over the date range as a CSV string."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    data = _fetch_daily_bars(symbol, start, end)
    for col in ("Open", "High", "Low", "Close"):
        data[col] = data[col].round(2)

    csv_string = data.to_csv(index=False)
    canonical = symbol.strip().upper()
    header = f"# Stock data for {canonical} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def get_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Return a window of ``indicator`` values computed from Alpaca bars.

    Output matches the yfinance vendor: one ``YYYY-MM-DD: value`` line per
    calendar day in the look-back window followed by the indicator's
    description. Enough prior history is pulled so trailing indicators (e.g.
    the 200 SMA) are warmed up before the window starts.
    """
    from stockstats import wrap

    if indicator not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(_INDICATOR_DESCRIPTIONS.keys())}"
        )

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Pull ~1 extra trading year before the window so 200-period indicators are
    # fully warmed up; cap the end at curr_date to avoid look-ahead bias.
    history_start = before - timedelta(days=400)
    data = _fetch_daily_bars(symbol, history_start, curr_date_dt)

    df = wrap(data)
    df[indicator]  # trigger stockstats to compute the indicator column
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    values_by_date = {
        row["Date"]: ("N/A" if pd.isna(row[indicator]) else str(row[indicator]))
        for _, row in df.iterrows()
    }

    ind_string = ""
    cursor = curr_date_dt
    while cursor >= before:
        date_str = cursor.strftime("%Y-%m-%d")
        value = values_by_date.get(date_str, "N/A: Not a trading day (weekend or holiday)")
        ind_string += f"{date_str}: {value}\n"
        cursor -= relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + _INDICATOR_DESCRIPTIONS[indicator]
    )
