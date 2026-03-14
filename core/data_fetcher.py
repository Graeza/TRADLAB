from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from core.mt5_worker import MT5Client


_TIMEFRAME_SECONDS = {
    # minute-style values
    1: 60,         # M1
    2: 120,        # M2
    3: 180,        # M3
    4: 240,        # M4
    5: 300,        # M5
    6: 360,        # M6
    10: 600,       # M10
    12: 720,       # M12
    15: 900,       # M15
    20: 1200,      # M20
    30: 1800,      # M30
    60: 3600,      # H1
    120: 7200,     # H2
    180: 10800,    # H3
    240: 14400,    # H4
    360: 21600,    # H6
    480: 28800,    # H8
    720: 43200,    # H12
    1440: 86400,   # D1
    10080: 604800, # W1
    43200: 2592000,# MN1 approx by minutes

    # common MT5 enum constants that can appear in your settings / GUI / DB
    16385: 3600,    # H1
    16386: 7200,    # H2
    16387: 10800,   # H3
    16388: 14400,   # H4
    16390: 21600,   # H6
    16392: 28800,   # H8
    16396: 43200,   # H12
    16408: 86400,   # D1
    32769: 604800,  # W1
    49153: 2592000, # MN1 approx
}


class DataFetcher:
    """Fetch raw OHLCV bars from MT5 (via MT5Client).

    Important: we only return closed candles by default
    (the currently-forming bar is removed).
    """

    def __init__(self, mt5: MT5Client):
        self.mt5 = mt5

    def _drop_unclosed_tail(self, df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        tf_sec = _TIMEFRAME_SECONDS.get(int(timeframe))
        if not tf_sec:
            # safest fallback: drop the last bar because MT5 often includes the live candle
            return df.iloc[:-1].copy() if len(df) > 1 else df.iloc[0:0].copy()

        now_s = int(time.time())
        last_open_s = int(df.iloc[-1]["time"])

        # candle is closed if we've reached or passed open + timeframe seconds
        if last_open_s + tf_sec > now_s:
            return df.iloc[:-1].copy() if len(df) > 1 else df.iloc[0:0].copy()

        return df

    def fetch_window(self, symbol: str, timeframe: int, n_bars: int = 2000) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = df["time"].astype(int)

        df = self._drop_unclosed_tail(df, timeframe)
        df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def fetch_since(self, symbol: str, timeframe: int, last_time_s: int | None, n_bars: int = 2000) -> pd.DataFrame:
        """Fetch bars and return only rows newer than last_time_s (epoch seconds)."""
        df = self.fetch_window(symbol, timeframe, n_bars=n_bars)
        if df.empty:
            return df
        if last_time_s is None:
            return df
        return df[df["time"] > int(last_time_s)].copy()

    def fetch_range(self, symbol: str, timeframe: int, from_time_s: int, to_time_s: int) -> pd.DataFrame:
        """Fetch bars between two UTC epoch-second timestamps, inclusive-ish on MT5 side.

        The currently-forming candle is removed automatically.
        """
        from_dt = datetime.fromtimestamp(int(from_time_s), tz=timezone.utc)
        to_dt = datetime.fromtimestamp(int(to_time_s), tz=timezone.utc)

        rates = self.mt5.copy_rates_range(symbol, timeframe, from_dt, to_dt)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = df["time"].astype(int)

        df = self._drop_unclosed_tail(df, timeframe)

        # keep the requested window only
        df = df[(df["time"] >= int(from_time_s)) & (df["time"] <= int(to_time_s))].copy()
        if df.empty:
            return df

        df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def latest_tick(self, symbol: str) -> dict | None:
        tick = self.mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": float(tick.ask - tick.bid),
        }