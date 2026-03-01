from __future__ import annotations
import numpy as np
import pandas as pd

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Classic Wilder ADX (simple rolling; good enough for regime gating)
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr * period)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr * period)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(period).mean()
    return adx

def detect_regime(df: pd.DataFrame,
                  adx_period: int = 14,
                  atr_period: int = 14,
                  slope_lookback: int = 20,
                  trend_adx_threshold: float = 22.0,
                  vol_atr_pct_threshold: float = 0.004):
    """
    Returns dict:
      {
        "trend": "TREND"|"RANGE",
        "vol": "HIGH_VOL"|"LOW_VOL",
        "adx": float,
        "atr_pct": float,
        "slope": float
      }
    Assumes df has columns: open, high, low, close
    """
    if df is None or df.empty or len(df) < max(adx_period, atr_period, slope_lookback) + 5:
        return {"trend": "UNKNOWN", "vol": "UNKNOWN", "adx": 0.0, "atr_pct": 0.0, "slope": 0.0}

    adx = _adx(df, adx_period).iloc[-1]
    atr = _atr(df, atr_period).iloc[-1]
    close = float(df["close"].iloc[-1]) if "close" in df else 0.0

    # ATR as percent of price (volatility proxy)
    atr_pct = float(atr / close) if close > 0 else 0.0

    # Trend slope proxy: EMA(10) - EMA(30) over lookback normalized by price
    ema_fast = _ema(df["close"], 10)
    ema_slow = _ema(df["close"], 30)
    spread = (ema_fast - ema_slow)

    # Slope of spread over last N bars
    y = spread.tail(slope_lookback).values
    if len(y) < slope_lookback:
        slope = 0.0
    else:
        x = np.arange(len(y))
        # simple linear regression slope
        slope = float(np.polyfit(x, y, 1)[0])

    # TREND if ADX high AND spread slope meaningful
    trend = "TREND" if (float(adx) >= trend_adx_threshold and abs(slope) > 1e-6) else "RANGE"
    vol = "HIGH_VOL" if atr_pct >= vol_atr_pct_threshold else "LOW_VOL"

    return {
        "trend": trend,
        "vol": vol,
        "adx": float(adx) if not np.isnan(adx) else 0.0,
        "atr_pct": float(atr_pct),
        "slope": float(slope),
    }