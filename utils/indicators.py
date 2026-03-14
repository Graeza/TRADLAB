# NOTE: This file is based on your previous version.
# It assumes columns like: open, high, low, close; and for trend uses DateTime.
# If your pipeline uses a different datetime column, adjust analyze_stock_trend accordingly.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional
import talib as ta


# =========================================================
#  SHARED HELPERS
# =========================================================

def linear_regression_trend(y_values):
    """
    Fit a linear regression to a 1D array of values.
    Returns slope and intercept.
    """
    y = np.array(y_values).reshape(-1, 1)
    x = np.arange(len(y_values)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    return slope, intercept

# =========================================================
#  INDICATORS
# =========================================================

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def calculate_ema(df, column, period=10):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    ema_series = ta.EMA(df[column].values, timeperiod=period)
    df[f"{column}_EMA{period}"] = ema_series

    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    close = df['close'].values

    macd, signal_line, hist = ta.MACD(
        close,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )

    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist

    return df

# =========================================================
#  TREND ANALYSIS
# =========================================================

def analyze_stock_trend(df):
    """
    Fit a regression line to closing prices over time.
    Returns slope, intercept, trendline array, and trend label.
    """
    df['Date_Ordinal'] = df['DateTime'].map(pd.Timestamp.toordinal)

    slope, intercept = linear_regression_trend(df['close'].values)
    trendline = slope * np.arange(len(df)) + intercept

    trend = 'Bullish' if slope > 0 else 'Bearish'
    df['Trend'] = trend

    return slope, intercept, trendline, trend

def analyze_positive_candles(df, lookback=50, slope_threshold=0.0):
    """
    Analyze trend of positive candles using regression on opens and highs.
    """
    positive = df[df['close'] > df['open']]

    if len(positive) < lookback:
        df['positive_candles'] = 'Insufficient data'
        return df

    recent = positive.tail(lookback)

    open_slope, _ = linear_regression_trend(recent['open'].values)
    high_slope, _ = linear_regression_trend(recent['high'].values)

    if open_slope > slope_threshold and high_slope > slope_threshold:
        trend = 'Rising'
    elif open_slope < -slope_threshold and high_slope < -slope_threshold:
        trend = 'Falling'
    else:
        trend = 'Sideways'

    df['positive_candles'] = None
    df.at[df.index[-1], 'positive_candles'] = trend

    return df

import math


def _find_price_col(df: pd.DataFrame) -> str:
    for c in ("close", "Close", "price"):
        if c in df.columns:
            return c
    raise ValueError("No supported price column found")


def hourly_trend_direction(
    h1_df: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200,
    slope_lookback: int = 5,
    slope_threshold: float = 0.0,
) -> dict:
    """
    Determine higher-timeframe trend using EMA alignment + slow EMA slope.
    Returns one of: bullish / bearish / neutral
    """
    if h1_df is None or h1_df.empty:
        return {
            "h1_trend": "neutral",
            "ema_fast": None,
            "ema_slow": None,
            "ema_slow_slope": None,
            "reason": "no_h1_data",
        }

    df = h1_df.copy()
    price_col = _find_price_col(df)

    if len(df) < max(slow_period + slope_lookback + 5, 220):
        return {
            "h1_trend": "neutral",
            "ema_fast": None,
            "ema_slow": None,
            "ema_slow_slope": None,
            "reason": "insufficient_h1_bars",
        }

    df = calculate_ema(df, price_col, fast_period)
    df = calculate_ema(df, price_col, slow_period)

    fast_col = f"{price_col}_EMA{fast_period}"
    slow_col = f"{price_col}_EMA{slow_period}"

    latest = df.iloc[-1]
    ema_fast = float(latest[fast_col])
    ema_slow = float(latest[slow_col])
    close_px = float(latest[price_col])

    slow_series = df[slow_col].dropna()
    if len(slow_series) >= slope_lookback + 1:
        ema_slow_slope = float(slow_series.iloc[-1] - slow_series.iloc[-1 - slope_lookback])
    else:
        ema_slow_slope = 0.0

    if close_px > ema_fast > ema_slow and ema_slow_slope > slope_threshold:
        trend = "bullish"
    elif close_px < ema_fast < ema_slow and ema_slow_slope < -slope_threshold:
        trend = "bearish"
    else:
        trend = "neutral"

    return {
        "h1_trend": trend,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "ema_slow_slope": ema_slow_slope,
        "reason": "ok",
    }


def find_hourly_support_resistance(
    h1_df: pd.DataFrame,
    current_price: Optional[float] = None,   
    lookback: int = 120,
    swing_window: int = 2,
) -> dict:
    """
    Find nearest H1 support below price and nearest H1 resistance above price
    using simple swing lows / swing highs.
    """
    if h1_df is None or h1_df.empty:
        return {
            "h1_support": None,
            "h1_resistance": None,
            "dist_to_h1_support": None,
            "dist_to_h1_resistance": None,
            "reason": "no_h1_data",
        }

    if not {"high", "low", "close"}.issubset(h1_df.columns):
        return {
            "h1_support": None,
            "h1_resistance": None,
            "dist_to_h1_support": None,
            "dist_to_h1_resistance": None,
            "reason": "missing_ohlc",
        }

    df = h1_df.tail(max(lookback, 10)).copy().reset_index(drop=True)

    if current_price is None:
        current_price = float(df.iloc[-1]["close"])

    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(swing_window, len(df) - swing_window):
        h = highs[i]
        l = lows[i]

        left_highs = highs[i - swing_window:i]
        right_highs = highs[i + 1:i + 1 + swing_window]
        left_lows = lows[i - swing_window:i]
        right_lows = lows[i + 1:i + 1 + swing_window]

        if all(h >= x for x in left_highs) and all(h >= x for x in right_highs):
            swing_highs.append(float(h))

        if all(l <= x for x in left_lows) and all(l <= x for x in right_lows):
            swing_lows.append(float(l))

    # de-duplicate close levels a bit
    def _dedupe(levels: list[float], min_sep_ratio: float = 0.0005) -> list[float]:
        out: list[float] = []
        for lv in sorted(levels):
            if not out:
                out.append(lv)
                continue
            if abs(lv - out[-1]) / max(abs(out[-1]), 1e-9) > min_sep_ratio:
                out.append(lv)
        return out

    swing_highs = _dedupe(swing_highs)
    swing_lows = _dedupe(swing_lows)

    support_candidates = [x for x in swing_lows if x < current_price]
    resistance_candidates = [x for x in swing_highs if x > current_price]

    support = max(support_candidates) if support_candidates else None
    resistance = min(resistance_candidates) if resistance_candidates else None

    dist_support = None if support is None else float(current_price - support)
    dist_resistance = None if resistance is None else float(resistance - current_price)

    return {
        "h1_support": support,
        "h1_resistance": resistance,
        "dist_to_h1_support": dist_support,
        "dist_to_h1_resistance": dist_resistance,
        "swing_high_count": len(swing_highs),
        "swing_low_count": len(swing_lows),
        "reason": "ok",
    }


def add_h1_context_to_df(
    primary_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    atr_col: Optional[str] = None,
    sr_atr_buffer_mult: float = 0.50,
) -> pd.DataFrame:
    """
    Add latest H1 trend + S/R context columns to every row of primary_df.
    Typically used so strategies can read the last row normally.
    """
    if primary_df is None or primary_df.empty:
        return primary_df

    out = primary_df.copy()
    current_price = float(out.iloc[-1]["close"]) if "close" in out.columns else None

    trend_info = hourly_trend_direction(h1_df)
    sr_info = find_hourly_support_resistance(h1_df, current_price=current_price)

    atr_value = None
    if atr_col and atr_col in out.columns:
        try:
            atr_value = float(out.iloc[-1][atr_col])
        except Exception:
            atr_value = None

    dist_sup = sr_info.get("dist_to_h1_support")
    dist_res = sr_info.get("dist_to_h1_resistance")

    near_support = False
    near_resistance = False

    if atr_value is not None and atr_value > 0:
        if dist_sup is not None and dist_sup <= atr_value * sr_atr_buffer_mult:
            near_support = True
        if dist_res is not None and dist_res <= atr_value * sr_atr_buffer_mult:
            near_resistance = True

    ctx = {
        "h1_trend": trend_info.get("h1_trend", "neutral"),
        "h1_ema_fast": trend_info.get("ema_fast"),
        "h1_ema_slow": trend_info.get("ema_slow"),
        "h1_ema_slow_slope": trend_info.get("ema_slow_slope"),
        "h1_support": sr_info.get("h1_support"),
        "h1_resistance": sr_info.get("h1_resistance"),
        "dist_to_h1_support": dist_sup,
        "dist_to_h1_resistance": dist_res,
        "near_h1_support": near_support,
        "near_h1_resistance": near_resistance,
    }

    for k, v in ctx.items():
        out[k] = v

    return out