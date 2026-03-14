from __future__ import annotations

from typing import Optional
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from utils.indicators import add_h1_context_to_df


class BreakoutStrategy(Strategy):
    name = "BREAKOUT"
    allowed_trends = {"TREND"}

    def __init__(
        self,
        lookback: int = 20,
        buffer_atr_mult: float = 0.30,
        wick_to_body_max: float = 2.0,
        use_h1_filter: bool = True,
        h1_tf: int = 60,
        h1_sr_buffer_atr_mult: float = 0.50,
        allow_neutral_h1: bool = False,
    ):
        self.lookback = int(lookback)
        self.buffer_atr_mult = float(buffer_atr_mult)
        self.wick_to_body_max = float(wick_to_body_max)

        self.use_h1_filter = bool(use_h1_filter)
        self.h1_tf = int(h1_tf)
        self.h1_sr_buffer_atr_mult = float(h1_sr_buffer_atr_mult)
        self.allow_neutral_h1 = bool(allow_neutral_h1)

    @staticmethod
    def _find_atr_col(df: pd.DataFrame) -> Optional[str]:
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _wick_metrics(row: pd.Series):
        o = float(row.get("open", row.get("close", 0.0)) or 0.0)
        h = float(row.get("high", 0.0) or 0.0)
        l = float(row.get("low", 0.0) or 0.0)
        c = float(row.get("close", 0.0) or 0.0)
        upper = h - max(o, c)
        lower = min(o, c) - l
        return float(max(upper, 0.0)), float(max(lower, 0.0))

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or len(df) < self.lookback + 1:
            return StrategyResult(
                name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                meta={"reason": "insufficient_bars"},
            )

        atr_col = self._find_atr_col(df)

        h1_df = data_by_tf.get(self.h1_tf)
        if self.use_h1_filter and h1_df is not None and not h1_df.empty:
            try:
                df = add_h1_context_to_df(
                    df,
                    h1_df,
                    atr_col=atr_col,
                    sr_atr_buffer_mult=self.h1_sr_buffer_atr_mult,
                )
            except Exception:
                pass

        recent = df.tail(self.lookback + 1)
        if len(recent) < self.lookback + 1:
            return StrategyResult(
                name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                meta={"reason": "insufficient_recent_bars"},
            )

        prev = recent.iloc[:-1]
        last = recent.iloc[-1]

        hh = float(prev["high"].max())
        ll = float(prev["low"].min())
        c = float(last["close"])

        buffer = 0.0
        if atr_col:
            try:
                atr = float(last.get(atr_col, 0.0) or 0.0)
                if atr > 0:
                    buffer = float(self.buffer_atr_mult) * atr
            except Exception:
                buffer = 0.0

        upper_wick, lower_wick = self._wick_metrics(last)
        o = float(last.get("open", last.get("close", 0.0)) or 0.0)
        body = abs(float(last.get("close", 0.0) or 0.0) - o)
        body = max(body, 1e-9)

        h1_trend = str(last.get("h1_trend", "neutral")).lower()
        near_h1_support = bool(last.get("near_h1_support", False))
        near_h1_resistance = bool(last.get("near_h1_resistance", False))

        base_meta = {
            "hh": hh,
            "ll": ll,
            "close": c,
            "buffer": buffer,
            "h1_trend": h1_trend,
            "h1_support": last.get("h1_support"),
            "h1_resistance": last.get("h1_resistance"),
            "dist_to_h1_support": last.get("dist_to_h1_support"),
            "dist_to_h1_resistance": last.get("dist_to_h1_resistance"),
            "near_h1_support": near_h1_support,
            "near_h1_resistance": near_h1_resistance,
        }

        buy_breakout = c > (hh + buffer)
        sell_breakout = c < (ll - buffer)

        if buy_breakout:
            if (upper_wick / body) > float(self.wick_to_body_max):
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={**base_meta, "reason": "buy_breakout_rejected_wick", "upper_wick": upper_wick, "body": body},
                )

            if self.use_h1_filter:
                buy_allowed = (h1_trend == "bullish") or (self.allow_neutral_h1 and h1_trend == "neutral")
                if not buy_allowed:
                    return StrategyResult(
                        name=self.name,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        meta={**base_meta, "reason": "blocked_by_h1_trend"},
                    )
                if near_h1_resistance:
                    return StrategyResult(
                        name=self.name,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        meta={**base_meta, "reason": "too_close_to_h1_resistance"},
                    )

            conf = 0.60 + (0.05 if h1_trend == "bullish" else 0.0)
            return StrategyResult(
                name=self.name,
                signal=Signal.BUY,
                confidence=min(conf, 0.95),
                meta=base_meta,
            )

        if sell_breakout:
            if (lower_wick / body) > float(self.wick_to_body_max):
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={**base_meta, "reason": "sell_breakout_rejected_wick", "lower_wick": lower_wick, "body": body},
                )

            if self.use_h1_filter:
                sell_allowed = (h1_trend == "bearish") or (self.allow_neutral_h1 and h1_trend == "neutral")
                if not sell_allowed:
                    return StrategyResult(
                        name=self.name,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        meta={**base_meta, "reason": "blocked_by_h1_trend"},
                    )
                if near_h1_support:
                    return StrategyResult(
                        name=self.name,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        meta={**base_meta, "reason": "too_close_to_h1_support"},
                    )

            conf = 0.60 + (0.05 if h1_trend == "bearish" else 0.0)
            return StrategyResult(
                name=self.name,
                signal=Signal.SELL,
                confidence=min(conf, 0.95),
                meta=base_meta,
            )

        return StrategyResult(
            name=self.name,
            signal=Signal.HOLD,
            confidence=0.0,
            meta=base_meta,
        )