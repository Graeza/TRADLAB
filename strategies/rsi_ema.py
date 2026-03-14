from __future__ import annotations
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from utils.indicators import add_h1_context_to_df


class RSIEMAStrategy(Strategy):
    name = "RSI_EMA"
    allowed_trends = {"RANGE"}

    def __init__(
        self,
        rsi_low: float = 30,
        rsi_high: float = 70,
        ema_fast: int = 10,
        ema_slow: int = 21,
        spike_cooldown_bars: int = 3,
        spike_atr_mult: float = 2.0,
        use_h1_filter: bool = True,
        h1_tf: int = 60,
        h1_sr_buffer_atr_mult: float = 0.50,
        allow_neutral_h1: bool = True,
    ):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.spike_cooldown_bars = int(spike_cooldown_bars)
        self.spike_atr_mult = float(spike_atr_mult)

        self.use_h1_filter = bool(use_h1_filter)
        self.h1_tf = int(h1_tf)
        self.h1_sr_buffer_atr_mult = float(h1_sr_buffer_atr_mult)
        self.allow_neutral_h1 = bool(allow_neutral_h1)

    @staticmethod
    def _find_atr_col(df: pd.DataFrame) -> str | None:
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in df.columns:
                return c
        return None

    def _evaluate(self, data_by_tf: dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return StrategyResult(
                name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                meta={"reason": "no_data"},
            )

        if len(df) < 1:
            return StrategyResult(
                name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                meta={"reason": "insufficient_bars"},
            )

        atr_col = self._find_atr_col(df)

        # attach H1 context onto the primary dataframe when available
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
                # fail open: keep strategy running even if H1 context calc fails
                pass

        # df is already closed-bar data in your current pipeline
        row = df.iloc[-1]

        # spike cooldown
        if atr_col and {"high", "low"}.issubset(df.columns) and self.spike_cooldown_bars > 0:
            tail = df.tail(max(self.spike_cooldown_bars, 1))
            try:
                atr = float(row.get(atr_col, 0.0) or 0.0)
                if atr > 0:
                    ranges = (tail["high"].astype(float) - tail["low"].astype(float)).abs()
                    if float(ranges.max()) > float(self.spike_atr_mult) * atr:
                        return StrategyResult(
                            name=self.name,
                            signal=Signal.HOLD,
                            confidence=0.0,
                            meta={
                                "reason": "spike_cooldown",
                                "max_range": float(ranges.max()),
                                "atr": atr,
                            },
                        )
            except Exception:
                pass

        rsi = float(row.get("RSI", 50))
        ema_f = float(row.get(f"close_EMA{self.ema_fast}", row.get("close", 0)))
        ema_s = float(row.get(f"close_EMA{self.ema_slow}", row.get("close", 0)))

        h1_trend = str(row.get("h1_trend", "neutral")).lower()
        near_h1_support = bool(row.get("near_h1_support", False))
        near_h1_resistance = bool(row.get("near_h1_resistance", False))

        base_meta = {
            "rsi": rsi,
            "ema": (ema_f, ema_s),
            "h1_trend": h1_trend,
            "h1_support": row.get("h1_support"),
            "h1_resistance": row.get("h1_resistance"),
            "dist_to_h1_support": row.get("dist_to_h1_support"),
            "dist_to_h1_resistance": row.get("dist_to_h1_resistance"),
            "near_h1_support": near_h1_support,
            "near_h1_resistance": near_h1_resistance,
        }

        buy_signal = (rsi <= self.rsi_low and ema_f > ema_s)
        sell_signal = (rsi >= self.rsi_high and ema_f < ema_s)

        if self.use_h1_filter:
            buy_allowed = (h1_trend == "bullish") or (self.allow_neutral_h1 and h1_trend == "neutral")
            sell_allowed = (h1_trend == "bearish") or (self.allow_neutral_h1 and h1_trend == "neutral")

            if buy_signal and not buy_allowed:
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={**base_meta, "reason": "blocked_by_h1_trend"},
                )

            if sell_signal and not sell_allowed:
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={**base_meta, "reason": "blocked_by_h1_trend"},
                )

            if buy_signal and near_h1_resistance:
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={**base_meta, "reason": "too_close_to_h1_resistance"},
                )

            if sell_signal and near_h1_support:
                return StrategyResult(
                    name=self.name,
                    signal=Signal.HOLD,
                    confidence=0.0,
                    meta={**base_meta, "reason": "too_close_to_h1_support"},
                )

        if buy_signal:
            conf = 0.62
            if h1_trend == "bullish":
                conf += 0.06
            return StrategyResult(
                name=self.name,
                signal=Signal.BUY,
                confidence=min(conf, 0.95),
                meta=base_meta,
            )

        if sell_signal:
            conf = 0.62
            if h1_trend == "bearish":
                conf += 0.06
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