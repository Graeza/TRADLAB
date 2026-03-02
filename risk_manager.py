"""Risk management and position sizing.

This module keeps policy in one place:
- Whether a signal is tradable (confidence / spread gating)
- How big the trade should be (risk-based sizing)
- Where SL/TP should go (volatility-based distance)

NOTE: This implementation uses MetaTrader5 account + symbol metadata.
If you later isolate MT5 calls to a single thread, you can pass
market/account snapshots into `assess()` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import MetaTrader5 as mt5


@dataclass(frozen=True)
class RiskDecision:
    symbol: str
    action: str
    lot_size: float
    sl: float
    tp: float
    deviation: int

    def to_params(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "lot_size": float(self.lot_size),
            "sl": float(self.sl),
            "tp": float(self.tp),
            "deviation": int(self.deviation),
        }


class RiskManager:
    def __init__(
        self,
        max_risk_pct: float = 1.0,
        min_confidence: float = 0.60,
        sl_atr_mult: float = 2.0,
        tp_rr: float = 1.5,
        fallback_sl_pct: float = 0.003,
        max_spread_points: int = 50,
        base_deviation_points: int = 20,
    ):
        """Create a risk manager.

        Args:
            max_risk_pct: % of equity to risk per trade (e.g. 1.0 => 1%).
            min_confidence: minimum final signal confidence to allow entry.
            sl_atr_mult: SL distance multiplier applied to ATR% proxy.
            tp_rr: take-profit distance as SL_distance * tp_rr.
            fallback_sl_pct: if no ATR% available, use this % of price as SL distance.
            max_spread_points: reject trades if spread exceeds this many points.
            base_deviation_points: default slippage tolerance in points.
        """
        self.max_risk_pct = float(max_risk_pct)
        self.min_confidence = float(min_confidence)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_rr = float(tp_rr)
        self.fallback_sl_pct = float(fallback_sl_pct)
        self.max_spread_points = int(max_spread_points)
        self.base_deviation_points = int(base_deviation_points)

    @staticmethod
    def _equity() -> Optional[float]:
        acc = mt5.account_info()
        if acc is None:
            return None
        eq = getattr(acc, "equity", None)
        bal = getattr(acc, "balance", None)
        # Prefer equity (includes floating PnL); fallback to balance.
        val = eq if (eq is not None and eq > 0) else bal
        return float(val) if (val is not None and val > 0) else None

    @staticmethod
    def _money_per_lot_for_move(symbol_info, price_delta: float) -> float:
        """Approx loss for 1.0 lot if price moves by `price_delta`.

        Uses MT5 symbol metadata:
          money = (price_delta / tick_size) * tick_value
        """
        tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)
        if tick_size <= 0:
            tick_size = float(getattr(symbol_info, "point", 0.0) or 0.0)
        if tick_size <= 0 or tick_value <= 0:
            return 0.0
        ticks = abs(price_delta) / tick_size
        return float(ticks * tick_value)

    def assess(self, signal: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Return trade params dict for TradeExecutor, or None to reject."""
        if not isinstance(signal, dict):
            return None

        action = str(signal.get("signal") or "HOLD").upper()
        conf = float(signal.get("confidence") or 0.0)

        if action == "HOLD":
            return None
        if conf < self.min_confidence:
            return None

        # Make sure symbol is available/visible in MT5
        mt5.symbol_select(symbol, True)

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        eq = self._equity()
        if info is None or tick is None or eq is None:
            return None

        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        spread = abs(ask - bid)
        spread_points = int(round(spread / point)) if point > 0 else 0
        if point > 0 and spread_points > self.max_spread_points:
            return None

        # Entry reference price
        entry_price = ask if action == "BUY" else bid

        # Volatility-based SL distance
        regime = signal.get("regime") if isinstance(signal.get("regime"), dict) else {}
        atr_pct = float(regime.get("atr_pct") or 0.0)
        if atr_pct > 0:
            sl_dist = entry_price * atr_pct * self.sl_atr_mult
        else:
            sl_dist = entry_price * self.fallback_sl_pct

        # Safety: do not allow microscopic stops
        min_stop = (point * 10) if point > 0 else 0.0
        if sl_dist <= min_stop:
            sl_dist = max(min_stop, entry_price * self.fallback_sl_pct)

        if action == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * self.tp_rr
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * self.tp_rr

        # Risk-based sizing
        risk_money = eq * (self.max_risk_pct / 100.0)
        per_lot_loss = self._money_per_lot_for_move(info, sl_dist)
        if per_lot_loss <= 0:
            return None

        raw_lot = risk_money / per_lot_loss
        if raw_lot <= 0:
            return None

        # Deviation: base + a bit of spread buffer
        deviation = max(self.base_deviation_points, int(self.base_deviation_points + spread_points))

        decision = RiskDecision(
            symbol=symbol,
            action=action,
            lot_size=float(raw_lot),
            sl=float(sl),
            tp=float(tp),
            deviation=int(deviation),
        )
        return decision.to_params()
