from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

from strategies.base import Strategy, StrategyOutput


class EnsembleEngine:
    def __init__(
        self,
        strategies: List[Strategy],
        weights: Optional[Dict[str, float]] = None,
        min_conf: float = 0.55,
        regime_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.strategies = strategies
        self.weights = weights or {s.name: 1.0 for s in strategies}
        self.min_conf = min_conf
        self.regime_multipliers = regime_multipliers or {}

    def _effective_weight(self, strategy_name: str, regime: Optional[Dict[str, Any]]) -> tuple[float, float, float, float]:
        """
        Returns: (base_weight, trend_mult, vol_mult, effective_weight)
        regime_multipliers format:
          {
            "TREND": {"RSIEMAStrategy": 0.7, ...},
            "RANGE": {...},
            "HIGH_VOL": {...},
            "LOW_VOL": {...}
          }
        """
        base = float(self.weights.get(strategy_name, 1.0))
        if not regime:
            return base, 1.0, 1.0, base

        trend = str(regime.get("trend", "")).upper()
        vol = str(regime.get("vol", "")).upper()

        trend_mult = float(self.regime_multipliers.get(trend, {}).get(strategy_name, 1.0))
        vol_mult = float(self.regime_multipliers.get(vol, {}).get(strategy_name, 1.0))
        eff = base * trend_mult * vol_mult
        return base, trend_mult, vol_mult, eff

    def run(
        self,
        data_by_tf: dict[int, pd.DataFrame],
        regime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[dict, List[StrategyOutput]]:
        outputs: List[StrategyOutput] = []
        score = 0.0
        total = 0.0

        for s in self.strategies:
            out = s.evaluate(data_by_tf) or {}
            # ensure name is consistent
            name = str(out.get("name") or getattr(s, "name", s.__class__.__name__))

            sig = str(out.get("signal") or "HOLD").upper()
            conf = float(out.get("confidence") or 0.0)

            base_w, trend_mult, vol_mult, w = self._effective_weight(name, regime)

            # attach useful debug info for your Strategy Debug Panel
            meta = dict(out.get("meta") or {})
            meta.update({
                "base_weight": base_w,
                "trend_mult": trend_mult,
                "vol_mult": vol_mult,
                "effective_weight": w,
            })
            out["meta"] = meta
            out["name"] = name

            outputs.append(out)

            if sig == "HOLD" or conf < self.min_conf:
                continue

            x = 1.0 if sig == "BUY" else -1.0
            score += w * conf * x
            total += w * conf

        if total == 0.0:
            final = {"signal": "HOLD", "confidence": 0.0}
        else:
            final = {
                "signal": "BUY" if score > 0 else "SELL",
                "confidence": min(1.0, abs(score) / total),
            }

        return final, outputs