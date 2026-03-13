from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional, Any

from core.data_pipeline import DataPipeline
from core.ensemble import EnsembleEngine
from core.database import MarketDatabase
from core.performance_tracker import PerformanceTracker
from core.labeling import make_labels_from_bars
from utils.regime import detect_regime


class Orchestrator:
    def __init__(
        self,
        pipeline: DataPipeline,
        ensemble: EnsembleEngine,
        risk_manager,
        executor,
        db: MarketDatabase,
        symbols: list[str],
        timeframes: list[int],
        primary_tf: int,
        label_horizon_bars: int,
        log: Optional[Callable[[str], None]] = None,
        allow_new_trades_getter: Optional[Callable[[], bool]] = None,
        decision_callback: Optional[Callable[[str, dict, list], None]] = None,
    ):
        self.pipeline = pipeline
        self.ensemble = ensemble
        self.risk = risk_manager
        self.executor = executor
        self.db = db
        self.symbols = symbols
        self.timeframes = timeframes
        self.primary_tf = primary_tf
        self.label_horizon_bars = label_horizon_bars
        self.log = log or (lambda s: print(s, flush=True))
        self.allow_new_trades_getter = allow_new_trades_getter or (lambda: True)
        self.decision_callback = decision_callback
        self.perf = PerformanceTracker()
        self.current_session_id: int | None = None
        self.current_session_started_at: datetime | None = None

    def set_trade_session(self, session_id: int | None, started_at: datetime | None = None) -> None:
        self.current_session_id = session_id
        self.current_session_started_at = started_at

    def _serialize_exec_result(self, res: dict[str, Any]) -> str:
        try:
            safe = {}
            for k, v in (res or {}).items():
                if k == "raw":
                    continue
                if isinstance(v, datetime):
                    safe[k] = v.astimezone(timezone.utc).isoformat()
                else:
                    safe[k] = v
            return json.dumps(safe, default=str)
        except Exception:
            return ""

    def _log_open_trade(self, symbol: str, final_signal: dict, res: dict[str, Any]) -> None:
        session_id = self.current_session_id
        if not session_id or not res.get("ok"):
            return
        position_id = int(res.get("position_id") or 0)
        if position_id <= 0:
            position_id = int(res.get("order_ticket") or 0)
        self.db.log_trade_open(
            session_id=session_id,
            event_time=res.get("event_time"),
            symbol=symbol,
            side=str(res.get("action") or ""),
            volume=res.get("volume"),
            entry_price=res.get("price"),
            initial_sl=res.get("sl"),
            initial_tp=res.get("tp"),
            position_id=position_id if position_id > 0 else None,
            order_ticket=int(res.get("order_ticket") or 0) or None,
            deal_ticket=int(res.get("deal_ticket") or 0) or None,
            strategy_name=str(final_signal.get("winning_strategy") or ""),
            comment=str(final_signal.get("signal") or ""),
            raw_result_json=self._serialize_exec_result(res),
        )

    def _apply_trailing_stop_logic(self) -> None:
        session_id = self.current_session_id
        events = self.executor.manage_trailing_stops()
        for ev in events:
            if ev.get("ok"):
                self.log(
                    f"[TRAIL] {ev.get('symbol')} pos={ev.get('position_id')} "
                    f"SL {ev.get('old_sl')} -> {ev.get('new_sl')} @ price={ev.get('live_price')}"
                )
                if session_id and ev.get("position_id"):
                    try:
                        self.db.log_trade_stop_event(
                            session_id=session_id,
                            position_id=int(ev["position_id"]),
                            symbol=str(ev.get("symbol") or ""),
                            event_time=ev.get("event_time"),
                            event_type="TRAIL",
                            sl=ev.get("new_sl"),
                            tp=ev.get("tp"),
                            source="executor.trailing",
                            note=(
                                f"old_sl={ev.get('old_sl')} live_price={ev.get('live_price')} "
                                f"initial_risk={ev.get('initial_risk')}"
                            ),
                        )
                    except Exception as e:
                        self.log(f"[WARN] trail journal log failed for {ev.get('symbol')}: {e}")
            elif ev.get("reason") not in (None, "not_in_profit", "trigger_not_reached", "not_better_than_current", "step_not_reached", "no_initial_sl", "zero_initial_risk"):
                self.log(f"[WARN] trailing stop check failed for {ev.get('symbol')}: {ev}")

    def run_forever(self, sleep_s: int = 300, stop_event: Optional[threading.Event] = None):
        """Main loop.

        Fixes:
        - Remove duplicated SAFETY SWITCH / assess block.
        - Ensure primary timeframe presence check is correct.
        - Keep control flow clean: decide -> (optional) execute.
        """
        stop_event = stop_event or threading.Event()
        self.log("[BOT] Started")

        while not stop_event.is_set():
            try:
                self._apply_trailing_stop_logic()
            except Exception as e:
                self.log(f"[WARN] pre-loop trailing management failed: {e}")

            for symbol in self.symbols:
                if stop_event.is_set():
                    break
                try:
                    data_by_tf = self.pipeline.update_symbol(symbol, self.timeframes)
                    if self.primary_tf not in data_by_tf:
                        self.log(f"[WARN] {symbol}: no primary timeframe data")
                        continue

                    primary_df = data_by_tf.get(self.primary_tf)
                    regime = detect_regime(primary_df) if primary_df is not None else {"trend": "UNKNOWN", "vol": "UNKNOWN"}
                    final_signal, outputs = self.ensemble.run(data_by_tf, regime=regime)

                    # attach regime for downstream components
                    if isinstance(final_signal, dict):
                        final_signal["regime"] = regime
                        if outputs:
                            winner = max(outputs, key=lambda o: float(o.get("confidence", 0.0) or 0.0))
                            final_signal["winning_strategy"] = str(winner.get("name") or "")
                            final_signal["comment"] = str(winner.get("signal") or "")

                    # Performance tracking (non-blocking metadata)
                    self.perf.add_prediction(
                        symbol=symbol,
                        df_primary=primary_df,
                        horizon_bars=self.label_horizon_bars,
                        final=final_signal,
                        outputs=outputs,
                    )
                    self.perf.update_with_bars(symbol, primary_df)

                    # Emit structured decision event (GUI can subscribe)
                    if self.decision_callback:
                        try:
                            self.decision_callback(symbol, final_signal, outputs)
                        except Exception as e:
                            self.log(f"[WARN] decision_callback failed: {e}")

                    self.log(
                        f"[SIGNAL] {symbol}: {final_signal} | details={[(o.get('name'), o.get('signal'), o.get('confidence')) for o in outputs]}"
                    )

                    # Delayed labeling (primary timeframe)
                    bars = self.db.load_bars(symbol, self.primary_tf, limit=5000)
                    labels = make_labels_from_bars(bars, symbol, self.primary_tf, self.label_horizon_bars)
                    if not labels.empty:
                        n = self.db.upsert_labels(labels)
                        self.log(f"[LABEL] {symbol}: upserted {n} labels")

                    # SAFETY SWITCH: allow data + signals, but block opening new trades
                    if not self.allow_new_trades_getter():
                        self.log(f"[SAFE MODE] Entries blocked. Would have acted on {symbol}: {final_signal}")
                        continue

                    trade_params = self.risk.assess(final_signal, symbol)
                    if not trade_params:
                        self.log(f"[RISK] {symbol}: rejected")
                        continue

                    trade_params["strategy_name"] = str(final_signal.get("winning_strategy") or "")
                    trade_params["comment"] = str(final_signal.get("comment") or "ModularBot")
                    res = self.executor.execute(trade_params)
                    self.log(f"[EXEC] {symbol}: {res}")

                    if isinstance(res, dict) and res.get("ok"):
                        try:
                            self._log_open_trade(symbol, final_signal, res)
                        except Exception as e:
                            self.log(f"[WARN] journal open log failed for {symbol}: {e}")

                    try:
                        self._apply_trailing_stop_logic()
                    except Exception as e:
                        self.log(f"[WARN] trailing management failed after {symbol}: {e}")

                except Exception as e:
                    self.log(f"[ERROR] {symbol}: {e}")

            # responsive stop
            total = max(1, int(sleep_s))
            for _ in range(total):
                if stop_event.is_set():
                    break
                time.sleep(1)
                try:
                    self._apply_trailing_stop_logic()
                except Exception as e:
                    self.log(f"[WARN] idle trailing management failed: {e}")

        self.log("[BOT] Stopped")
