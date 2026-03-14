from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyResult, Signal
from utils.indicators import add_h1_context_to_df


class MLStrategy(Strategy):
    name = "ML"

    def __init__(
        self,
        model,
        feature_cols: Optional[List[str]] = None,
        *,
        model_version: Optional[str] = None,
        schema_version: int = 1,
        strict_schema: bool = True,
        class_to_signal: Optional[Dict[object, str]] = None,
        drop_cols: Optional[List[str]] = None,
        fillna_value: Optional[float] = None,
        feature_set_version: Optional[int] = None,
        feature_set_id: Optional[str] = None,
        use_h1_meta: bool = True,
        h1_tf: int = 60,
        h1_sr_buffer_atr_mult: float = 0.50,
    ):
        """
        ML-backed strategy with feature-schema enforcement.

        The core failure mode in production ML trading systems is schema drift:
        columns added/removed/renamed, or ordering changes between training and live.

        This strategy can run in a strict mode that will REFUSE to trade if:
          - the live feature schema differs from what the model expects
          - required features are missing
          - features contain NaN/inf (unless fillna_value is provided)

        H1 context is attached to meta/debug output only by default.
        It is NOT fed into model inference unless your trained schema already includes it.
        """
        self.model = model
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.model_version = model_version
        self.schema_version = int(schema_version)
        self.strict_schema = bool(strict_schema)
        self.class_to_signal = class_to_signal
        self.drop_cols = set((drop_cols or []))
        self.fillna_value = fillna_value
        self.feature_set_version = int(feature_set_version) if feature_set_version is not None else None
        self.feature_set_id = str(feature_set_id) if feature_set_id is not None else None

        self.use_h1_meta = bool(use_h1_meta)
        self.h1_tf = int(h1_tf)
        self.h1_sr_buffer_atr_mult = float(h1_sr_buffer_atr_mult)

        self._expected_cols = None  # type: Optional[List[str]]
        if self.feature_cols:
            self._expected_cols = list(self.feature_cols)
        else:
            cols = getattr(self.model, "feature_names_in_", None)
            if cols is not None:
                try:
                    self._expected_cols = [str(c) for c in list(cols)]
                except Exception:
                    self._expected_cols = None

        self._expected_schema_id = self._schema_id(self._expected_cols) if self._expected_cols else None

    @staticmethod
    def _schema_id(cols: Optional[List[str]]) -> Optional[str]:
        if not cols:
            return None
        payload = "\n".join([str(c) for c in cols]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def _infer_live_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols = []
        for c in df.columns:
            if c in self.drop_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(str(c))
        return cols

    @staticmethod
    def _pred_to_signal(pred: Any) -> str:
        s = str(pred).upper()
        if s in {"BUY", "SELL", "HOLD"}:
            return s

        # common numeric conventions
        try:
            x = int(pred)
            if x > 0:
                return "BUY"
            if x < 0:
                return "SELL"
            return "HOLD"
        except Exception:
            return "HOLD"

    def _classes_default_mapping(self) -> Dict[object, str]:
        cls = list(getattr(self.model, "classes_", []) or [])
        if set(cls) == {-1, 0, 1}:
            return {-1: "SELL", 0: "HOLD", 1: "BUY"}
        if set(cls) == {0, 1, 2}:
            return {0: "SELL", 1: "HOLD", 2: "BUY"}
        if set(cls) == {0, 1}:  # default: 0=SELL, 1=BUY
            return {0: "SELL", 1: "BUY"}
        return {}

    def _clean_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.fillna_value is None:
            if X.isna().any().any():
                bad_cols = [str(c) for c in X.columns[X.isna().any()].tolist()]
                raise ValueError(f"NaN/inf in features: {bad_cols[:25]}")
            return X
        return X.fillna(float(self.fillna_value))

    @staticmethod
    def _find_atr_col(df: pd.DataFrame) -> Optional[str]:
        for c in ("ATR", "ATR14", "atr", "atr14"):
            if c in df.columns:
                return c
        return None

    def _evaluate(self, data_by_tf: Dict[int, pd.DataFrame]):
        df = next(iter(data_by_tf.values()))
        if df is None or df.empty:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {"reason": "no_data"},
            }

        # DataFetcher already returns closed-bar data, so use the latest row directly.
        if len(df) < 1:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {"reason": "insufficient_bars"},
            }

        # Keep a separate debug/context copy so we do NOT mutate the inference schema.
        debug_df = df
        if self.use_h1_meta:
            h1_df = data_by_tf.get(self.h1_tf)
            if h1_df is not None and not h1_df.empty:
                try:
                    debug_df = add_h1_context_to_df(
                        df.copy(),
                        h1_df,
                        atr_col=self._find_atr_col(df),
                        sr_atr_buffer_mult=self.h1_sr_buffer_atr_mult,
                    )
                except Exception:
                    debug_df = df

        meta_ctx = {}
        try:
            last_dbg = debug_df.iloc[-1]
            meta_ctx = {
                "h1_trend": str(last_dbg.get("h1_trend", "neutral")).lower(),
                "h1_support": last_dbg.get("h1_support"),
                "h1_resistance": last_dbg.get("h1_resistance"),
                "dist_to_h1_support": last_dbg.get("dist_to_h1_support"),
                "dist_to_h1_resistance": last_dbg.get("dist_to_h1_resistance"),
                "near_h1_support": bool(last_dbg.get("near_h1_support", False)),
                "near_h1_resistance": bool(last_dbg.get("near_h1_resistance", False)),
            }
        except Exception:
            meta_ctx = {}

        # --- Feature set version gating ---
        if self.feature_set_version is not None:
            if "feature_set_version" not in df.columns:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "missing_feature_set_version",
                        "expected_feature_set_version": self.feature_set_version,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }
            try:
                live_v = int(df["feature_set_version"].iloc[-1])
            except Exception:
                live_v = None
            if live_v != self.feature_set_version:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "feature_set_version_mismatch",
                        "expected_feature_set_version": self.feature_set_version,
                        "live_feature_set_version": live_v,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }

        if self.feature_set_id is not None:
            if "feature_set_id" not in df.columns:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "missing_feature_set_id",
                        "expected_feature_set_id": self.feature_set_id,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }
            live_id = str(df["feature_set_id"].iloc[-1])
            if live_id != self.feature_set_id:
                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "feature_set_id_mismatch",
                        "expected_feature_set_id": self.feature_set_id,
                        "live_feature_set_id": live_id,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }

        live_cols = self._infer_live_feature_cols(df)
        live_schema_id = self._schema_id(live_cols)

        expected_cols = self._expected_cols
        expected_schema_id = self._expected_schema_id

        # If we have an expected schema and strict mode, reject on drift (including order drift)
        if expected_cols and self.strict_schema:
            if live_cols != expected_cols:
                missing = [c for c in expected_cols if c not in live_cols]
                extra = [c for c in live_cols if c not in expected_cols]

                first_mismatch = None
                for i in range(min(len(live_cols), len(expected_cols))):
                    if live_cols[i] != expected_cols[i]:
                        first_mismatch = {"index": i, "expected": expected_cols[i], "got": live_cols[i]}
                        break

                return {
                    "name": self.name,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "meta": {
                        **meta_ctx,
                        "reason": "feature_schema_mismatch",
                        "expected_n": len(expected_cols),
                        "got_n": len(live_cols),
                        "missing_n": len(missing),
                        "extra_n": len(extra),
                        "missing": missing[:25],
                        "extra": extra[:25],
                        "first_mismatch": first_mismatch,
                        "expected_schema_id": expected_schema_id,
                        "live_schema_id": live_schema_id,
                        "schema_version": self.schema_version,
                        "model_version": self.model_version,
                    },
                }

            feature_cols = expected_cols
        else:
            if self.feature_cols is not None:
                feature_cols = list(self.feature_cols)
            else:
                cols = getattr(self.model, "feature_names_in_", None)
                if cols is not None:
                    try:
                        feature_cols = [str(c) for c in list(cols)]
                    except Exception:
                        feature_cols = live_cols
                else:
                    feature_cols = live_cols

        if not feature_cols:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {**meta_ctx, "reason": "no_features"},
            }

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {
                    **meta_ctx,
                    "reason": "missing_features",
                    "missing": missing[:25],
                    "missing_n": len(missing),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        try:
            X = df.loc[df.index[-1:], feature_cols].copy()
            X = self._clean_X(X)
        except Exception as e:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {
                    **meta_ctx,
                    "reason": "bad_features",
                    "error": str(e),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        conf = 0.55
        signal = "HOLD"
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                idx = int(np.argmax(proba))
                conf = float(np.max(proba))
                pred = getattr(self.model, "classes_", [None] * len(proba))[idx]
            else:
                pred = self.model.predict(X)[0]

            mapping = self.class_to_signal or self._classes_default_mapping()
            if mapping and pred in mapping:
                signal = str(mapping[pred]).upper()
            else:
                signal = self._pred_to_signal(pred)

        except Exception as e:
            return {
                "name": self.name,
                "signal": "HOLD",
                "confidence": 0.0,
                "meta": {
                    **meta_ctx,
                    "reason": "model_error",
                    "error": str(e),
                    "expected_schema_id": expected_schema_id,
                    "live_schema_id": live_schema_id,
                    "schema_version": self.schema_version,
                    "model_version": self.model_version,
                },
            }

        return {
            "name": self.name,
            "signal": signal,
            "confidence": float(conf),
            "meta": {
                **meta_ctx,
                "features_n": int(X.shape[1]),
                "filled_na": bool(self.fillna_value is not None),
                "expected_schema_id": expected_schema_id,
                "live_schema_id": live_schema_id,
                "schema_version": self.schema_version,
                "model_version": self.model_version,
            },
        }