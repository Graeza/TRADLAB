from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config.settings import ML_MODEL_PATH


NON_FEATURE_COLS_DEFAULT = {
    "time", "dt", "datetime", "timestamp",
    "symbol", "timeframe", "tf",
    "label", "y", "target",
    "future_return", "y_class",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an ML classifier on an exported dataset CSV and save a model bundle (model + feature schema)."
    )
    p.add_argument("--csv", default="dataset.csv", help="Path to dataset CSV exported by scripts/export_dataset.py")
    p.add_argument("--label-col", default="y_class", help="Label column name in the CSV (default: y_class)")
    p.add_argument("--model-path", default=str(ML_MODEL_PATH), help="Output path for joblib bundle")
    p.add_argument("--model-version", default=None, help="String model version tag (e.g. ml_v3_2026-03-02)")
    p.add_argument("--schema-version", type=int, default=1, help="Integer schema version (bump when feature set changes)")
    p.add_argument("--strict-schema", action="store_true", help="If set, bot will refuse to trade on schema drift")
    p.add_argument("--fillna-value", default=None, help="Optional fill value for NaNs (e.g. 0.0). If omitted, NaNs cause rejection in live.")
    p.add_argument("--test-frac", type=float, default=0.2, help="Holdout fraction for evaluation (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--n-estimators", type=int, default=400, help="RandomForest n_estimators")
    p.add_argument("--max-depth", type=int, default=10, help="RandomForest max_depth")
    p.add_argument(
        "--non-feature-cols",
        default="",
        help="Comma-separated additional non-feature column names to exclude",
    )
    p.add_argument(
        "--class-to-signal",
        default='{"-1":"SELL","0":"HOLD","1":"BUY"}',
        help='JSON mapping from class to signal, e.g. '{"0":"SELL","1":"BUY"}'',
    )
    return p.parse_args()


def _infer_feature_cols(df: pd.DataFrame, label_col: str, non_feature_cols: set[str]) -> list[str]:
    # Keep original column order from CSV for stable schema.
    cols = []
    for c in df.columns:
        if c == label_col:
            continue
        if c in non_feature_cols:
            continue
        # keep numeric columns only
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found in CSV. Columns: {list(df.columns)[:20]}...")

    # If time exists, sort to avoid leakage
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    non_feature_cols = set(NON_FEATURE_COLS_DEFAULT)
    if args.non_feature_cols.strip():
        non_feature_cols |= {c.strip() for c in args.non_feature_cols.split(",") if c.strip()}

    feature_cols = _infer_feature_cols(df, args.label_col, non_feature_cols)
    if not feature_cols:
        raise SystemExit("No feature columns found (numeric columns excluding non-feature cols).")

    X = df[feature_cols].copy()
    y = df[args.label_col].copy()

    # Optional fill for training convenience
    fillna_value = None if args.fillna_value in (None, "", "None", "none") else float(args.fillna_value)
    if fillna_value is not None:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(fillna_value)

    # Time-aware split: last N% as test (no shuffle) if time column exists, else a normal split
    if "time" in df.columns:
        n = len(df)
        n_test = max(1, int(round(n * args.test_frac)))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_frac, random_state=args.random_state, shuffle=True
        )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    # Eval
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)

    print("=== Evaluation ===")
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report)

    # Bundle for the bot (schema enforcement + mapping)
    class_to_signal_raw = json.loads(args.class_to_signal)
    # normalize keys: allow "1" or 1
    class_to_signal = {}
    for k, v in class_to_signal_raw.items():
        try:
            kk = int(k)
        except Exception:
            kk = str(k)
        class_to_signal[kk] = str(v).upper()

    model_version = args.model_version or "ml_" + pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    bundle = {
        "model": clf,
        "feature_cols": feature_cols,  # exact list in training order
        "model_version": model_version,
        "schema_version": int(args.schema_version),
        "strict_schema": bool(args.strict_schema),
        "class_to_signal": class_to_signal,
        "fillna_value": fillna_value,  # None means: do not fill in live; reject NaN/inf instead
        "train_metrics": {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
        },
    }

    out_path = Path(args.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print("=== Saved ===")
    print(f"Bundle -> {out_path.resolve()}")
    print(f"model_version={model_version} schema_version={args.schema_version} strict_schema={args.strict_schema}")


if __name__ == "__main__":
    main()
