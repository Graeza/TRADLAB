from __future__ import annotations

import argparse

import pandas as pd

from core.database import MarketDatabase
from config.settings import DB_PATH, PRIMARY_TIMEFRAME, LABEL_HORIZON_BARS


def export(
    symbol: str,
    timeframe: int = PRIMARY_TIMEFRAME,
    out_csv: str = "dataset.csv",
    limit: int = 200000,
) -> str:
    """Export a supervised training dataset by joining features + labels."""
    db = MarketDatabase(DB_PATH)
    feats = db.load_features(symbol, timeframe, limit=limit)
    if feats.empty:
        raise SystemExit("No features found.")

    labels = pd.read_sql_query(
        "SELECT time, future_return, y_class FROM labels WHERE symbol=? AND timeframe=? AND horizon_bars=?",
        db.conn,
        params=(symbol, timeframe, LABEL_HORIZON_BARS),
    )

    ds = feats.merge(labels, on="time", how="inner")
    ds.to_csv(out_csv, index=False)
    print(f"Wrote {len(ds)} rows -> {out_csv}")
    return out_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a dataset CSV from the bot's SQLite DB (features + labels).")
    p.add_argument("--symbol", default="EURUSD", help="Symbol to export")
    p.add_argument("--timeframe", type=int, default=PRIMARY_TIMEFRAME, help="MT5 timeframe int")
    p.add_argument("--out", default="dataset.csv", help="Output CSV filename")
    p.add_argument("--limit", type=int, default=200000, help="Max feature rows to load")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(symbol=args.symbol, timeframe=args.timeframe, out_csv=args.out, limit=args.limit)
