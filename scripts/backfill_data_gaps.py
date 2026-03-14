from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# Ensure project root (parent of /scripts) is on sys.path when running as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import DB_PATH, SYMBOL_LIST, TIMEFRAME_LIST, DATA_QUALITY_OUT_DIR
from core.mt5_worker import MT5Client
from core.data_fetcher import DataFetcher
from core.database import MarketDatabase


TF_SECONDS = {
    1: 60,
    2: 120,
    3: 180,
    4: 240,
    5: 300,
    6: 360,
    10: 600,
    12: 720,
    15: 900,
    20: 1200,
    30: 1800,
    60: 3600,
    120: 7200,
    180: 10800,
    240: 14400,
    360: 21600,
    480: 28800,
    720: 43200,
    1440: 86400,
    10080: 604800,
    43200: 2592000,

    16385: 3600,
    16386: 7200,
    16387: 10800,
    16388: 14400,
    16390: 21600,
    16392: 28800,
    16396: 43200,
    16408: 86400,
    32769: 604800,
    49153: 2592000,
}


def _safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out)
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2.strip("_")


def _read_gap_rows(gaps_csv_path: str) -> list[dict]:
    if not os.path.exists(gaps_csv_path):
        return []

    rows: list[dict] = []
    with open(gaps_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "prev_time": int(float(row["prev_time"])),
                    "time": int(float(row["time"])),
                    "delta_s": int(float(row["delta_s"])),
                    "expected_s": int(float(row["expected_s"])),
                    "missing_bars": int(float(row["missing_bars"])),
                })
            except Exception:
                continue
    return rows


def _iter_target_pairs(symbols: list[str], timeframes: list[int]) -> list[tuple[str, int]]:
    return [(s, int(tf)) for s in symbols for tf in timeframes]


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill missing OHLCV bars using gap CSVs from the audit output.")
    ap.add_argument("--db", type=str, default=DB_PATH)
    ap.add_argument("--audit-dir", type=str, default=DATA_QUALITY_OUT_DIR)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--timeframes", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Show what would be fetched without writing to DB")
    ap.add_argument("--rerun-audit", action="store_true", help="Print reminder to rerun audit afterward")
    args = ap.parse_args()

    symbols = list(args.symbols) if args.symbols else list(SYMBOL_LIST)
    timeframes = list(args.timeframes) if args.timeframes else list(TIMEFRAME_LIST)

    gaps_dir = os.path.join(os.path.abspath(args.audit_dir), "gap_details")
    if not os.path.isdir(gaps_dir):
        raise SystemExit(f"Gap details folder not found: {gaps_dir}")

    db = MarketDatabase(args.db)
    mt5 = MT5Client()
    mt5.start()
    fetcher = DataFetcher(mt5)

    total_intervals = 0
    total_inserted = 0
    repaired_series = 0

    try:
        for symbol, timeframe in _iter_target_pairs(symbols, timeframes):
            expected_s = TF_SECONDS.get(int(timeframe))
            if not expected_s:
                print(f"[BACKFILL] Skipping unsupported timeframe mapping: symbol={symbol} tf={timeframe}")
                continue

            base = f"{_safe_name(symbol)}_tf{int(timeframe)}_gaps.csv"
            gaps_csv_path = os.path.join(gaps_dir, base)
            gap_rows = _read_gap_rows(gaps_csv_path)

            if not gap_rows:
                print(f"[BACKFILL] No gap file or no gaps: symbol={symbol} tf={timeframe}")
                continue

            print(f"[BACKFILL] {symbol} tf={timeframe}: {len(gap_rows)} gap interval(s)")
            repaired_this_series = 0

            for i, gap in enumerate(gap_rows, start=1):
                prev_time = int(gap["prev_time"])
                next_time = int(gap["time"])

                # exact missing-bar range
                start_s = prev_time + expected_s
                end_s = next_time - expected_s

                if end_s < start_s:
                    continue

                total_intervals += 1
                start_dt = datetime.fromtimestamp(start_s, tz=timezone.utc).isoformat()
                end_dt = datetime.fromtimestamp(end_s, tz=timezone.utc).isoformat()

                print(f"[BACKFILL]   gap {i}: {start_dt} -> {end_dt}")

                if args.dry_run:
                    continue

                df = fetcher.fetch_range(symbol, timeframe, start_s, end_s)
                if df.empty:
                    print(f"[BACKFILL]   no bars returned for this interval")
                    continue

                cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[cols].drop_duplicates(subset=["time"]).sort_values("time")

                n = db.upsert_bars(df, symbol, timeframe)
                total_inserted += int(n)
                repaired_this_series += int(n)
                print(f"[BACKFILL]   inserted/upserted {n} bar(s)")

            if repaired_this_series > 0:
                repaired_series += 1

        print("\n=== Backfill Complete ===")
        print(f"Gap intervals processed: {total_intervals}")
        print(f"Series repaired:         {repaired_series}")
        print(f"Bars inserted/upserted:  {total_inserted}")

        if args.rerun_audit:
            print("\nNext step: rerun the DB gap audit to verify the holes were filled.")

    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()