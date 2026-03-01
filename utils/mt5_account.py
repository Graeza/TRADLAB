from __future__ import annotations
import MetaTrader5 as mt5

def get_account_summary() -> dict:
    acc = mt5.account_info()
    if acc is None:
        return {"ok": False, "error": f"account_info() failed: {mt5.last_error()}"}

    # Some terminals/brokers may not populate every field; use getattr safely
    summary = {
        "ok": True,
        "login": getattr(acc, "login", None),
        "server": getattr(acc, "server", None),
        "currency": getattr(acc, "currency", ""),
        "balance": float(getattr(acc, "balance", 0.0) or 0.0),
        "equity": float(getattr(acc, "equity", 0.0) or 0.0),
        "profit": float(getattr(acc, "profit", 0.0) or 0.0),  # floating PnL
        "margin": float(getattr(acc, "margin", 0.0) or 0.0),
        "margin_free": float(getattr(acc, "margin_free", 0.0) or 0.0),
        "margin_level": float(getattr(acc, "margin_level", 0.0) or 0.0),
        "leverage": int(getattr(acc, "leverage", 0) or 0),
    }
    return summary