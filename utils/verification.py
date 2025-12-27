# utils/verification.py
from __future__ import annotations

from typing import Optional, Tuple
from datetime import datetime
import pandas as pd


def _parse_date(s) -> pd.Timestamp:
    """Parse a date-like value to a normalized (00:00) pandas Timestamp."""
    return pd.to_datetime(s, errors="coerce").normalize()


def _detect_date_and_price_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Heuristically find a date column and a price column.
    Returns (date_col, price_col). Raises ValueError if not found.
    """
    # --- date col
    candidates_date = ["Date", "Datetime", "Timestamp", "date", "DATE", "datetime", "timestamp"]
    date_col = next((c for c in candidates_date if c in df.columns), None)
    if date_col is None:
        # very robust fallback: first column with at least 50% parsable datetimes
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    date_col = c
                    break
            except Exception:
                continue
    if date_col is None:
        raise ValueError("No date-like column found (tried Date/Datetime/Timestamp).")

    # --- price col
    candidates_price = ["Close", "Adj Close", "Adj_Close", "ClosePrice", "close", "adj_close"]
    price_col = next((c for c in candidates_price if c in df.columns), None)
    if price_col is None:
        # fallback: last numeric column
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        price_col = numeric_cols[-1] if numeric_cols else None
    if price_col is None:
        raise ValueError("No price column found (looked for Close/Adj Close/etc.).")

    return date_col, price_col


def next_trading_date(all_dates: pd.Series, last_known_date: str) -> Optional[pd.Timestamp]:
    """
    Given a sorted, normalized Series of trading dates, return the first trading date
    strictly greater than last_known_date. Returns None if not available.
    """
    if all_dates.empty:
        return None
    lk = _parse_date(last_known_date)
    fut = all_dates[all_dates > lk]
    return pd.to_datetime(fut.min()) if not fut.empty else None


def first_available_on_or_after(all_dates: pd.Series, target: str) -> Optional[pd.Timestamp]:
    """
    Be robust to weekends/holidays: pick the first available date >= target.
    Returns None if none exist.
    """
    if all_dates.empty:
        return None
    td = _parse_date(target)
    cand = all_dates[all_dates >= td]
    return pd.to_datetime(cand.min()) if not cand.empty else None


def verify_next_day_from_csv(csv_path: str, target_date: str, price_col: str = "Close"):
    """
    Load the CSV, find the actual close for the first trading day >= target_date,
    and return (actual_date_iso, actual_close).

    Parameters
    ----------
    csv_path : str
        Path to historical OHLCV CSV (must include a date column).
    target_date : str
        The expected next trading day (ISO or any pandas-parsable date).
    price_col : str
        Preferred price column (falls back automatically if missing).

    Returns
    -------
    (actual_date_iso, actual_close) or (None, None)
    """
    df = pd.read_csv(csv_path)

    # Detect columns
    dcol, detected_price = _detect_date_and_price_cols(df)
    if price_col not in df.columns:
        price_col = detected_price

    # Sort & normalize dates
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    df["__DATE__"] = df[dcol].dt.normalize()

    all_dates = df["__DATE__"]
    tgt = first_available_on_or_after(all_dates, target_date)
    if tgt is None:
        return None, None

    row = df.loc[all_dates == tgt]
    if row.empty:
        return None, None

    return tgt.date().isoformat(), float(row.iloc[0][price_col])


def evaluate_hit(pred_price: float, actual_price: float) -> dict:
    """
    Return a verdict bucket plus error stats for a regression (price) prediction.
    Buckets:
      - Within ±1% (Excellent)
      - Within ±2% (Good)
      - Within ±5% (OK)
      - Miss
    """
    if actual_price is None:
        return {"verdict": "Unknown (no actual)", "abs_error": None, "pct_error": None}

    abs_err = abs(actual_price - float(pred_price))
    denom = actual_price if actual_price != 0 else 1e-12
    pct_err = abs_err / denom * 100.0

    if pct_err <= 1.0:
        verdict = "Within ±1% (Excellent)"
    elif pct_err <= 2.0:
        verdict = "Within ±2% (Good)"
    elif pct_err <= 5.0:
        verdict = "Within ±5% (OK)"
    else:
        verdict = "Miss"

    return {
        "verdict": verdict,
        "abs_error": abs_err,
        "pct_error": pct_err
    }


def direction_correct(last_close: float, actual_close: float, pred_direction: str | None) -> Optional[dict]:
    """
    Evaluate directional correctness for a classification (Up/Down) prediction.
    Returns dict with true_direction and boolean direction_correct, or None if pred_direction missing.
    """
    if pred_direction is None:
        return None
    pred_direction = str(pred_direction).strip().capitalize()

    if actual_close > last_close:
        true_dir = "Up"
    elif actual_close < last_close:
        true_dir = "Down"
    else:
        true_dir = "Flat"  # no change; treat as neither Up nor Down

    return {"true_direction": true_dir, "direction_correct": (pred_direction == true_dir)}
