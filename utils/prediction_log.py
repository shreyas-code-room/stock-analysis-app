# utils/prediction_log.py
from __future__ import annotations

import os
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

# Where to store one-line JSON records of predictions
_LOG_PATH = os.path.join("artifacts", "predictions_log.jsonl")
_LOCK = threading.Lock()

# Ensure directory exists
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)


def log_prediction(entry: Dict[str, Any]) -> None:
    """
    Append a single JSON line recording a 'next-day' prediction.
    Recommended fields:
      {
        "ts": "2025-10-12T16:11:23",
        "file_path": "uploads/RELIANCE.csv",
        "ticker": "RELIANCE",
        "model": "lstm",                # one of your model keys
        "time_step": 60,
        "last_known_date": "2025-10-10",
        "target_date": "2025-10-13",
        "pred_price": 2751.34,          # for regressors
        "pred_direction": "Up",         # for logistic
        "pred_proba_up": 0.63
      }
    Any missing fields are allowed; verification will use what it can.
    """
    row = dict(entry or {})
    row.setdefault("ts", datetime.utcnow().isoformat(timespec="seconds"))

    with _LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_logs() -> List[Dict[str, Any]]:
    """Load all JSONL records as a list (empty if none)."""
    if not os.path.exists(_LOG_PATH):
        return []
    out: List[Dict[str, Any]] = []
    with open(_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip malformed line
                continue
    return out


def latest_for(file_path: str, model: str, time_step: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Return the newest log row for a given (file_path, model[, time_step]).
    If time_step is None, returns the latest regardless of time_step.
    """
    rows = load_logs()
    cand = [r for r in rows if r.get("file_path") == file_path and str(r.get("model","")).lower() == str(model).lower()]
    if time_step is not None:
        cand = [r for r in cand if int(r.get("time_step", -1)) == int(time_step)]
    if not cand:
        return None

    def _key(r: Dict[str, Any]):
        # Prefer timestamp ordering; fallback to natural order
        try:
            return r.get("ts") or ""
        except Exception:
            return ""
    cand.sort(key=_key, reverse=True)
    return cand[0]
