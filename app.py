from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import requests
import logging
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import numpy as np

# --- verification helpers & logging
from utils.prediction_log import log_prediction, latest_for
from utils.verification import (
    next_trading_date,
    verify_next_day_from_csv,
    evaluate_hit,
    direction_correct
)

# =========================
# Import project modules
# =========================
# LSTM (Keras) – regression (price)
from models.lstm_model import train_and_predict as lstm_train_and_predict
# Classical models (pure sklearn/xgboost) – regression + classification
from models.classical_models import (
    train_and_predict_linear_regression,
    train_and_predict_random_forest,
    train_and_predict_svm_model,
    train_and_predict_xgboost_model,
    train_and_predict_logistic_regression
)

from utils.evaluation import (
    _align_truth_preds_from_graph,
    regression_metrics,
    build_naive_from_truth,
    classification_labels_from_truth,
)

# Optional: detect xgboost availability
try:
    import xgboost  # noqa
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# If your project keeps these utilities in a different place, adjust imports accordingly
from utils.data_processing import process_data
from utils.stock_fetcher import fetch_stock_data

# =========================
# App & Config
# =========================
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

log_level = logging.DEBUG if os.getenv("FLASK_ENV") != "production" else logging.INFO
logging.basicConfig(level=log_level)

# =========================
# Model Registry
# =========================
# All fns share signature: (df, socketio=None, retrain=False, time_step=60, ...)
MODEL_REGISTRY = {
    "lstm": {
        "fn": lambda df, retrain, time_step: lstm_train_and_predict(
            df, socketio=socketio, retrain=retrain, time_step=time_step
        ),
        "kind": "regression",
        "needs_xgb": False
    },
    "linear": {
        "fn": lambda df, retrain, time_step: train_and_predict_linear_regression(
            df, retrain=retrain, time_step=time_step
        ),
        "kind": "regression",
        "needs_xgb": False
    },
    "rf": {
        "fn": lambda df, retrain, time_step: train_and_predict_random_forest(
            df, retrain=retrain, time_step=time_step
        ),
        "kind": "regression",
        "needs_xgb": False
    },
    "svm": {
        "fn": lambda df, retrain, time_step: train_and_predict_svm_model(
            df, retrain=retrain, time_step=time_step
        ),
        "kind": "regression",
        "needs_xgb": False
    },
    "xgb": {
        "fn": lambda df, retrain, time_step: train_and_predict_xgboost_model(
            df, retrain=retrain, time_step=time_step
        ),
        "kind": "regression",
        "needs_xgb": True
    },
    "logit": {
        # Logistic Regression – classification (Up/Down + probability)
        "fn": lambda df, retrain, time_step: train_and_predict_logistic_regression(
            df, retrain=retrain, time_step=time_step
        ),
        "kind": "classification",
        "needs_xgb": False
    },
}

# =========================
# Helpers
# =========================
def read_csv_safe(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError("CSV file path is missing or invalid.")
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

def _get_model_choice():
    """Read model from form/json; default to 'lstm'."""
    model = (request.form.get("model") or
             (request.json.get("model") if request.is_json else None) or
             "lstm").strip().lower()
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Use one of: {list(MODEL_REGISTRY.keys())}")
    if MODEL_REGISTRY[model]["needs_xgb"] and not _HAS_XGB:
        raise ImportError("xgboost not installed. Install with: pip install xgboost")
    return model

def _get_bool(name: str, default=False):
    raw = (request.form.get(name) or
           (request.json.get(name) if request.is_json else None))
    if raw is None:
        return default
    return str(raw).lower() in ("1", "true", "yes", "y")

def _get_int(name: str, default: int):
    raw = (request.form.get(name) or
           (request.json.get(name) if request.is_json else None))
    if raw is None:
        return default
    return int(raw)

def _detect_date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).lower().startswith("date"):
            return c
    for c in ("Datetime", "Timestamp"):
        if c in df.columns:
            return c
    return None

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

# Upload CSV (no training here)
@app.route("/uploads", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        df = read_csv_safe(file_path)
        try:
            processed = process_data(df)
        except Exception:
            processed = df
        preview = processed.head(10)
        meta = {"rows": int(len(processed)), "columns": list(processed.columns)}
        logging.debug("File uploaded & preview prepared.")
        return jsonify({
            "message": "File uploaded successfully",
            "file_path": file_path,
            "preview": preview.to_json(orient="records"),
            "meta": meta
        })
    except Exception as e:
        logging.exception("Upload processing failed")
        return jsonify({"error": str(e)}), 400

# Download by ticker (no training here)
@app.route("/download", methods=["POST"])
def download_stock():
    ticker = request.form.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    try:
        csv_file = fetch_stock_data(ticker)
        if not csv_file or not os.path.exists(csv_file):
            return jsonify({"error": "Downloaded file not found"}), 404
        _ = read_csv_safe(csv_file)
        logging.debug("Stock data downloaded successfully")
        return jsonify({
            "message": "Stock data downloaded successfully",
            "file_path": csv_file
        })
    except Exception as e:
        logging.exception("Error downloading stock data")
        return jsonify({"error": str(e)}), 500

# List available models (and xgb availability)
@app.route("/available_models", methods=["GET"])
def available_models():
    models = []
    for k, v in MODEL_REGISTRY.items():
        if v["needs_xgb"] and not _HAS_XGB:
            status = "unavailable (install xgboost)"
        else:
            status = "available"
        models.append({"key": k, "type": v["kind"], "status": status})
    return jsonify({"models": models})

# Train (retrain=True to force training)
@app.route("/train", methods=["POST"])
def train_model_route():
    """
    Params (form or JSON):
      - file_path: str (required)
      - model: one of ['lstm','linear','rf','svm','xgb','logit'] (default 'lstm')
      - time_step: int (default 60)
      - retrain: bool (default True for /train)
    """
    file_path = request.form.get("file_path") or (request.json.get("file_path") if request.is_json else None)
    try:
        model_key = _get_model_choice()
        time_step = _get_int("time_step", 60)
        retrain = _get_bool("retrain", True)  # train endpoint defaults to retrain

        df = read_csv_safe(file_path)
        fn = MODEL_REGISTRY[model_key]["fn"]
        predictions, graph_data = fn(df, retrain, time_step)

        # --- LOG PREDICTION so /verify-latest works later ---
        try:
            dcol = _detect_date_col(df)
            if dcol:
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                last_known_date = df[dcol].dropna().max().date().isoformat()
                tgt = next_trading_date(df[dcol].dt.normalize(), last_known_date)
                target_date = tgt.date().isoformat() if tgt is not None else None
            else:
                last_known_date, target_date = None, None

            entry = {
                "file_path": file_path,
                "model": model_key,
                "time_step": int(time_step),
                "last_known_date": last_known_date,
                "target_date": target_date,
            }
            if model_key != "logit":
                entry["pred_price"] = float(predictions.get("next_day", np.nan))
            else:
                entry["pred_direction"] = predictions.get("next_day_direction")
                entry["pred_proba_up"] = float(predictions.get("next_day_proba", np.nan))

            log_prediction(entry)
            app.logger.debug(f"[log_prediction] {entry}")
        except Exception as _e:
            app.logger.warning(f"Prediction log failed (train): {_e}")
        # ---------------------------------------------------

        logging.info(f"Model '{model_key}' trained and artifacts saved.")
        return jsonify({
            "message": f"Model '{model_key}' trained and saved",
            "model": model_key,
            "predictions": predictions,
            "graph_data": graph_data
        })
    except Exception as e:
        logging.exception("Training error")
        return jsonify({"error": str(e)}), 500

# Predict (fast path; retrain=False)
@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Params (form or JSON):
      - file_path: str (required)
      - model: one of ['lstm','linear','rf','svm','xgb','logit'] (default 'lstm')
      - time_step: int (default 60)
      - retrain: bool (default False for /predict)
    """
    file_path = request.form.get("file_path") or (request.json.get("file_path") if request.is_json else None)
    try:
        model_key = _get_model_choice()
        time_step = _get_int("time_step", 60)
        retrain = _get_bool("retrain", False)  # predict endpoint defaults to fast path

        df = read_csv_safe(file_path)
        fn = MODEL_REGISTRY[model_key]["fn"]
        predictions, graph_data = fn(df, retrain, time_step)

        # --- LOG PREDICTION so /verify-latest works after Predict too ---
        try:
            dcol = _detect_date_col(df)
            if dcol:
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                last_known_date = df[dcol].dropna().max().date().isoformat()
                tgt = next_trading_date(df[dcol].dt.normalize(), last_known_date)
                target_date = tgt.date().isoformat() if tgt is not None else None
            else:
                last_known_date, target_date = None, None

            entry = {
                "file_path": file_path,
                "model": model_key,
                "time_step": int(time_step),
                "last_known_date": last_known_date,
                "target_date": target_date,
            }
            if model_key != "logit":
                entry["pred_price"] = float(predictions.get("next_day", np.nan))
            else:
                entry["pred_direction"] = predictions.get("next_day_direction")
                entry["pred_proba_up"] = float(predictions.get("next_day_proba", np.nan))

            log_prediction(entry)
            app.logger.debug(f"[log_prediction] {entry}")
        except Exception as _e:
            app.logger.warning(f"Prediction log failed (predict): {_e}")
        # ---------------------------------------------------------------

        logging.debug(f"Predictions({model_key}): {predictions}")
        return jsonify({
            "model": model_key,
            "predictions": predictions,
            "graph_data": graph_data
        })
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

# Fetch CSV via URL (no training)
@app.route("/fetch_from_url", methods=["POST"])
def fetch_csv_from_url():
    url = request.form.get("url") or (request.json.get("url") if request.is_json else None)
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "text/csv" not in ctype and "application/octet-stream" not in ctype:
            return jsonify({"error": "URL does not appear to be a CSV (Content-Type)"}), 400
        if len(resp.content) > 10 * 1024 * 1024:
            return jsonify({"error": "File too large (limit 10MB)"}), 400

        filename = secure_filename(url.split("/")[-1] or "downloaded_data.csv")
        if not filename.endswith(".csv"):
            filename += ".csv"

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(resp.content)

        df = read_csv_safe(file_path)
        try:
            processed = process_data(df)
        except Exception:
            processed = df
        preview = processed.head(10)
        meta = {"rows": int(len(processed)), "columns": list(processed.columns)}

        logging.debug("CSV fetched from URL and preview prepared.")
        return jsonify({
            "message": "CSV downloaded from URL",
            "file_path": file_path,
            "preview": preview.to_json(orient="records"),
            "meta": meta
        })
    except Exception as e:
        logging.exception("Error fetching CSV from URL")
        return jsonify({"error": str(e)}), 500

# Evaluate all models consistently
@app.route("/evaluate", methods=["POST"])
def evaluate_all_models():
    """
    Params (form or JSON):
      - file_path: str (required)
      - time_step: int (default 60) -> used for all models for fairness
      - retrain: bool (default False) -> set True to regenerate artifacts for *each* model before eval
      - models: list[str] optional (subset of MODEL_REGISTRY keys); default evaluates all
    Returns:
      JSON with per-model metrics + naïve baseline.
    """
    payload = request.json if request.is_json else request.form
    file_path = payload.get("file_path")
    time_step = int(payload.get("time_step", 60))
    retrain   = str(payload.get("retrain", "false")).lower() in ("1", "true", "yes", "y")

    sel_models = payload.get("models")
    if isinstance(sel_models, str):
        sel_models = [m.strip() for m in sel_models.split(",") if m.strip()]
    if not sel_models:
        sel_models = list(MODEL_REGISTRY.keys())

    try:
        df = read_csv_safe(file_path)
    except Exception as e:
        return jsonify({"error": f"Bad CSV: {e}"}), 400

    results = {}
    truth_ref = None
    test_start_idx_ref = None
    test_len_ref = None
    ts_ref = time_step

    for key in sel_models:
        if key not in MODEL_REGISTRY:
            results[key] = {"error": f"Unknown model '{key}'"}
            continue
        if MODEL_REGISTRY[key]["needs_xgb"] and not _HAS_XGB:
            results[key] = {"error": "xgboost not installed"}
            continue

        fn = MODEL_REGISTRY[key]["fn"]
        try:
            preds_obj, graph_data = fn(df, retrain=retrain, time_step=time_step)
        except Exception as e:
            results[key] = {"error": f"Run failed: {e}"}
            continue

        # Align y_true_test and y_pred_test
        try:
            y_true_test, y_pred_test, test_start_idx = _align_truth_preds_from_graph(graph_data)
        except Exception as e:
            results[key] = {"error": f"Alignment failed: {e}"}
            continue

        # Cache for naïve baseline
        if truth_ref is None:
            truth_ref = graph_data["true_prices"]
            test_start_idx_ref = test_start_idx
            test_len_ref = len(y_pred_test)
            ts_ref = int(graph_data["time_step"])

        if MODEL_REGISTRY[key]["kind"] == "regression":
            mets = regression_metrics(y_true_test, y_pred_test)
            results[key] = {
                "type": "regression",
                "metrics": mets,
                "next_day": float(preds_obj.get("next_day", float("nan"))),
                "graph_meta": {"time_step": int(graph_data["time_step"]),
                               "test_points": int(len(y_pred_test))}
            }
        else:
            # classification (logistic) — evaluate on test segment using probabilities
            y_true_cls = classification_labels_from_truth(
                graph_data["true_prices"],
                graph_data["time_step"],
                len(graph_data["train_predict"]),
                len(graph_data["test_predict"]),
            )
            y_proba_test = y_pred_test
            # mets = classification_metrics(y_true_cls, y_proba_test)
            results[key] = {
                "type": "classification",
                "metrics": mets,
                "next_day_direction": preds_obj.get("next_day_direction"),
                "next_day_proba": float(preds_obj.get("next_day_proba", float("nan"))),
                "graph_meta": {"time_step": int(graph_data["time_step"]),
                               "test_points": int(len(y_proba_test))}
            }

    # Naïve baseline on the same test span
    if truth_ref is not None and test_start_idx_ref is not None and test_len_ref is not None:
        y_true_test_all = np.asarray(truth_ref, dtype=float)[test_start_idx_ref:test_start_idx_ref+test_len_ref]
        y_naive = build_naive_from_truth(truth_ref, ts_ref, test_start_idx_ref, test_len_ref)
        results["naive"] = {
            "type": "regression",
            "metrics": regression_metrics(y_true_test_all, y_naive),
            "note": "Baseline ŷ[t] = y[t-1] on the same test window"
        }

    return jsonify({
        "file_path": file_path,
        "time_step": time_step,
        "retrained": bool(retrain),
        "evaluated_models": sel_models,
        "results": results
    })

# Verify the most recent next-day prediction against actual close
@app.route("/verify-latest", methods=["POST"])
def verify_latest():
    """
    Verify the most recent next-day prediction for a given file & model.
    Checks your logged predictions against the updated CSV's actual close.
    """
    import pandas as pd
    payload = request.json if request.is_json else request.form
    file_path = payload.get("file_path")
    model = (payload.get("model") or "").strip().lower()
    time_step = payload.get("time_step")
    price_col = payload.get("price_col") or "Close"

    if not file_path or model not in {"lstm","linear","rf","svm","xgb","logit"}:
        return jsonify({"error":"file_path and a valid model are required"}), 400

    ts_int = int(time_step) if time_step is not None else None
    row = latest_for(file_path, model, ts_int)
    if not row:
        return jsonify({"error":"No logged prediction found for this file/model/time_step"}), 404

    target_date = row.get("target_date") or row.get("last_known_date")
    if not target_date:
        return jsonify({"error":"No target_date/last_known_date in log entry"}), 400

    # Load CSV and get actual close on or after target_date
    actual_date, actual_close = verify_next_day_from_csv(file_path, target_date, price_col=price_col)
    if actual_date is None or actual_close is None:
        return jsonify({
            "status":"PENDING",
            "message":"No actual close available yet (holiday or data not updated).",
            "target_date": target_date
        }), 200

    out = {
        "file_path": file_path,
        "model": model,
        "time_step": row.get("time_step"),
        "logged_at": row.get("ts"),
        "last_known_date": row.get("last_known_date"),
        "target_date": target_date,
        "actual_date_used": actual_date,
        "actual_close": actual_close,
        "status": "OK"
    }

    if model != "logit":
        pred_price = float(row.get("pred_price", "nan"))
        verdict = evaluate_hit(pred_price, actual_close)
        out.update({
            "pred_price": pred_price,
            "verdict": verdict["verdict"],
            "abs_error": verdict["abs_error"],
            "pct_error": verdict["pct_error"]
        })
    else:
        # For Logistic, check direction correctness
        df = pd.read_csv(file_path)
        dcol = _detect_date_col(df)
        if not dcol:
            return jsonify({"error":"Could not locate date column to compute last close."}), 500
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
        last_known = pd.to_datetime(row.get("last_known_date"))
        slice_last = df[df[dcol].dt.normalize() == last_known]
        if slice_last.empty:
            slice_last = df[df[dcol] <= last_known].tail(1)
        price_col_local = price_col if price_col in df.columns else ("Adj Close" if "Adj Close" in df.columns else df.columns[-1])
        last_close = float(slice_last.iloc[0][price_col_local])
        info = direction_correct(last_close, actual_close, row.get("pred_direction"))
        out.update({
            "pred_direction": row.get("pred_direction"),
            "pred_proba_up": row.get("pred_proba_up"),
            "true_direction": info["true_direction"] if info else None,
            "direction_correct": info["direction_correct"] if info else None
        })

    return jsonify(out), 200

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Disable reloader so file writes don't restart mid-train
    socketio.run(app, debug=False, use_reloader=False)


# from flask import Flask, render_template, request, jsonify
# import os
# import pandas as pd
# import requests
# import logging
# from flask_socketio import SocketIO
# from werkzeug.utils import secure_filename
# import numpy as np  # <-- add this

# from utils.prediction_log import log_prediction, latest_for
# from utils.verification import (
#     next_trading_date,
#     verify_next_day_from_csv,
#     evaluate_hit,
#     direction_correct
# )

# # =========================
# # Import project modules
# # =========================
# # LSTM (Keras) – regression (price)
# from models.lstm_model import train_and_predict as lstm_train_and_predict
# # Classical models (pure sklearn/xgboost) – regression + classification
# from models.classical_models import (
#     train_and_predict_linear_regression,
#     train_and_predict_random_forest,
#     train_and_predict_svm_model,
#     train_and_predict_xgboost_model,
#     train_and_predict_logistic_regression
# )





# # Optional: detect xgboost availability for /available_models
# try:
#     import xgboost  # noqa
#     _HAS_XGB = True
# except Exception:
#     _HAS_XGB = False

# # Utility helpers you already have
# from utils.data_processing import process_data
# from utils.stock_fetcher import fetch_stock_data

# # =========================
# # App & Config
# # =========================
# app = Flask(__name__)
# socketio = SocketIO(app, async_mode="threading")

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# log_level = logging.DEBUG if os.getenv("FLASK_ENV") != "production" else logging.INFO
# logging.basicConfig(level=log_level)

# # =========================
# # Model Registry
# # =========================
# # Each entry gives a callable and a type for shaping UI expectations.
# # All fns share signature: (df, socketio=None, retrain=False, time_step=60, ...)
# MODEL_REGISTRY = {
#     "lstm": {
#         "fn": lambda df, retrain, time_step: lstm_train_and_predict(
#             df, socketio=socketio, retrain=retrain, time_step=time_step
#         ),
#         "kind": "regression",
#         "needs_xgb": False
#     },
#     "linear": {
#         "fn": lambda df, retrain, time_step: train_and_predict_linear_regression(
#             df, retrain=retrain, time_step=time_step
#         ),
#         "kind": "regression",
#         "needs_xgb": False
#     },
#     "rf": {
#         "fn": lambda df, retrain, time_step: train_and_predict_random_forest(
#             df, retrain=retrain, time_step=time_step
#         ),
#         "kind": "regression",
#         "needs_xgb": False
#     },
#     "svm": {
#         "fn": lambda df, retrain, time_step: train_and_predict_svm_model(
#             df, retrain=retrain, time_step=time_step
#         ),
#         "kind": "regression",
#         "needs_xgb": False
#     },
#     "xgb": {
#         "fn": lambda df, retrain, time_step: train_and_predict_xgboost_model(
#             df, retrain=retrain, time_step=time_step
#         ),
#         "kind": "regression",
#         "needs_xgb": True
#     },
#     "logit": {
#         # Logistic Regression – classification (Up/Down + probability)
#         "fn": lambda df, retrain, time_step: train_and_predict_logistic_regression(
#             df, retrain=retrain, time_step=time_step
#         ),
#         "kind": "classification",
#         "needs_xgb": False
#     },
# }

# # =========================
# # Helpers
# # =========================
# def read_csv_safe(path: str) -> pd.DataFrame:
#     if not path or not os.path.exists(path):
#         raise FileNotFoundError("CSV file path is missing or invalid.")
#     try:
#         df = pd.read_csv(path)
#         if df.empty:
#             raise ValueError("CSV is empty.")
#         return df
#     except Exception as e:
#         raise ValueError(f"Failed to read CSV: {e}")

# def _get_model_choice():
#     """Read model from form/json; default to 'lstm'."""
#     model = (request.form.get("model") or
#              (request.json.get("model") if request.is_json else None) or
#              "lstm").strip().lower()
#     if model not in MODEL_REGISTRY:
#         raise ValueError(f"Unknown model '{model}'. Use one of: {list(MODEL_REGISTRY.keys())}")
#     if MODEL_REGISTRY[model]["needs_xgb"] and not _HAS_XGB:
#         raise ImportError("xgboost not installed. Install with: pip install xgboost")
#     return model

# def _get_bool(name: str, default=False):
#     raw = (request.form.get(name) or
#            (request.json.get(name) if request.is_json else None))
#     if raw is None:
#         return default
#     return str(raw).lower() in ("1", "true", "yes", "y")

# def _get_int(name: str, default: int):
#     raw = (request.form.get(name) or
#            (request.json.get(name) if request.is_json else None))
#     if raw is None:
#         return default
#     return int(raw)

# # =========================
# # Routes (existing)
# # =========================
# @app.route("/")
# def index():
#     return render_template("index.html")

# # Upload CSV (no training here)
# @app.route("/uploads", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(file_path)

#     try:
#         df = read_csv_safe(file_path)
#         try:
#             processed = process_data(df)
#         except Exception:
#             processed = df
#         preview = processed.head(10)
#         meta = {"rows": int(len(processed)), "columns": list(processed.columns)}
#         logging.debug("File uploaded & preview prepared.")
#         return jsonify({
#             "message": "File uploaded successfully",
#             "file_path": file_path,
#             "preview": preview.to_json(orient="records"),
#             "meta": meta
#         })
#     except Exception as e:
#         logging.exception("Upload processing failed")
#         return jsonify({"error": str(e)}), 400

# # Download by ticker (no training here)
# @app.route("/download", methods=["POST"])
# def download_stock():
#     ticker = request.form.get("ticker")
#     if not ticker:
#         return jsonify({"error": "Ticker is required"}), 400
#     try:
#         csv_file = fetch_stock_data(ticker)
#         if not csv_file or not os.path.exists(csv_file):
#             return jsonify({"error": "Downloaded file not found"}), 404
#         _ = read_csv_safe(csv_file)
#         logging.debug("Stock data downloaded successfully")
#         return jsonify({
#             "message": "Stock data downloaded successfully",
#             "file_path": csv_file
#         })
#     except Exception as e:
#         logging.exception("Error downloading stock data")
#         return jsonify({"error": str(e)}), 500

# # =========================
# # New: list available models
# # =========================
# @app.route("/available_models", methods=["GET"])
# def available_models():
#     models = []
#     for k, v in MODEL_REGISTRY.items():
#         if v["needs_xgb"] and not _HAS_XGB:
#             status = "unavailable (install xgboost)"
#         else:
#             status = "available"
#         models.append({"key": k, "type": v["kind"], "status": status})
#     return jsonify({"models": models})

# # =========================
# # Train (retrain=True to force training)
# # =========================
# @app.route("/train", methods=["POST"])
# def train_model_route():
#     """
#     Params (form or JSON):
#       - file_path: str (required)
#       - model: one of ['lstm','linear','rf','svm','xgb','logit'] (default 'lstm')
#       - time_step: int (default 60)
#       - retrain: bool (default True for /train)
#     """
#     file_path = request.form.get("file_path") or (request.json.get("file_path") if request.is_json else None)
#     try:
#         model_key = _get_model_choice()
#         time_step = _get_int("time_step", 60)
#         retrain = _get_bool("retrain", True)  # train endpoint defaults to retrain

#         df = read_csv_safe(file_path)
#         fn = MODEL_REGISTRY[model_key]["fn"]
#         predictions, graph_data = fn(df, retrain, time_step)

#         logging.info(f"Model '{model_key}' trained and artifacts saved.")
#         return jsonify({
#             "message": f"Model '{model_key}' trained and saved",
#             "model": model_key,
#             "predictions": predictions,
#             "graph_data": graph_data
#         })
#     except Exception as e:
#         logging.exception("Training error")
#         return jsonify({"error": str(e)}), 500

# # =========================
# # Predict (fast path; retrain=False)
# # =========================
# @app.route("/predict", methods=["POST"])
# def predict_route():
#     """
#     Params (form or JSON):
#       - file_path: str (required)
#       - model: one of ['lstm','linear','rf','svm','xgb','logit'] (default 'lstm')
#       - time_step: int (default 60)
#       - retrain: bool (default False for /predict)
#     """
#     file_path = request.form.get("file_path") or (request.json.get("file_path") if request.is_json else None)
#     try:
#         model_key = _get_model_choice()
#         time_step = _get_int("time_step", 60)
#         retrain = _get_bool("retrain", False)  # predict endpoint defaults to fast path

#         df = read_csv_safe(file_path)
#         fn = MODEL_REGISTRY[model_key]["fn"]
#         predictions, graph_data = fn(df, retrain, time_step)

#         logging.debug(f"Predictions({model_key}): {predictions}")
#         return jsonify({
#             "model": model_key,
#             "predictions": predictions,
#             "graph_data": graph_data
#         })
#     except Exception as e:
#         logging.exception("Prediction error")
#         return jsonify({"error": str(e)}), 500

# # =========================
# # Fetch CSV via URL (no training)
# # =========================
# @app.route("/fetch_from_url", methods=["POST"])
# def fetch_csv_from_url():
#     url = request.form.get("url") or (request.json.get("url") if request.is_json else None)
#     if not url:
#         return jsonify({"error": "No URL provided"}), 400

#     try:
#         resp = requests.get(url, timeout=10)
#         resp.raise_for_status()
#         ctype = resp.headers.get("Content-Type", "")
#         if "text/csv" not in ctype and "application/octet-stream" not in ctype:
#             return jsonify({"error": "URL does not appear to be a CSV (Content-Type)"}), 400
#         if len(resp.content) > 10 * 1024 * 1024:
#             return jsonify({"error": "File too large (limit 10MB)"}), 400

#         filename = secure_filename(url.split("/")[-1] or "downloaded_data.csv")
#         if not filename.endswith(".csv"):
#             filename += ".csv"

#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         with open(file_path, "wb") as f:
#             f.write(resp.content)

#         df = read_csv_safe(file_path)
#         try:
#             processed = process_data(df)
#         except Exception:
#             processed = df
#         preview = processed.head(10)
#         meta = {"rows": int(len(processed)), "columns": list(processed.columns)}

#         logging.debug("CSV fetched from URL and preview prepared.")
#         return jsonify({
#             "message": "CSV downloaded from URL",
#             "file_path": file_path,
#             "preview": preview.to_json(orient="records"),
#             "meta": meta
#         })
#     except Exception as e:
#         logging.exception("Error fetching CSV from URL")
#         return jsonify({"error": str(e)}), 500
    


# # =========================
# # Main
# # =========================
# if __name__ == "__main__":
#     # Disable reloader so file writes don't restart mid-train
#     socketio.run(app, debug=False, use_reloader=False)

