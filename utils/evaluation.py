# utils/evaluation.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _align_truth_preds_from_graph(graph_data):
    true_prices = np.asarray(graph_data["true_prices"], dtype=float)
    train_pred  = np.asarray(graph_data["train_predict"], dtype=float)
    test_pred   = np.asarray(graph_data["test_predict"], dtype=float)
    ts          = int(graph_data["time_step"])

    test_start_idx = ts + len(train_pred)
    y_true_test = true_prices[test_start_idx : test_start_idx + len(test_pred)]
    return y_true_test, test_pred, test_start_idx


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else float("nan")
    mae  = float(mean_absolute_error(y_true, y_pred)) if len(y_true) else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100.0 if len(y_true) else float("nan")
        if np.isnan(mape):
            mape = float("nan")
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def build_naive_from_truth(true_prices, time_step, test_start_idx, test_len):
    true_prices = np.asarray(true_prices, dtype=float)
    preds = []
    for i in range(test_start_idx, test_start_idx + test_len):
        prev_idx = i - 1
        if prev_idx < 0:
            preds.append(true_prices[i])
        else:
            preds.append(true_prices[prev_idx])
    return np.array(preds, dtype=float)


def classification_labels_from_truth(true_prices, time_step, train_pred_len, test_pred_len):
    true_prices = np.asarray(true_prices, dtype=float)
    ts = int(time_step)
    start_k = ts + int(train_pred_len)
    end_k   = start_k + int(test_pred_len)

    labels = []
    for k in range(start_k, end_k):
        prev_k = k - 1
        if prev_k < 0:
            labels.append(0)
        else:
            labels.append(1 if true_prices[k] > true_prices[prev_k] else 0)
    return np.array(labels, dtype=int)
