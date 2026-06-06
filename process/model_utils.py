"""Helper imported by the embedded Python interpreter in `dt_pipeline infer`.

Loads a trained scikit-learn / xgboost MultiOutputClassifier from a .pkl file
and returns per-class probabilities for a feature matrix. Keep this file next to
the dt_pipeline executable (the build copies it automatically) or point at its
directory with `--scripts`.

Runtime requirements: numpy, pandas, joblib, and whatever the saved model needs
(e.g. scikit-learn / xgboost).
"""

import time

import numpy as np
import pandas as pd
import joblib


def _predict_proba_from_segment_blob(blob: dict, dfInput: pd.DataFrame) -> list:
    estimator = blob["estimator"]
    X = dfInput.to_numpy(dtype=np.float32)
    col_mean = blob.get("col_mean")
    if col_mean is not None:
        col_mean = np.asarray(col_mean, dtype=np.float32)
        bad = ~np.isfinite(X)
        if bad.any():
            X = X.copy()
            X[bad] = np.take(col_mean, np.where(bad)[1])

    proba = estimator.predict_proba(X)
    num_classes = int(blob.get("num_classes", 1))
    full = np.zeros((X.shape[0], num_classes + 1), dtype=np.float32)
    for col, cls in enumerate(estimator.classes_):
        full[:, int(cls)] = proba[:, col]
    return [full]


def _as_process_outputs(result) -> list:
    # sklearn MultiOutputClassifier already returns one array per binary target.
    if isinstance(result, list):
        return result
    # A single binary classifier is valid when dt_pipeline is run with one class.
    if isinstance(result, np.ndarray):
        return [result.astype(np.float32, copy=False)]
    return result


def execModel(mdlPath: str, aryInput: np.ndarray, labels: list) -> np.ndarray:
    start_time = time.perf_counter()

    dfInput = pd.DataFrame(aryInput, columns=labels)
    model = joblib.load(mdlPath)
    result = model.predict(dfInput)

    print("Execution time:", time.perf_counter() - start_time, "seconds")
    return result


def execProb(mdlPath: str, aryInput: np.ndarray, labels: list) -> np.ndarray:
    start_time = time.perf_counter()

    dfInput = pd.DataFrame(aryInput, columns=labels)
    model = joblib.load(mdlPath)
    if isinstance(model, dict) and "estimator" in model:
        result = _predict_proba_from_segment_blob(model, dfInput)
    else:
        result = _as_process_outputs(model.predict_proba(dfInput))

    print("Execution time:", time.perf_counter() - start_time, "seconds")
    return result
