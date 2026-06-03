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
    result = model.predict_proba(dfInput)

    print("Execution time:", time.perf_counter() - start_time, "seconds")
    return result
