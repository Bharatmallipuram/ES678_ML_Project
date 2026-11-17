"""Deletion audit for the fitted RBF SVM.

Run: python src/deletion.py
The controlled experiment freezes the already-fitted feature map and gamma,
then retrains on all rows except one. This isolates the SVC's dependence on
training examples. It does not claim that a production rebuild can retain the
old scaler: a refit scaler is itself data-derived and can change predictions.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

from data import load_data, make_split


def main():
    model = joblib.load("results/models/rbf_svm.joblib")
    train, _, test = make_split(load_data())
    X_train = train.drop(columns="default")
    y_train = train["default"].to_numpy()
    X_test = test.drop(columns="default")

    pre = model.named_steps["pre"]
    svc = model.named_steps["clf"]
    Z_train = pre.transform(X_train)
    Z_test = pre.transform(X_test)
    baseline = svc.decision_function(Z_test)
    sv_mask = np.zeros(len(train), dtype=bool)
    sv_mask[svc.support_] = True
    sv_idx = np.flatnonzero(sv_mask)
    non_sv_idx = np.flatnonzero(~sv_mask)
    if len(non_sv_idx) == 0:
        raise RuntimeError("No non-support vectors; zero-change experiment unavailable")

    # Fixed deterministic examples, deliberately drawn from both groups.
    delete_non_sv = int(non_sv_idx[0])
    delete_sv = int(sv_idx[0])
    rows = []
    for label, delete_idx in [("non_support_vector", delete_non_sv),
                              ("support_vector", delete_sv)]:
        keep = np.arange(len(train)) != delete_idx
        rebuilt = svc.__class__(**svc.get_params())
        start = time.perf_counter()
        rebuilt.fit(Z_train[keep], y_train[keep])
        elapsed = time.perf_counter() - start
        changed = rebuilt.decision_function(Z_test)
        delta = changed - baseline
        rows.append({
            "deletion_type": label,
            "deleted_train_row_index": int(train.index[delete_idx]),
            "deleted_row_was_support_vector": bool(sv_mask[delete_idx]),
            "n_support_vectors_before": int(svc.n_support_.sum()),
            "n_support_vectors_after": int(rebuilt.n_support_.sum()),
            "test_decision_max_abs_change": float(np.max(np.abs(delta))),
            "test_decision_mean_abs_change": float(np.mean(np.abs(delta))),
            "test_predictions_changed": int(np.count_nonzero((changed >= 0) != (baseline >= 0))),
            "refit_seconds_frozen_preprocessor": elapsed,
        })
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/deletion_audit.csv", index=False)
    print(out.to_string(index=False))
    # Exact score invariance is an ideal optimizer property. libsvm refitting
    # can move scores slightly; the empirical check is unchanged test labels.
    non_sv_label_changes = out.loc[out.deletion_type == "non_support_vector",
                                   "test_predictions_changed"].iloc[0]
    if non_sv_label_changes != 0:
        raise AssertionError("Non-SV deletion changed test-set class predictions")
    print("\nNon-support-vector deletion preserved all test class predictions.")


if __name__ == "__main__":
    main()
