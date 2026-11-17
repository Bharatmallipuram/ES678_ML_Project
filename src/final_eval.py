"""Phase 3b: calibrate on VALIDATION, choose thresholds on VALIDATION,
then evaluate ONCE on the TEST set with bootstrap confidence intervals.
Run from the project folder:  python src/final_eval.py
Do not change any model after running this.
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data import SEED, load_data, make_split
from features import make_preprocessor
from metrics import COST_THRESHOLD, average_precision_score, evaluate, expected_cost

LOGREG_C = 1.0          # chosen by 5-fold CV in baselines.py
N_BOOT = 1000
os.makedirs("results/figures", exist_ok=True)

train, val, test = make_split(load_data())
X_tr, y_tr = train.drop(columns="default"), train["default"].values
X_va, y_va = val.drop(columns="default"), val["default"].values
X_te, y_te = test.drop(columns="default"), test["default"].values

# ---- models: refit logistic (baseline), load the tuned ones ----
logreg = Pipeline([("pre", make_preprocessor()),
                   ("clf", LogisticRegression(C=LOGREG_C, max_iter=5000))]).fit(X_tr, y_tr)
models = {"logistic": logreg}
for name in ["linear_svm", "rbf_svm", "grad_boost"]:
    models[name] = joblib.load(f"results/models/{name}.joblib")


def raw_score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") \
        else model.predict_proba(X)[:, 1]


def platt(scores_val, y_val):
    """Fit a 1-D logistic map from raw score to probability (on validation data)."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(scores_val.reshape(-1, 1), y_val)


def best_threshold(y, prob):
    grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y, prob, t) for t in grid]
    return grid[int(np.argmin(costs))]


def bootstrap(y, probs_by_model, thr_by_model, ref="logistic", n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    n = len(y)
    stats = {m: {"pr_auc": [], "cost": []} for m in probs_by_model}
    diffs = {m: {"pr_auc": [], "cost": []} for m in probs_by_model if m != ref}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        cur = {}
        for m, p in probs_by_model.items():
            cur[m] = (average_precision_score(yb, p[idx]),
                      expected_cost(yb, p[idx], thr_by_model[m]))
            stats[m]["pr_auc"].append(cur[m][0]); stats[m]["cost"].append(cur[m][1])
        for m in diffs:
            diffs[m]["pr_auc"].append(cur[m][0] - cur[ref][0])
            diffs[m]["cost"].append(cur[m][1] - cur[ref][1])
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return ({m: {k: ci(v) for k, v in d.items()} for m, d in stats.items()},
            {m: {k: ci(v) for k, v in d.items()} for m, d in diffs.items()})


rows, test_probs, thresholds, preds = [], {}, {}, {"y_true": y_te}
for name, model in models.items():
    s_va, s_te = raw_score(model, X_va), raw_score(model, X_te)
    cal = platt(s_va, y_va)                                   # calibration: validation only
    p_va = cal.predict_proba(s_va.reshape(-1, 1))[:, 1]
    p_te = cal.predict_proba(s_te.reshape(-1, 1))[:, 1]
    thr_val = best_threshold(y_va, p_va)                      # threshold: validation only

    test_probs[name], thresholds[name], preds[f"p_{name}"] = p_te, COST_THRESHOLD, p_te
    row = {"model": name, **evaluate(y_te, p_te, threshold=COST_THRESHOLD),
           "val_chosen_threshold": thr_val,
           "cost_at_val_threshold": expected_cost(y_te, p_te, thr_val)}
    rows.append(row)

# trivial references on the test set
rows.append({"model": "flag nobody", "cost_per_customer": expected_cost(y_te, np.zeros(len(y_te)), 0.5)})
rows.append({"model": "flag everybody", "cost_per_customer": expected_cost(y_te, np.ones(len(y_te)), 0.5)})

ci, diff = bootstrap(y_te, test_probs, thresholds)
results = pd.DataFrame(rows)
for m in ci:
    i = results["model"] == m
    results.loc[i, "pr_auc_95ci"] = f"[{ci[m]['pr_auc'][0]:.3f}, {ci[m]['pr_auc'][1]:.3f}]"
    results.loc[i, "cost_95ci"] = f"[{ci[m]['cost'][0]:.3f}, {ci[m]['cost'][1]:.3f}]"
for m in diff:
    i = results["model"] == m
    results.loc[i, "pr_auc_minus_logistic_95ci"] = f"[{diff[m]['pr_auc'][0]:+.3f}, {diff[m]['pr_auc'][1]:+.3f}]"
    results.loc[i, "cost_minus_logistic_95ci"] = f"[{diff[m]['cost'][0]:+.3f}, {diff[m]['cost'][1]:+.3f}]"

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(results.round(4).to_string(index=False))
results.to_csv("results/final_test.csv", index=False)
pd.DataFrame(preds).to_csv("results/test_predictions.csv", index=False)

# reliability diagram (test set, calibrated probabilities)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
for name, p in test_probs.items():
    frac, mean_p = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, marker="o", ms=3, label=name)
ax.set_xlabel("Predicted probability of default"); ax.set_ylabel("Observed default rate")
ax.set_title("Reliability diagram (test set)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/figures/05_reliability_test.png", dpi=150); plt.close()
print("\nSaved results/final_test.csv, results/test_predictions.csv, results/figures/05_reliability_test.png")