"""Phase 3: tune linear SVM, RBF SVM and gradient boosting with CV on the
training set; compare on VALIDATION ranking metrics; save fitted models.
Run from the project folder:  python src/models.py
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"   # set BEFORE importing sklearn/joblib

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from data import SEED, load_data, make_split
from features import make_preprocessor
from metrics import average_precision_score, recall_at_fpr, roc_auc_score

os.makedirs("results/models", exist_ok=True)
SVM_TUNE_SIZE = 6000            # RBF SVM is tuned on a subsample (training cost ~ n^2)

train, val, test = make_split(load_data())
del test
X_tr, y_tr = train.drop(columns="default"), train["default"].values
X_va, y_va = val.drop(columns="default"), val["default"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# RBF grid along the diagonal C = C_tilde / (2*gamma) (Keerthi & Lin, 2003),
# so small gamma is paired with a large enough C to reach the linear limit.
rbf_grid = []
for g in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    Cs = [ct / (2 * g) for ct in (0.1, 1, 10)]
    rbf_grid.append({"clf__gamma": [g],
                     "clf__C": [round(c, 4) for c in Cs if c <= 20000]})

candidates = {
    "linear_svm": (
        LinearSVC(max_iter=20000, random_state=SEED),
        {"clf__C": np.logspace(-3, 2, 6)},
        False),                                   # tune on full training set
    "rbf_svm": (
        SVC(kernel="rbf", cache_size=1000),
        rbf_grid,
        True),                                    # tune on subsample
    "grad_boost": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"clf__learning_rate": [0.01, 0.03, 0.1],
         "clf__max_depth": [3, None],
         "clf__max_iter": [300, 600]},
        False),
}

rows, best_params = [], {}
for name, (clf, grid, subsample) in candidates.items():
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])

    X_tune, y_tune = X_tr, y_tr
    if subsample:
        X_tune, _, y_tune, _ = train_test_split(
            X_tr, y_tr, train_size=SVM_TUNE_SIZE, stratify=y_tr, random_state=SEED)

    t0 = time.time()
    search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
    search.fit(X_tune, y_tune)
    tune_time = time.time() - t0
    pd.DataFrame(search.cv_results_).to_csv(f"results/cv_{name}.csv", index=False)

    # refit the best settings on the FULL training set
    final = clone(pipe).set_params(**search.best_params_)
    t0 = time.time()
    final.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    scores = (final.decision_function(X_va) if hasattr(final, "decision_function")
              else final.predict_proba(X_va)[:, 1])
    row = {"model": name,
           "cv_pr_auc": search.best_score_,
           "val_pr_auc": average_precision_score(y_va, scores),
           "val_roc_auc": roc_auc_score(y_va, scores),
           "val_recall_at_10fpr": recall_at_fpr(y_va, scores),
           "tune_time_s": tune_time,
           "full_fit_time_s": fit_time}
    if name == "rbf_svm":
        n_sv = final.named_steps["clf"].n_support_.sum()
        row["n_support_vectors"] = n_sv
        row["sv_fraction"] = n_sv / len(y_tr)
    rows.append(row)

    best_params[name] = {k: (int(v) if isinstance(v, (int, np.integer))
                             else float(v) if isinstance(v, (float, np.floating)) else v)
                         for k, v in search.best_params_.items()}
    joblib.dump(final, f"results/models/{name}.joblib")
    print(f"{name}: best {search.best_params_}, val PR-AUC {row['val_pr_auc']:.4f}, "
          f"full fit {fit_time:.1f}s", flush=True)

results = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n", results.round(4).to_string(index=False))
results.to_csv("results/models_val.csv", index=False)
with open("results/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("\nSaved results/models_val.csv, results/best_params.json, results/cv_*.csv, results/models/")