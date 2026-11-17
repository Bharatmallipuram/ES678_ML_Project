"""Phase 2: baselines, evaluated on the VALIDATION set (test set untouched).
Run from the project folder:  python src/baselines.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data import load_data, make_split, SEED
from features import make_preprocessor
from metrics import evaluate, expected_cost, C_FN

os.makedirs("results", exist_ok=True)

train, val, test = make_split(load_data())
del test                                          # never used before the final run
X_tr, y_tr = train.drop(columns="default"), train["default"].values
X_va, y_va = val.drop(columns="default"), val["default"].values

rows = []

# Reference costs of the two trivial decisions
rows.append({"model": "flag nobody", "cost_per_customer": C_FN * y_va.mean()})
rows.append({"model": "flag everybody", "cost_per_customer": expected_cost(y_va, np.ones(len(y_va)), 0.5)})

# 1. Majority / prior baseline
majority = Pipeline([("pre", make_preprocessor()), ("clf", DummyClassifier(strategy="prior"))])
majority.fit(X_tr, y_tr)
rows.append({"model": "majority (prior)", **evaluate(y_va, majority.predict_proba(X_va)[:, 1])})

# 2. Logistic regression, C tuned by 5-fold CV inside the training set only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for sensitive in [True, False]:
    pipe = Pipeline([("pre", make_preprocessor(include_sensitive=sensitive)),
                     ("clf", LogisticRegression(max_iter=5000))])
    search = GridSearchCV(pipe, {"clf__C": np.logspace(-3, 2, 6)},
                          scoring="average_precision", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    prob = search.predict_proba(X_va)[:, 1]
    rows.append({"model": f"logistic (sensitive={'yes' if sensitive else 'no'})",
                 "best_C": search.best_params_["clf__C"],
                 "cv_pr_auc": search.best_score_,
                 **evaluate(y_va, prob)})

results = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(results.round(4).to_string(index=False))
results.to_csv("results/baselines_val.csv", index=False)
print("\nSaved results/baselines_val.csv")