"""Leakage-safe preprocessing. Everything learned (log scaling, standardising,
one-hot categories) is fitted inside a Pipeline on training data only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PAY = [f"PAY_{k}" for k in range(1, 7)]
BILL = [f"BILL_AMT{k}" for k in range(1, 7)]
PAYAMT = [f"PAY_AMT{k}" for k in range(1, 7)]
AMOUNTS = ["LIMIT_BAL"] + BILL + PAYAMT


def clean(df):
    """Row-by-row fixes; nothing here is learned from the data."""
    df = df.copy()
    # undocumented codes -> the documented "others" category
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    # repayment status: months late (0 if not late) + flags for the -2 / -1 codes
    for c in PAY:
        df[f"{c}_late"] = df[c].clip(lower=0)
        df[f"{c}_m2"] = (df[c] == -2).astype(int)
        df[f"{c}_m1"] = (df[c] == -1).astype(int)
    return df.drop(columns=PAY)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(include_sensitive=True):
    late = [f"{c}_late" for c in PAY]
    flags = [f"{c}_m2" for c in PAY] + [f"{c}_m1" for c in PAY]
    cat = ["EDUCATION"] + (["SEX", "MARRIAGE"] if include_sensitive else [])
    numeric = late + (["AGE"] if include_sensitive else [])

    columns = ColumnTransformer([
        ("amounts", Pipeline([("log", FunctionTransformer(signed_log)),
                              ("scale", StandardScaler())]), AMOUNTS),
        ("numeric", StandardScaler(), numeric),
        ("flags", "passthrough", flags),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ], remainder="drop")

    return Pipeline([("clean", FunctionTransformer(clean)), ("columns", columns)])