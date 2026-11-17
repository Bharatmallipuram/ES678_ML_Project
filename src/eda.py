"""Exploratory data analysis on the TRAINING set only.
Run from the project folder:  python src/eda.py
Figures -> results/figures/, tables -> results/eda/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # save figures without opening windows
import matplotlib.pyplot as plt

from data import load_data, make_split

FIG = "results/figures"
TAB = "results/eda"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def class_balance(train):
    counts = train["default"].value_counts().sort_index()
    print("Class counts:\n", counts)
    print(f"Default rate: {train['default'].mean():.3f}")
    print(f"'Always predict no default' accuracy: {1 - train['default'].mean():.3f}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["No default (0)", "Default (1)"], counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Customers")
    ax.set_title("Class balance (training set)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v/len(train):.1%})", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{FIG}/01_class_balance.png", dpi=150)
    plt.close()


def undefined_codes(train):
    print("\nEDUCATION (documented: 1-4):\n", train["EDUCATION"].value_counts().sort_index())
    print("\nMARRIAGE (documented: 1-3):\n", train["MARRIAGE"].value_counts().sort_index())

    pay_cols = [f"PAY_{k}" for k in range(1, 7)]
    pay_counts = (pd.DataFrame({c: train[c].value_counts() for c in pay_cols})
                  .fillna(0).astype(int).sort_index())
    print("\nRepayment status counts (documented: -1, 1-9):\n", pay_counts)
    pay_counts.to_csv(f"{TAB}/pay_status_counts.csv")

    undoc = pd.Series({
        "EDUCATION in {0,5,6}": train["EDUCATION"].isin([0, 5, 6]).sum(),
        "MARRIAGE == 0": (train["MARRIAGE"] == 0).sum(),
        "PAY_1 == -2": (train["PAY_1"] == -2).sum(),
        "PAY_1 == 0": (train["PAY_1"] == 0).sum(),
    })
    print("\nUndocumented-code counts:\n", undoc)
    undoc.to_csv(f"{TAB}/undocumented_codes.csv", header=["count"])


def default_by_pay1(train):
    by_pay = train.groupby("PAY_1")["default"].agg(rate="mean", count="size")
    print("\nDefault rate by PAY_1 (September repayment status):\n", by_pay.round(3))
    by_pay.to_csv(f"{TAB}/default_rate_by_pay1.csv")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(by_pay.index.astype(str), by_pay["rate"], color="#4C72B0")
    ax.axhline(train["default"].mean(), color="gray", ls="--", label="overall rate")
    for x, (r, c) in enumerate(zip(by_pay["rate"], by_pay["count"])):
        ax.text(x, r, f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("PAY_1 (-2, -1, 0 = no-delay codes; 1-8 = months late)")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by September repayment status")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG}/03_default_by_pay1.png", dpi=150)
    plt.close()


def skew(train):
    cols = ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    for j, c in enumerate(cols):
        axes[0, j].hist(train[c], bins=60, color="#4C72B0")
        axes[0, j].set_title(f"{c} (raw), skew={train[c].skew():.1f}")
        t = signed_log(train[c])
        axes[1, j].hist(t, bins=60, color="#55A868")
        axes[1, j].set_title(f"{c} (signed log), skew={t.skew():.1f}")
    plt.tight_layout()
    plt.savefig(f"{FIG}/04_skew_raw_vs_log.png", dpi=150)
    plt.close()

    amount_cols = (["LIMIT_BAL"] + [f"BILL_AMT{k}" for k in range(1, 7)]
                   + [f"PAY_AMT{k}" for k in range(1, 7)])
    sk = pd.DataFrame({"raw": train[amount_cols].skew(),
                       "signed_log": signed_log(train[amount_cols]).skew()}).round(2)
    print("\nSkewness of amount columns:\n", sk)
    sk.to_csv(f"{TAB}/skewness.csv")


def negative_bills(train):
    bill_cols = [f"BILL_AMT{k}" for k in range(1, 7)]
    neg = pd.DataFrame({"negative_count": (train[bill_cols] < 0).sum(),
                        "min_value": train[bill_cols].min()})
    print("\nNegative bill amounts:\n", neg)
    print("Customers with at least one negative bill:",
          (train[bill_cols] < 0).any(axis=1).sum())
    neg.to_csv(f"{TAB}/negative_bills.csv")


if __name__ == "__main__":
    df = load_data()
    train, val, test = make_split(df)
    del val, test                        # EDA uses the training set only
    print("Training set:", train.shape)

    class_balance(train)
    undefined_codes(train)
    default_by_pay1(train)
    skew(train)
    negative_bills(train)
    print(f"\nDone. Figures in {FIG}/, tables in {TAB}/")