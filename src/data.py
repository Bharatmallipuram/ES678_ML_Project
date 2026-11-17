# src/data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def load_data(path="data/raw/default of credit card clients.xls"):
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"default payment next month": "default",
                            "PAY_0": "PAY_1"})      # makes PAY_1..PAY_6 consistent
    df = df.drop(columns=["ID"])
    return df

def make_split(df):
    # 60 / 20 / 20, stratified on the target
    train, temp = train_test_split(df, test_size=0.4, stratify=df["default"],
                                   random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["default"],
                                 random_state=SEED)
    # Persist original row IDs so the split is auditable and stable.
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"index": list(train.index) + list(val.index) + list(test.index),
                  "split": ["train"]*len(train) + ["val"]*len(val) + ["test"]*len(test)}
                ).to_csv("results/split_indices.csv", index=False)
    return train, val, test


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)                         # expect (30000, 24)
    print("Defaults:", df["default"].sum())           # expect 6636

    train, val, test = make_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part)} rows, default rate {part['default'].mean():.3f}")
