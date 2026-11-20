"""Stage 1: look for the seven data problems in the original train.csv.

Read-only. Nothing is written to the CSV files.
Run from the project folder:  .venv\\Scripts\\python.exe src\\mp145\\data_checks.py
"""

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    target = params["target"]
    train = pd.read_csv(ROOT / params["data"]["train"])
    later = pd.read_csv(ROOT / params["data"]["later"])
    holdout = pd.read_csv(ROOT / params["data"]["holdout"])

    numeric_cols = [c for c in train.select_dtypes("number").columns if c != target]
    text_cols = ["phone_tier", "circle"]

    print("== 1. leaking column")
    # A column in train that the holdout file does not have is filled in after the event.
    leaking = [c for c in train.columns if c not in holdout.columns and c != target]
    print("missing from holdout:", leaking)
    print(train[leaking + [target]].corr().round(4))

    print("\n== 2. unit mix-up (plan_gb_limit by circle)")
    print("train:")
    print(train.groupby("circle")["plan_gb_limit"].agg(["min", "mean", "max"]).round(2))
    print("later (clean):")
    print(later.groupby("circle")["plan_gb_limit"].agg(["min", "mean", "max"]).round(2))
    semi = train[train.circle == "Semi-urban"]
    print("Semi-urban plan_gb_limit divided by 1024 (first 8 rows):")
    print((semi.plan_gb_limit / 1024).head(8).round(2).tolist())

    print("\n== 3. sentinel value (smallest and most common values)")
    for col in numeric_cols:
        print(col, "min:", train[col].min(), " most common:", train[col].value_counts().head(1).to_dict())

    print("\n== 4. duplicate rows")
    print("extra copies:", train.duplicated().sum())

    print("\n== 5. stuck sensor (longest run of the same value, sorted by date)")
    ordered = train.sort_values(["date", "record_id"]).reset_index(drop=True)
    for col in numeric_cols:
        run_id = (ordered[col] != ordered[col].shift()).cumsum()
        print(col, "longest run:", run_id.value_counts().max())
    run_id = (ordered["video_minutes"] != ordered["video_minutes"].shift()).cumsum()
    longest = run_id.value_counts().idxmax()
    run = ordered[run_id == longest]
    print("video_minutes run:", run.video_minutes.iloc[0], "from", run.date.iloc[0], "to", run.date.iloc[-1])

    print("\n== 6. target typos (largest targets)")
    print(train[target].nlargest(10).tolist())
    print("rows above 3 times the median:", (train[target] > 3 * train[target].median()).sum())

    print("\n== 7. spelling variants (text columns)")
    for col in text_cols:
        print(col, "train:", sorted(map(repr, train[col].unique())))
        print(col, "not in later.csv:", sorted(set(train[col].unique()) - set(later[col].unique())))
        print(col, "levels in later.csv:", later[col].nunique())


if __name__ == "__main__":
    main()
