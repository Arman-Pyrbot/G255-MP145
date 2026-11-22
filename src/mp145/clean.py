"""Cleaning rules for the original train.csv.

Each rule fixes one of the data problems recorded in reports/findings.yaml.
Settings come from the `clean` block of params.yaml.
"""


def clean(df, cfg):
    # Duplicate rows: keep the first copy.
    df = df.drop_duplicates()

    # Sentinel value: a typed -999 in place of a blank. Drop those rows.
    df = df[df[cfg["sentinel_column"]] != cfg["sentinel_value"]]

    # Stuck sensor: the same video_minutes reading on every row in the date range.
    stuck = (df[cfg["stuck_column"]] == cfg["stuck_value"]) & df["date"].between(
        cfg["stuck_start"], cfg["stuck_end"]
    )
    df = df[~stuck].copy()

    # Unit mix-up: one circle recorded plan_gb_limit in another unit. The factor is in findings.yaml.
    rows = df["circle"] == cfg["unit_group"]
    df.loc[rows, cfg["unit_column"]] = df.loc[rows, cfg["unit_column"]] / cfg["unit_factor"]

    # Leaking column: only known after the event. Never a feature.
    df = df.drop(columns=[cfg["leaking_column"]])

    return df.reset_index(drop=True)
