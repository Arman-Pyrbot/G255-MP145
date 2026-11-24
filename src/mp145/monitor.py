"""Drift monitoring, fair-group check, retraining and holdout prediction.

Run from the project folder:
    $env:PYTHONPATH = "src"
    python -m mp145.monitor

It follows PROBLEM.md section 7 and 8 exactly, then fills tracking and
monitoring in reports/findings.yaml.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from .clean import clean
from .train import make_pipe

ROOT = Path(__file__).resolve().parents[2]

PARAMS_PATH = ROOT / "params.yaml"
FINDINGS_PATH = ROOT / "reports" / "findings.yaml"
EXPERIMENT = "mp145-gb-per-day"
MODEL_NAME = "mp145-gb-per-day"
V1_RUN_ID = "63a280adb06e4fe4b36397bacf6f8974"  # champion v1 from mlflow.db


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from `expected`."""
    edges = np.unique(np.quantile(expected.astype(float), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    e = e_counts / e_counts.sum()
    a = a_counts / a_counts.sum()
    eps = 1e-6
    return float(np.sum((a - e) * np.log((a + eps) / (e + eps))))


def load_clean(params: dict, name: str) -> pd.DataFrame:
    """Read one CSV and apply the same cleaning rules used for training."""
    df = pd.read_csv(ROOT / params["data"][name])
    # Holdout has no target or bill column; add a dummy so clean() can drop it.
    if "gb_from_monthly_bill" not in df.columns:
        df["gb_from_monthly_bill"] = np.nan
    return clean(df, params["clean"]).reset_index(drop=True)


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    numeric = params["features"]["numeric"]
    cols = numeric + params["features"]["categorical"]
    target = params["target"]

    train = load_clean(params, "train")
    later = load_clean(params, "later")

    # 1. PSI for every numeric column (10 bins), train vs later.
    psi_values = {c: round(psi(train[c], later[c]), 6) for c in numeric}
    most_drifted = max(psi_values, key=psi_values.get)

    # 2. v1 = the registered champion: a linear model on cleaned train only.
    v1 = make_pipe({"model": "linear"}, params)
    v1.fit(train[cols], train[target])
    pred_later = v1.predict(later[cols])
    mae_by_group = {
        group: round(float(mean_absolute_error(
            later.loc[later["circle"] == group, target],
            pred_later[later["circle"] == group],
        )), 6)
        for group in ["Urban", "Semi-urban", "Rural"]
    }
    worst_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Retrain v2 on cleaned train + first 70% of later (sorted by date).
    ordered = later.sort_values("date").reset_index(drop=True)
    cut = int(len(ordered) * 0.7)
    check = ordered.iloc[cut:].reset_index(drop=True)
    train2 = pd.concat([train, ordered.iloc[:cut]], ignore_index=True)
    v2 = make_pipe({"model": "linear"}, params)
    v2.fit(train2[cols], train2[target])
    mae_v1 = round(float(mean_absolute_error(check[target], v1.predict(check[cols]))), 6)
    mae_v2 = round(float(mean_absolute_error(check[target], v2.predict(check[cols]))), 6)

    # 4. Client's promotion rule: >= 15% lower MAE on the last 30%.
    improve = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improve >= 0.15 else "keep"
    champion = v2 if decision == "promote" else v1

    # Log v2 to MLflow, register it as version 2, promote if the rule passed.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="v2-retrain") as run:
        mlflow.log_param("data_sha256", __import__("hashlib").sha256(
            (ROOT / params["data"]["train"]).read_bytes()).hexdigest())
        mlflow.log_params({"model": "linear-v2", "train_rows": len(train2)})
        mlflow.log_metrics({
            "mae_v1_check": mae_v1,
            "mae_v2_check": mae_v2,
            "mae_improvement": round(improve, 6),
        })
        mlflow.sklearn.log_model(
            v2, name="model", serialization_format="cloudpickle"
        )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME).version
    if decision == "promote":
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)

    # Save the champion as the joblib artifact that the API serves.
    import joblib

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_dir / "model.joblib")

    # 5. Predict the holdout file with the champion model.
    holdout = load_clean(params, "holdout")
    predictions = champion.predict(holdout[cols])
    out = ROOT / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": np.round(predictions, 6),
    }).to_csv(out / "holdout_predictions.csv", index=False)

    # 6. Fill the form.
    findings = yaml.safe_load(FINDINGS_PATH.read_text(encoding="utf-8"))
    findings["tracking"] = {
        "experiment_name": EXPERIMENT,
        "best_run_id": V1_RUN_ID,
        "registered_model_name": MODEL_NAME,
        "champion_version": version if decision == "promote" else 1,
    }
    findings["monitoring"] = {
        "psi": psi_values,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "decision": decision,
    }
    FINDINGS_PATH.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")

    print("PSI:", psi_values, "-> most drifted:", most_drifted)
    print("MAE by group:", mae_by_group, "-> worst:", worst_group)
    print(f"check MAE v1={mae_v1} v2={mae_v2} improvement={improve:.2%} -> {decision}")
    print(f"registered v2 as version {version}, champion version="
          f"{version if decision == 'promote' else 1}")
    print("wrote predictions/holdout_predictions.csv and reports/findings.yaml")


if __name__ == "__main__":
    main()
