"""Train candidate models on the cleaned train.csv, log every run to MLflow,
and register the best one as the champion.

Run from the project folder (PowerShell):
    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe -m mp145.train
"""

import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from mp145.clean import clean


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "mp145-gb-per-day"
MODEL_NAME = "mp145-gb-per-day"


def make_model(cand, seed):
    if cand["model"] == "linear":
        return LinearRegression()
    if cand["model"] == "tree":
        return DecisionTreeRegressor(max_depth=cand["max_depth"], random_state=seed)
    return RandomForestRegressor(
        n_estimators=cand["n_estimators"], max_depth=cand["max_depth"], random_state=seed
    )


def make_pipe(cand, params):
    features = params["features"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), features["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features["categorical"]),
    ])
    return make_pipeline(prep, make_model(cand, params["train"]["random_state"]))


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    tr = params["train"]
    raw_path = ROOT / params["data"]["train"]

    data = clean(pd.read_csv(raw_path), params["clean"])
    columns = params["features"]["numeric"] + params["features"]["categorical"]
    X, y = data[columns], data[params["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tr["test_size"], random_state=tr["random_state"]
    )

    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    # Choose by cross-validated MAE on the training part, never by the test part.
    best_cv, best_run_id = None, None
    for cand in tr["candidates"]:
        pipe = make_pipe(cand, params)
        cv_mae = -cross_val_score(
            pipe, X_train, y_train, cv=tr["cv_folds"], scoring="neg_mean_absolute_error"
        ).mean()
        pipe.fit(X_train, y_train)
        test_mae = mean_absolute_error(y_test, pipe.predict(X_test))

        run_name = " ".join(f"{k}={v}" for k, v in cand.items())
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", sha)
            mlflow.log_params(cand)
            mlflow.log_metric("mae_cv", cv_mae)
            mlflow.log_metric("mae_test", test_mae)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
            print(f"{run_name:45s} mae_cv={cv_mae:.4f} mae_test={test_mae:.4f} run={run.info.run_id}")

        if best_cv is None or cv_mae < best_cv:
            best_cv, best_run_id = cv_mae, run.info.run_id

    version = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version.version)
    print(f"champion: {MODEL_NAME} version {version.version} from run {best_run_id} (mae_cv={best_cv:.4f})")


if __name__ == "__main__":
    main()
