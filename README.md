# MP145: Mobile Data Use (Patna)

SCSE3040 MLOps mini project.

- Group: **G255**
- Problem: **MP145**
- Faculty: Upasna Joshi, Batch 78

| Enrolment no. | Name |
|---|---|
| S24CSEU0529 | ShubhYadavOm |
| S24CSEU0681 | ARMANPYRBOT |
| S24CSEU0493 | ShambhaviSaumya |
| S24CSEU0590 | TanishVankadaraRamgopal |

## Task

Predict `gb_per_day`, the mobile data a customer uses in a day (regression, scored with MAE).

## Data

The CSVs are tracked with DVC and are not committed to git. To get them, copy the three
files from `data/` in the problem zip into `data/raw/`, then check they match:

```
dvc status
```

`data_sha256` of the original `train.csv`:
`02f97e14c5496b63e60f86746eef3a4849a2d1b697368ccb33d7938321b32c11`

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install dvc
```

## Run

Tests (no CSV files needed):

```
.venv\Scripts\python.exe -m pytest
```

Training. Cleans `data/raw/train.csv`, trains 4 candidate models, logs each run to `mlflow.db`, and registers the best one as `champion`:

```
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m mp145.train
```

Cleaning rules are in `src/mp145/clean.py`, with settings in `params.yaml`. The data checks that produced `reports/findings.yaml` are in `src/mp145/data_checks.py`.
