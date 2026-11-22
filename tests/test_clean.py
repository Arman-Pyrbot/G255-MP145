"""Tests for the cleaning rules. They use small made-up rows, never the CSV files."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mp145.clean import clean


CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "params.yaml").read_text())["clean"]


def make_rows(**overrides):
    row = {
        "record_id": "R1",
        "date": "2026-02-01",
        "age": 30,
        "screen_hours": 5.0,
        "apps_installed": 100,
        "plan_gb_limit": 3.0,
        "video_minutes": 120,
        "wifi_hours": 6.0,
        "phone_tier": "mid",
        "circle": "Urban",
        "gb_from_monthly_bill": 4.0,
        "gb_per_day": 4.1,
    }
    row.update(overrides)
    return row


def test_duplicates_and_sentinel_rows_are_removed():
    rows = [
        make_rows(record_id="R1"),
        make_rows(record_id="R1"),
        make_rows(record_id="R2", age=-999),
    ]
    out = clean(pd.DataFrame(rows), CFG)
    assert list(out["record_id"]) == ["R1"]


def test_semi_urban_plan_is_divided_by_unit_factor():
    rows = [
        make_rows(record_id="R1", circle="Semi-urban", plan_gb_limit=2867.2),
        make_rows(record_id="R2", circle="Urban", plan_gb_limit=3.0),
    ]
    out = clean(pd.DataFrame(rows), CFG).set_index("record_id")
    assert out.loc["R1", "plan_gb_limit"] == pytest.approx(2.8)
    assert out.loc["R2", "plan_gb_limit"] == pytest.approx(3.0)


def test_stuck_video_rows_removed_only_inside_date_range():
    rows = [
        make_rows(record_id="R1", date="2026-05-10", video_minutes=152),
        make_rows(record_id="R2", date="2026-05-30", video_minutes=152),
        make_rows(record_id="R3", date="2026-05-10", video_minutes=99),
    ]
    out = clean(pd.DataFrame(rows), CFG)
    assert set(out["record_id"]) == {"R2", "R3"}


def test_leaking_column_is_dropped():
    out = clean(pd.DataFrame([make_rows()]), CFG)
    assert "gb_from_monthly_bill" not in out.columns
    assert "gb_per_day" in out.columns
