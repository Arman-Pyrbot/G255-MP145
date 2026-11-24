"""Tests for feature selection and request framing. No CSVs required."""

from mp145.features import FEATURE_COLUMNS, make_request_frame, select_features


PAYLOAD = {
    "age": 40, "screen_hours": 6, "apps_installed": 100,
    "plan_gb_limit": 3.5, "video_minutes": 150, "wifi_hours": 8,
    "phone_tier": "mid", "circle": "Urban",
}


def test_select_features_excludes_leaking_column():
    df = __import__("pandas").DataFrame(
        columns=FEATURE_COLUMNS + ["gb_from_monthly_bill", "gb_per_day"]
    )
    result = select_features(df)
    assert list(result.columns) == FEATURE_COLUMNS
    assert "gb_from_monthly_bill" not in result.columns


def test_request_frame_has_exact_feature_columns():
    row = make_request_frame(PAYLOAD)
    assert list(row.columns) == FEATURE_COLUMNS
    assert len(row) == 1
