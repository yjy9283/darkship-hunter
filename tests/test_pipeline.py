"""
파이프라인 핵심 로직 단위 테스트.

tests/fixtures/sample_ais.csv 는 실제 분석용 데이터가 아니라,
아래 3가지 케이스를 검증하기 위해 직접 만든 소규모 테스트 픽스처다:

- MMSI 100000001: 정상 항로 (이상 없음)
- MMSI 100000002: 중간에 3시간 신호 중단 (dark gap 테스트용)
- MMSI 100000003: 중간에 갑자기 먼 지점으로 순간이동 (kinematic jump 테스트용)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.preprocessing.loader import load_ais_csv
from src.preprocessing.clean import clean_pipeline, remove_invalid_coordinates
from src.detection.anomaly import detect_dark_gaps, detect_kinematic_jumps, detect_kinematic_jumps_statistical
from src.detection.routes import extract_waypoints

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_ais.csv"


def test_load_ais_csv_maps_columns_correctly():
    df = load_ais_csv(FIXTURE_PATH)
    assert list(df.columns) == [
        "mmsi", "timestamp", "lat", "lon", "sog", "cog", "heading", "nav_status", "ship_type",
    ]
    assert len(df) == 19
    assert df["mmsi"].apply(lambda x: isinstance(x, str)).all()


def test_remove_invalid_coordinates_keeps_valid_rows():
    df = load_ais_csv(FIXTURE_PATH)
    cleaned = remove_invalid_coordinates(df)
    assert len(cleaned) == len(df)  # 이 픽스처엔 잘못된 좌표가 없어야 함


def test_clean_pipeline_filters_short_trajectories():
    df = load_ais_csv(FIXTURE_PATH)
    cleaned = clean_pipeline(df, min_points=3)
    assert set(cleaned["mmsi"].unique()) == {"100000001", "100000002", "100000003"}


def test_detect_dark_gaps_finds_vessel_2_gap():
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    gaps = detect_dark_gaps(df, max_gap_minutes=30)
    assert len(gaps) >= 1
    assert "100000002" in gaps["mmsi"].values
    vessel_2_gap = gaps[gaps["mmsi"] == "100000002"].iloc[0]
    assert vessel_2_gap["gap_minutes"] > 150  # 3시간 근처 gap


def test_detect_dark_gaps_no_false_positive_on_vessel_1():
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    gaps = detect_dark_gaps(df, max_gap_minutes=30)
    assert "100000001" not in gaps["mmsi"].values


def test_detect_kinematic_jumps_finds_vessel_3_teleport():
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    jumps = detect_kinematic_jumps(df, max_implied_speed_knots=40.0)
    assert "100000003" in jumps["mmsi"].values


def test_detect_kinematic_jumps_statistical_handles_small_groups_via_global_fallback():
    """표본이 min_group_size(기본 30)보다 훨씬 적은 소규모 픽스처에서도
    전체 데이터 기준(global fallback)으로 정상 동작하고 에러 없이 결과를 반환하는지 확인."""
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    ship_type_map = df.groupby("mmsi")["ship_type"].first()
    result = detect_kinematic_jumps_statistical(df, ship_type_map, z_thresh=1.0)
    assert set(["speed_zscore", "course_zscore", "speed_percentile", "course_percentile", "reason"]).issubset(
        result.columns
    )


def test_extract_waypoints_returns_dataframe_with_expected_columns():
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    waypoints = extract_waypoints(df, eps_km=5.0, min_samples=2)
    assert set(["waypoint_id", "lat", "lon", "point_count"]).issubset(waypoints.columns)
