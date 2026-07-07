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
from src.detection.anomaly import (
    detect_dark_gaps,
    detect_kinematic_jumps,
    detect_kinematic_jumps_statistical,
    score_with_isolation_forest,
    detect_route_deviation,
    detect_route_deviation_corridor,
)
from src.detection.routes import extract_waypoints, build_corridor_edges

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


def test_score_with_isolation_forest_returns_expected_columns_and_flags_some_anomalies():
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    waypoints = extract_waypoints(df, eps_km=5.0, min_samples=2)

    scored = score_with_isolation_forest(df, waypoints, contamination=0.2)

    assert set(["anomaly_score", "is_anomaly", "speed_feature", "course_change_feature", "route_deviation_km"]).issubset(
        scored.columns
    )
    assert scored["is_anomaly"].any()  # 픽스처에 명백한 이상치(순간이동 등)가 있으므로 최소 1건은 잡혀야 함
    assert len(scored) == len(df)  # 원본 행 수 그대로 유지(필터링 없이 전체에 점수만 부여)


def test_score_with_isolation_forest_handles_empty_waypoints():
    df = clean_pipeline(load_ais_csv(FIXTURE_PATH), min_points=3)
    empty_waypoints = pd.DataFrame(columns=["waypoint_id", "lat", "lon", "point_count"])

    scored = score_with_isolation_forest(df, empty_waypoints, contamination=0.2)
    assert (scored["route_deviation_km"] == 0.0).all()


def _build_corridor_test_data():
    """waypoint A(37.0,-122.0)와 B(37.5,-122.0) 사이를 여러 척이 왕복하는 합성 데이터.
    corridor 학습이 이 A-B 구간을 정상 경로로 인식하는지 검증하기 위한 픽스처."""
    import numpy as np

    rows = []
    base_time = pd.Timestamp("2026-01-01")
    # A, B 각각에 밀집 클러스터를 만들어 waypoint로 잡히게 하고, 그 사이를 잇는 직선상 포인트도 추가
    for mmsi in ["900000001", "900000002", "900000003"]:
        t = base_time
        # A 근처 밀집 (waypoint 형성용)
        for i in range(5):
            rows.append({"mmsi": mmsi, "timestamp": t, "lat": 37.0 + i * 0.001, "lon": -122.0, "sog": 10.0, "cog": 0.0})
            t += pd.Timedelta(minutes=5)
        # A -> B 직선 이동 (corridor 구간)
        for frac in np.linspace(0, 1, 8)[1:-1]:
            rows.append(
                {"mmsi": mmsi, "timestamp": t, "lat": 37.0 + frac * 0.5, "lon": -122.0, "sog": 12.0, "cog": 0.0}
            )
            t += pd.Timedelta(minutes=5)
        # B 근처 밀집 (waypoint 형성용)
        for i in range(5):
            rows.append({"mmsi": mmsi, "timestamp": t, "lat": 37.5 + i * 0.001, "lon": -122.0, "sog": 10.0, "cog": 0.0})
            t += pd.Timedelta(minutes=5)

    df = pd.DataFrame(rows)
    for col in ["heading", "nav_status", "ship_type"]:
        df[col] = pd.NA
    return df


def test_build_corridor_edges_learns_ab_segment():
    df = _build_corridor_test_data()
    waypoints = extract_waypoints(df, eps_km=1.0, min_samples=10)
    assert len(waypoints) == 2  # A, B 두 waypoint가 학습되어야 함

    corridors = build_corridor_edges(df, waypoints, min_vessel_count=2)
    assert len(corridors) == 1  # A-B 구간 하나만 학습되어야 함 (3척 모두 이 구간 이용)
    assert corridors.iloc[0]["vessel_count"] == 3


def test_corridor_based_deviation_reduces_false_positives_on_known_route():
    """A-B 사이를 정상 이동 중인 포인트가, 점 기반 방식에서는 이탈로 오탐되지만
    corridor 기반 방식에서는 정상 경로로 인식되어 이탈 건수가 줄어드는지 검증."""
    df = _build_corridor_test_data()
    waypoints = extract_waypoints(df, eps_km=1.0, min_samples=10)
    corridors = build_corridor_edges(df, waypoints, min_vessel_count=2)

    point_based = detect_route_deviation(df, waypoints, threshold_km=10.0)
    corridor_based = detect_route_deviation_corridor(df, waypoints, corridors, threshold_km=10.0)

    # A-B 중간 지점들은 waypoint(A 또는 B)에서 10km 넘게 떨어져 있어 점 기반으로는 이탈 처리됨
    assert len(point_based) > 0
    # 그러나 corridor 기반에서는 A-B 구간 위에 있으므로 이탈 건수가 확연히 줄어야 함
    assert len(corridor_based) < len(point_based)
