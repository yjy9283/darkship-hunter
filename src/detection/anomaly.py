"""
이상 항적 탐지 엔진.

3가지 규칙 기반 탐지 + 1가지 ML 기반 종합 스코어링으로 구성된 하이브리드 구조:

1. detect_dark_gaps        : AIS 신호가 비정상적으로 오래 끊긴 구간 (규칙 기반, 확정적)
2. detect_kinematic_jumps  : 속도/침로가 물리적으로 부자연스럽게 급변한 지점 (규칙 기반)
3. detect_route_deviation  : 학습된 정상 항로에서 벗어난 지점 (routes.py 결과 활용)
4. score_with_isolation_forest : 위 세 특징을 종합해 IsolationForest로 이상 점수 산출 (ML 기반)

규칙 기반 탐지는 "확실한 이상"을 놓치지 않게 하고, ML 기반 스코어링은
여러 지표가 복합적으로 애매하게 이상한 경우를 잡아내는 역할을 한다.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from .geo_utils import distance_km, bearing_deg, bearing_diff
from .routes import nearest_waypoint_distance, build_waypoint_tree, query_nearest_waypoint_distances
from .stats import group_zscore, percentile_rank


def detect_dark_gaps(df: pd.DataFrame, max_gap_minutes: float = 30.0) -> pd.DataFrame:
    """선박별로 연속 리포트 간 시간 간격이 max_gap_minutes를 초과하는 구간을 찾는다.

    groupby+shift로 벡터화되어 있어 파이썬 레벨 반복문 없이 처리한다.

    근거: ITU-R M.1371 국제표준상 AIS는 항해 중 2~10초, 정박 중이어도 최대 3분 간격으로
    신호를 보내야 한다. 기본값 60분은 정상 케이스 중 가장 느린 경우(3분)보다도 20배 긴
    시간이므로, 임의로 정한 값이 아니라 표준 규격 대비 통계적 근거를 갖는 기준이다.

    Returns:
        columns=[mmsi, gap_start, gap_end, gap_minutes, lat_before, lon_before, lat_after, lon_after]
    """
    d = df.sort_values(["mmsi", "timestamp"]).copy()
    g = d.groupby("mmsi")
    d["prev_timestamp"] = g["timestamp"].shift(1)
    d["prev_lat"] = g["lat"].shift(1)
    d["prev_lon"] = g["lon"].shift(1)

    gap_minutes = (d["timestamp"] - d["prev_timestamp"]).dt.total_seconds() / 60.0
    mask = gap_minutes > max_gap_minutes

    result = pd.DataFrame(
        {
            "mmsi": d.loc[mask, "mmsi"],
            "gap_start": d.loc[mask, "prev_timestamp"],
            "gap_end": d.loc[mask, "timestamp"],
            "gap_minutes": gap_minutes[mask],
            "lat_before": d.loc[mask, "prev_lat"],
            "lon_before": d.loc[mask, "prev_lon"],
            "lat_after": d.loc[mask, "lat"],
            "lon_after": d.loc[mask, "lon"],
        }
    )
    return result.reset_index(drop=True)


def _vectorized_haversine_km(lat1, lon1, lat2, lon2):
    """numpy 배열 입력을 받는 벡터화된 haversine 거리(km) 계산."""
    earth_radius_km = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _vectorized_bearing_diff(cog1, cog2):
    """두 방위각 배열 간 최소 차이(0~180도)를 벡터로 계산."""
    diff = np.abs(cog1 - cog2) % 360
    return np.minimum(diff, 360 - diff)


def _compute_point_kinematics(df: pd.DataFrame, min_speed_for_course_check_knots: float = 2.0) -> pd.DataFrame:
    """각 포인트에 대해 직전 포인트 대비 역산속도(implied_speed_knots)와
    침로변화(course_change_deg)를 벡터 연산으로 계산해 원본 df에 붙여 반환한다.

    detect_kinematic_jumps(고정 임계값)와 detect_kinematic_jumps_statistical(통계 기반)이
    이 계산 로직을 공유한다.
    """
    d = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True).copy()
    g = d.groupby("mmsi")
    d["prev_timestamp"] = g["timestamp"].shift(1)
    d["prev_lat"] = g["lat"].shift(1)
    d["prev_lon"] = g["lon"].shift(1)
    d["prev_cog"] = g["cog"].shift(1)

    valid = d["prev_timestamp"].notna()
    dt_hours = (d["timestamp"] - d["prev_timestamp"]).dt.total_seconds() / 3600.0
    valid = valid & (dt_hours > 0)

    dist_km = pd.Series(np.nan, index=d.index)
    dist_km[valid] = _vectorized_haversine_km(
        d.loc[valid, "prev_lat"].to_numpy(),
        d.loc[valid, "prev_lon"].to_numpy(),
        d.loc[valid, "lat"].to_numpy(),
        d.loc[valid, "lon"].to_numpy(),
    )
    d["implied_speed_knots"] = (dist_km / 1.852) / dt_hours.replace(0, np.nan)

    is_moving = d["sog"].notna() & (d["sog"] >= min_speed_for_course_check_knots)
    course_valid = valid & is_moving & d["prev_cog"].notna() & d["cog"].notna()
    course_change = pd.Series(np.nan, index=d.index)
    course_change[course_valid] = _vectorized_bearing_diff(
        d.loc[course_valid, "prev_cog"].to_numpy(), d.loc[course_valid, "cog"].to_numpy()
    )
    d["course_change_deg"] = course_change
    d["_valid"] = valid
    return d


def detect_kinematic_jumps(
    df: pd.DataFrame,
    max_implied_speed_knots: float = 40.0,
    max_course_change_deg: float = 90.0,
    min_speed_for_course_check_knots: float = 2.0,
) -> pd.DataFrame:
    """연속된 두 포인트 사이의 '내재 속도'(거리/시간)가 비정상적으로 크거나,
    보고된 침로(COG)가 급격히 바뀐 지점을 탐지한다. (고정 임계값 방식)

    이 방식은 모든 선박에 동일한 임계값을 적용하기 때문에, 예인선/파일럿선처럼
    원래 자주 급선회하는 선박종류가 과다 탐지되는 경향이 있다 (실데이터 검증에서 확인,
    트러블슈팅 로그 참고). 선박종류별 정상 행동 baseline과 비교하려면
    detect_kinematic_jumps_statistical()을 사용할 것.

    Returns:
        columns=[mmsi, timestamp, lat, lon, implied_speed_knots, course_change_deg, reason]
    """
    d = _compute_point_kinematics(df, min_speed_for_course_check_knots)
    valid = d["_valid"]

    is_speed_anomaly = valid & (d["implied_speed_knots"] > max_implied_speed_knots)
    is_course_anomaly = d["course_change_deg"] > max_course_change_deg
    is_anomaly = is_speed_anomaly | is_course_anomaly.fillna(False)

    result = d.loc[is_anomaly, ["mmsi", "timestamp", "lat", "lon"]].copy()
    result["implied_speed_knots"] = d.loc[is_anomaly, "implied_speed_knots"].round(1)
    result["course_change_deg"] = d.loc[is_anomaly, "course_change_deg"].round(1)

    reasons = []
    for idx in result.index:
        r = []
        if is_speed_anomaly[idx]:
            r.append("implausible_speed")
        if is_course_anomaly.get(idx, False):
            r.append("sharp_course_change")
        reasons.append(",".join(r))
    result["reason"] = reasons

    return result.reset_index(drop=True)


def detect_kinematic_jumps_statistical(
    df: pd.DataFrame,
    ship_type_map: pd.Series,
    z_thresh: float = 3.0,
    min_absolute_speed_knots: float = 25.0,
    min_absolute_course_deg: float = 60.0,
    min_speed_for_course_check_knots: float = 2.0,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """선박종류별 정상 행동 분포(평균·표준편차) 대비 z-score로 이상을 판정한다.

    고정 임계값 방식(detect_kinematic_jumps)의 한계 — "예인선은 원래 급선회가
    잦은데 화물선과 같은 기준을 적용하는 게 맞나?" — 를 보완하기 위한 방식.
    같은 선박종류 집단 안에서 통계적으로 얼마나 벗어났는지(z-score)를 기준으로 삼는다.

    이상 판정 조건 (AND — 통계적으로 드물면서 동시에 물리적으로도 의미있는 크기여야 함):
    - |z-score| > z_thresh (해당 선박종류 평균 대비 표준편차 기준 이상치)
    - 그리고 실제 값이 최소 절대 기준(min_absolute_*)을 넘어야 함
      (표준편차가 매우 작은 선박종류에서 z-score만으로 사소한 변동까지 잡는 것 방지)

    표본이 적은 선박종류(min_group_size 미만)는 전체 데이터 기준으로 자동 대체된다.

    Returns:
        columns=[mmsi, timestamp, lat, lon, ship_type_name, implied_speed_knots,
                 speed_zscore, speed_percentile, course_change_deg, course_zscore,
                 course_percentile, reason]
    """
    from ..preprocessing.vessel_types import get_vessel_type_name

    d = _compute_point_kinematics(df, min_speed_for_course_check_knots)
    d["ship_type"] = d["mmsi"].map(ship_type_map)

    speed_mask = d["_valid"] & d["implied_speed_knots"].notna()
    d["speed_zscore"] = np.nan
    d.loc[speed_mask, "speed_zscore"] = group_zscore(
        d.loc[speed_mask, "implied_speed_knots"], d.loc[speed_mask, "ship_type"], min_group_size
    )
    d["speed_percentile"] = np.nan
    d.loc[speed_mask, "speed_percentile"] = percentile_rank(d.loc[speed_mask, "implied_speed_knots"])

    course_mask = d["course_change_deg"].notna()
    d["course_zscore"] = np.nan
    if course_mask.any():
        d.loc[course_mask, "course_zscore"] = group_zscore(
            d.loc[course_mask, "course_change_deg"], d.loc[course_mask, "ship_type"], min_group_size
        )
    d["course_percentile"] = np.nan
    if course_mask.any():
        d.loc[course_mask, "course_percentile"] = percentile_rank(d.loc[course_mask, "course_change_deg"])

    is_speed_anomaly = (
        speed_mask
        & (d["speed_zscore"].abs() > z_thresh)
        & (d["implied_speed_knots"] > min_absolute_speed_knots)
    )
    is_course_anomaly = (
        course_mask
        & (d["course_zscore"].abs() > z_thresh)
        & (d["course_change_deg"] > min_absolute_course_deg)
    )
    is_anomaly = is_speed_anomaly | is_course_anomaly

    result = d.loc[is_anomaly, ["mmsi", "timestamp", "lat", "lon", "ship_type"]].copy()
    result["ship_type_name"] = result["ship_type"].apply(get_vessel_type_name)
    result["implied_speed_knots"] = d.loc[is_anomaly, "implied_speed_knots"].round(1)
    result["speed_zscore"] = d.loc[is_anomaly, "speed_zscore"].round(2)
    result["speed_percentile"] = d.loc[is_anomaly, "speed_percentile"].round(2)
    result["course_change_deg"] = d.loc[is_anomaly, "course_change_deg"].round(1)
    result["course_zscore"] = d.loc[is_anomaly, "course_zscore"].round(2)
    result["course_percentile"] = d.loc[is_anomaly, "course_percentile"].round(2)

    reasons = []
    for idx in result.index:
        r = []
        if is_speed_anomaly.get(idx, False):
            r.append("statistical_speed_outlier")
        if is_course_anomaly.get(idx, False):
            r.append("statistical_course_outlier")
        reasons.append(",".join(r))
    result["reason"] = reasons

    return result.drop(columns=["ship_type"]).reset_index(drop=True)


def detect_route_deviation(df: pd.DataFrame, waypoints: pd.DataFrame, threshold_km: float = 15.0) -> pd.DataFrame:
    """학습된 정상 waypoint들로부터 threshold_km 이상 떨어진 포인트를 찾는다.

    BallTree로 전체 포인트의 최근접 waypoint 거리를 한 번에 벡터 연산한다
    (파이썬 루프 방식 대비 17만행 기준 약 70배 빠름 — 72초 -> 1초대).

    주의: 정상 항로가 드문 해역(원양 등)에서는 오탐이 늘어날 수 있음.
    또한 waypoint가 실제 항로선이 아닌 밀집구역 점 클러스터라서, 두 항구 사이의
    정상 항해 구간도 이탈로 잡힐 수 있다는 점을 README에 한계점으로 명시함.
    """
    if waypoints.empty:
        return pd.DataFrame(columns=["mmsi", "timestamp", "lat", "lon", "distance_to_route_km"])

    tree = build_waypoint_tree(waypoints)
    distances = query_nearest_waypoint_distances(df, tree)

    result = df.loc[distances > threshold_km, ["mmsi", "timestamp", "lat", "lon"]].copy()
    result["distance_to_route_km"] = np.round(distances[distances > threshold_km], 2)
    return result.reset_index(drop=True)


def score_with_isolation_forest(
    df: pd.DataFrame, waypoints: pd.DataFrame, contamination: float = 0.02
) -> pd.DataFrame:
    """속도, 침로 변화량, 항로 이탈도를 종합 피처로 IsolationForest 이상 점수를 산출한다.

    규칙 기반 탐지가 놓칠 수 있는 '여러 지표가 동시에 애매하게 이상한' 케이스를 보완하는 역할.

    Returns:
        원본 df + anomaly_score(낮을수록 이상), is_anomaly(bool) 컬럼 추가
    """
    features = df.copy().sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    features["speed_feature"] = features.groupby("mmsi")["sog"].transform(lambda s: s.fillna(s.median()))
    features["course_change_feature"] = (
        features.groupby("mmsi")["cog"].diff().abs().fillna(0).apply(lambda x: min(x, 360 - x) if x > 180 else x)
    )
    features["route_deviation_km"] = (
        query_nearest_waypoint_distances(features, build_waypoint_tree(waypoints)) if not waypoints.empty else 0.0
    )

    feature_cols = ["speed_feature", "course_change_feature", "route_deviation_km"]
    X = features[feature_cols].fillna(0).to_numpy()

    model = IsolationForest(contamination=contamination, random_state=42)
    features["anomaly_score"] = model.fit_predict(X)  # -1: 이상, 1: 정상
    features["is_anomaly"] = features["anomaly_score"] == -1

    return features
