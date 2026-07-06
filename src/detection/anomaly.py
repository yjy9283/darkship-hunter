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
from .routes import nearest_waypoint_distance


def detect_dark_gaps(df: pd.DataFrame, max_gap_minutes: float = 30.0) -> pd.DataFrame:
    """선박별로 연속 리포트 간 시간 간격이 max_gap_minutes를 초과하는 구간을 찾는다.

    Returns:
        columns=[mmsi, gap_start, gap_end, gap_minutes, lat_before, lon_before, lat_after, lon_after]
    """
    records = []
    for mmsi, group in df.groupby("mmsi"):
        group = group.sort_values("timestamp")
        ts = group["timestamp"].to_numpy()
        gaps_minutes = np.diff(ts) / np.timedelta64(1, "m")

        for i, gap in enumerate(gaps_minutes):
            if gap > max_gap_minutes:
                before = group.iloc[i]
                after = group.iloc[i + 1]
                records.append(
                    {
                        "mmsi": mmsi,
                        "gap_start": before["timestamp"],
                        "gap_end": after["timestamp"],
                        "gap_minutes": float(gap),
                        "lat_before": before["lat"],
                        "lon_before": before["lon"],
                        "lat_after": after["lat"],
                        "lon_after": after["lon"],
                    }
                )
    return pd.DataFrame(records)


def detect_kinematic_jumps(
    df: pd.DataFrame,
    max_implied_speed_knots: float = 40.0,
    max_course_change_deg: float = 90.0,
) -> pd.DataFrame:
    """연속된 두 포인트 사이의 '내재 속도'(거리/시간)가 비정상적으로 크거나,
    보고된 침로(COG)가 급격히 바뀐 지점을 탐지한다.

    - implied_speed: 두 포인트 간 실제 이동거리로 역산한 속도. 이게 AIS가 보고한
      SOG나 선박 최대속력보다 훨씬 크면 '순간이동'급 이상치로 간주.
    - course_change: COG가 짧은 시간 안에 급격히 바뀌면 급변침 의심.

    Returns:
        columns=[mmsi, timestamp, lat, lon, implied_speed_knots, course_change_deg, reason]
    """
    records = []
    for mmsi, group in df.groupby("mmsi"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i in range(1, len(group)):
            prev, cur = group.iloc[i - 1], group.iloc[i]
            dt_hours = (cur["timestamp"] - prev["timestamp"]).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue

            dist_km = distance_km(prev["lat"], prev["lon"], cur["lat"], cur["lon"])
            implied_speed_knots = (dist_km / 1.852) / dt_hours  # km -> nautical miles -> knots

            course_change = np.nan
            if pd.notna(prev["cog"]) and pd.notna(cur["cog"]):
                course_change = bearing_diff(prev["cog"], cur["cog"])

            reasons = []
            if implied_speed_knots > max_implied_speed_knots:
                reasons.append("implausible_speed")
            if pd.notna(course_change) and course_change > max_course_change_deg:
                reasons.append("sharp_course_change")

            if reasons:
                records.append(
                    {
                        "mmsi": mmsi,
                        "timestamp": cur["timestamp"],
                        "lat": cur["lat"],
                        "lon": cur["lon"],
                        "implied_speed_knots": round(implied_speed_knots, 1),
                        "course_change_deg": round(course_change, 1) if pd.notna(course_change) else None,
                        "reason": ",".join(reasons),
                    }
                )
    return pd.DataFrame(records)


def detect_route_deviation(df: pd.DataFrame, waypoints: pd.DataFrame, threshold_km: float = 10.0) -> pd.DataFrame:
    """학습된 정상 waypoint들로부터 threshold_km 이상 떨어진 포인트를 찾는다.

    주의: 정상 항로가 드문 해역(원양 등)에서는 오탐이 늘어날 수 있음 — 한계점으로 명시.
    """
    if waypoints.empty:
        return pd.DataFrame(columns=["mmsi", "timestamp", "lat", "lon", "distance_to_route_km"])

    records = []
    for _, row in df.iterrows():
        dist = nearest_waypoint_distance(row["lat"], row["lon"], waypoints)
        if dist > threshold_km:
            records.append(
                {
                    "mmsi": row["mmsi"],
                    "timestamp": row["timestamp"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "distance_to_route_km": round(dist, 2),
                }
            )
    return pd.DataFrame(records)


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
    features["route_deviation_km"] = features.apply(
        lambda row: nearest_waypoint_distance(row["lat"], row["lon"], waypoints), axis=1
    )

    feature_cols = ["speed_feature", "course_change_feature", "route_deviation_km"]
    X = features[feature_cols].fillna(0).to_numpy()

    model = IsolationForest(contamination=contamination, random_state=42)
    features["anomaly_score"] = model.fit_predict(X)  # -1: 이상, 1: 정상
    features["is_anomaly"] = features["anomaly_score"] == -1

    return features
