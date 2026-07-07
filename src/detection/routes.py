"""
정상 항로(route) 모델을 학습하는 모듈.

접근 방식 (TREAD 논문 방법론 참고):
1. 전체 AIS 포인트를 대상으로 DBSCAN(haversine metric)을 돌려서
   선박들이 자주 지나가는 밀집 구역(waypoint)을 추출한다.
2. 각 waypoint 클러스터의 중심 좌표를 "정상 항로 포인트"로 저장한다.
3. 새로운 궤적의 각 포인트가 이 정상 포인트들과 얼마나 가까운지를
   이상탐지(anomaly.py)에서 "항로 이탈도"로 사용한다.

주의: DBSCAN은 밀집도 기반이라, 선박이 드문 항로(외곽 항로)는
waypoint로 잡히지 않을 수 있다. 이 경우 항로 이탈 탐지가 과민 반응할
수 있다는 걸 README에 한계점으로 명시해야 한다.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def extract_waypoints(
    df: pd.DataFrame, eps_km: float = 2.0, min_samples: int = 15, max_points: int = 30000, random_state: int = 42
) -> pd.DataFrame:
    """AIS 포인트 전체에서 밀집 waypoint를 추출한다.

    Args:
        df: 표준 스키마 DataFrame (lat, lon 컬럼 필수)
        eps_km: DBSCAN 이웃 반경 (km) — 작을수록 더 세밀한 클러스터
        min_samples: 클러스터로 인정할 최소 포인트 수
        max_points: DBSCAN에 투입할 최대 포인트 수. 정박지처럼 극도로 밀집된 구역이 있으면
            ball_tree 알고리즘도 메모리를 과도하게 사용할 수 있어, 항로/밀도 패턴 학습
            목적상 전체 포인트가 아닌 대표 샘플로 충분하므로 샘플링한다.
        random_state: 샘플링 재현성을 위한 시드

    Returns:
        columns=[waypoint_id, lat, lon, point_count] — 각 waypoint의 중심 좌표와 소속 포인트 수
    """
    if len(df) > max_points:
        df = df.sample(max_points, random_state=random_state)

    coords = df[["lat", "lon"]].to_numpy()
    coords_rad = np.radians(coords)

    # haversine metric은 라디안 입력을 받고, eps도 라디안 단위 거리로 변환해야 함
    earth_radius_km = 6371.0
    eps_rad = eps_km / earth_radius_km

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    labels = db.fit_predict(coords_rad)

    df_labeled = df.copy()
    df_labeled["waypoint_cluster"] = labels

    clustered = df_labeled[df_labeled["waypoint_cluster"] != -1]
    waypoints = (
        clustered.groupby("waypoint_cluster")
        .agg(lat=("lat", "mean"), lon=("lon", "mean"), point_count=("lat", "size"))
        .reset_index()
        .rename(columns={"waypoint_cluster": "waypoint_id"})
    )
    return waypoints.sort_values("point_count", ascending=False).reset_index(drop=True)


def nearest_waypoint_distance(lat: float, lon: float, waypoints: pd.DataFrame) -> float:
    """주어진 좌표에서 가장 가까운 waypoint까지의 거리(km)를 반환한다.

    단일 포인트용. 대량의 포인트를 처리할 땐 build_waypoint_tree +
    query_nearest_waypoint_distances (벡터화, 훨씬 빠름)를 사용할 것.
    """
    from .geo_utils import distance_km

    if waypoints.empty:
        return float("nan")
    dists = waypoints.apply(lambda row: distance_km(lat, lon, row["lat"], row["lon"]), axis=1)
    return float(dists.min())


def build_waypoint_tree(waypoints: pd.DataFrame):
    """waypoint 좌표들로 BallTree(haversine)를 미리 구축해둔다.

    대량 포인트에 대해 반복적으로 최근접 waypoint 거리를 구할 때,
    포인트마다 파이썬 루프+apply로 계산하면 매우 느리다 (17만행에 약 70초).
    BallTree를 한 번만 만들고 query를 벡터화하면 같은 작업이 1초 내외로 끝난다.
    """
    from sklearn.neighbors import BallTree

    coords_rad = np.radians(waypoints[["lat", "lon"]].to_numpy())
    return BallTree(coords_rad, metric="haversine")


def query_nearest_waypoint_distances(df: pd.DataFrame, waypoint_tree, earth_radius_km: float = 6371.0) -> np.ndarray:
    """df의 모든 포인트에 대해 최근접 waypoint까지의 거리(km)를 한 번에 벡터로 계산한다."""
    coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
    dist_rad, _ = waypoint_tree.query(coords_rad, k=1)
    return dist_rad[:, 0] * earth_radius_km


def query_nearest_waypoint_ids(df: pd.DataFrame, waypoint_tree, waypoints: pd.DataFrame) -> np.ndarray:
    """df의 모든 포인트에 대해 최근접 waypoint의 waypoint_id를 벡터로 반환한다."""
    coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
    _, idx = waypoint_tree.query(coords_rad, k=1)
    wp_ids = waypoints["waypoint_id"].to_numpy()
    return wp_ids[idx[:, 0]]


def build_corridor_edges(df: pd.DataFrame, waypoints: pd.DataFrame, min_vessel_count: int = 2) -> pd.DataFrame:
    """실제 선박들이 이동한 순서를 바탕으로, waypoint와 waypoint 사이의 "항로 구간(corridor)"을 학습한다.

    단순히 waypoint(점)만 학습하면 "두 지점 사이를 어떻게 이었는지(경로/방향)"를 모른다는
    한계가 있다 (트러블슈팅 로그 8번 참고). 이를 보완하기 위해:

    1. 각 포인트를 최근접 waypoint에 할당한다.
    2. 선박(mmsi)별로 시간순 waypoint 방문 순서를 만들고, 연속된 중복은 압축한다
       (같은 waypoint 근처에서 머무는 동안 반복 할당되는 것을 방지).
    3. 압축된 순서에서 연속된 두 waypoint 쌍을 "구간(edge)"으로 기록하고, 여러 선박에 걸쳐
       등장 빈도를 센다.
    4. min_vessel_count 이상 반복된 구간만 "정상 항로 구간"으로 채택한다 (한 번만 지나간
       구간은 우연일 수 있으므로 제외).

    Returns:
        columns=[wp_a, wp_b, lat_a, lon_a, lat_b, lon_b, vessel_count]
    """
    if waypoints.empty:
        return pd.DataFrame(columns=["wp_a", "wp_b", "lat_a", "lon_a", "lat_b", "lon_b", "vessel_count"])

    tree = build_waypoint_tree(waypoints)
    d = df.sort_values(["mmsi", "timestamp"]).copy()
    d["nearest_wp"] = query_nearest_waypoint_ids(d, tree, waypoints)

    edge_counts: dict[tuple[int, int], int] = {}
    for _, group in d.groupby("mmsi"):
        seq = group["nearest_wp"].to_numpy()
        if len(seq) < 2:
            continue
        keep = np.insert(seq[1:] != seq[:-1], 0, True)  # 연속 중복 제거
        compressed = seq[keep]
        for a, b in zip(compressed[:-1], compressed[1:]):
            key = tuple(sorted((int(a), int(b))))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    wp_lookup = waypoints.set_index("waypoint_id")[["lat", "lon"]]
    records = []
    for (a, b), count in edge_counts.items():
        if count >= min_vessel_count and a in wp_lookup.index and b in wp_lookup.index:
            records.append(
                {
                    "wp_a": a,
                    "wp_b": b,
                    "lat_a": wp_lookup.loc[a, "lat"],
                    "lon_a": wp_lookup.loc[a, "lon"],
                    "lat_b": wp_lookup.loc[b, "lat"],
                    "lon_b": wp_lookup.loc[b, "lon"],
                    "vessel_count": count,
                }
            )
    return pd.DataFrame(records)
