"""위경도 기반 거리/방위각 계산 유틸리티."""

from __future__ import annotations
import numpy as np
from haversine import haversine, Unit

EARTH_RADIUS_KM = 6371.0


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 대권거리(km)를 반환한다 (지구 곡률 고려)."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """지점1에서 지점2로 향하는 초기 방위각(0~360도)을 반환한다."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon_r = np.radians(lon2 - lon1)
    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def bearing_diff(b1: float, b2: float) -> float:
    """두 방위각 간의 최소 차이(0~180도)를 반환한다."""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def _vectorized_distance_km(lat1, lon1, lat2, lon2):
    """numpy 배열 입력을 받는 벡터화된 haversine 거리(km) 계산."""
    R = EARTH_RADIUS_KM
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _vectorized_bearing_deg(lat1, lon1, lat2, lon2):
    """numpy 배열 입력을 받는 벡터화된 초기 방위각(도) 계산."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon_r = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlon_r) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def cross_and_along_track_km(
    point_lat: np.ndarray, point_lon: np.ndarray, start_lat: float, start_lon: float, end_lat: float, end_lon: float
):
    """점(point_lat, point_lon) 배열이 대권 경로(start→end) 선분에서 얼마나 벗어났는지를
    구면 항법의 표준 공식(cross-track / along-track distance)으로 계산한다. 완전히
    벡터화되어 있어 17만 포인트 규모에서도 빠르게 처리된다.

    - cross_track_km: 경로 선(직선이 아니라 대권 경로)에서 수직으로 벗어난 거리(km, 부호 있음)
    - along_track_km: start 지점에서부터 점의 투영 위치까지 경로를 따라간 거리(km)
      (along_track이 0~두 지점 사이 거리 범위 안에 있어야 "선분 위에 투영된다"고 볼 수 있음)

    선박 항로처럼 "직선이 아니라 두 지점을 잇는 대권 경로"에서의 이탈 정도를 재는 데 쓴다.
    출처: 항법에서 널리 쓰이는 cross-track distance / along-track distance 공식
    (Ed Williams, Aviation Formulary).
    """
    R = EARTH_RADIUS_KM
    point_lat = np.asarray(point_lat, dtype=float)
    point_lon = np.asarray(point_lon, dtype=float)

    d13 = _vectorized_distance_km(start_lat, start_lon, point_lat, point_lon) / R
    brng13 = np.radians(_vectorized_bearing_deg(start_lat, start_lon, point_lat, point_lon))
    brng12 = np.radians(bearing_deg(start_lat, start_lon, end_lat, end_lon))

    cross_track_rad = np.arcsin(np.clip(np.sin(d13) * np.sin(brng13 - brng12), -1.0, 1.0))
    cross_track_km = cross_track_rad * R

    cos_along = np.clip(np.cos(d13) / np.cos(cross_track_rad), -1.0, 1.0)
    along_track_km = np.arccos(cos_along) * R

    return cross_track_km, along_track_km
