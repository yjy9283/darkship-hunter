"""위경도 기반 거리/방위각 계산 유틸리티."""

from __future__ import annotations
import numpy as np
from haversine import haversine, Unit


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
