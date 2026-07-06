"""
AIS 원본 CSV를 표준 스키마로 로드하는 모듈.

표준 스키마:
    mmsi        : str   - 선박 고유 식별번호
    timestamp   : datetime
    lat         : float
    lon         : float
    sog         : float - Speed Over Ground (knots)
    cog         : float - Course Over Ground (degrees)
    heading     : float - (optional, NaN 가능)
    nav_status  : str   - (optional)
    ship_type   : str   - (optional)

원본 데이터셋마다 컬럼명이 제각각이라, COLUMN_ALIASES에서 자주 쓰이는
변형들을 표준명으로 매핑한다. 새로운 데이터셋을 추가할 땐 이 딕셔너리에
별칭만 추가하면 된다 (오픈소스 확장 포인트).
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

STANDARD_COLUMNS = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "heading", "nav_status", "ship_type"]

COLUMN_ALIASES: dict[str, list[str]] = {
    "mmsi": ["mmsi", "MMSI", "vessel_id", "ship_id"],
    "timestamp": ["timestamp", "BaseDateTime", "datetime", "time", "date_time", "Timestamp"],
    "lat": ["lat", "LAT", "latitude", "Latitude"],
    "lon": ["lon", "LON", "longitude", "Longitude", "lng"],
    "sog": ["sog", "SOG", "speed", "Speed", "speed_over_ground"],
    "cog": ["cog", "COG", "course", "Course", "course_over_ground"],
    "heading": ["heading", "Heading", "HEADING", "true_heading"],
    "nav_status": ["nav_status", "Status", "navigational_status", "NavigationalStatus"],
    "ship_type": ["ship_type", "VesselType", "type", "Type", "vessel_type"],
}


def _build_rename_map(columns: list[str]) -> dict[str, str]:
    rename_map = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                rename_map[alias] = standard_name
                break
    return rename_map


def load_ais_csv(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """AIS CSV 파일을 로드하고 표준 스키마로 변환한다.

    Args:
        path: CSV 파일 경로
        column_map: 자동 인식이 실패할 경우 수동으로 {원본컬럼명: 표준컬럼명} 지정

    Returns:
        표준 스키마를 따르는 DataFrame (없는 optional 컬럼은 NaN으로 채움)
    """
    df = pd.read_csv(path)

    rename_map = column_map if column_map else _build_rename_map(list(df.columns))
    df = df.rename(columns=rename_map)

    missing_required = [c for c in ["mmsi", "timestamp", "lat", "lon"] if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"필수 컬럼을 찾을 수 없습니다: {missing_required}. "
            f"원본 컬럼: {list(df.columns)}. column_map 인자로 수동 매핑하세요."
        )

    for optional_col in ["sog", "cog", "heading", "nav_status", "ship_type"]:
        if optional_col not in df.columns:
            df[optional_col] = pd.NA

    df["mmsi"] = df["mmsi"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for numeric_col in ["lat", "lon", "sog", "cog", "heading"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    return df[STANDARD_COLUMNS].copy()
