"""
AIS 데이터 정제 모듈.

- 잘못된 좌표 제거 (위경도 범위 밖, 0,0 좌표 등)
- 중복 레코드 제거
- 선박별(MMSI) 궤적 정렬
- 너무 짧은 궤적(포인트 수 부족) 제거
"""

from __future__ import annotations
import pandas as pd


def remove_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """위경도 범위를 벗어나거나 (0,0) 근처인 좌표를 제거한다."""
    mask = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & ~((df["lat"].abs() < 0.01) & (df["lon"].abs() < 0.01))
        & df["lat"].notna()
        & df["lon"].notna()
        & df["timestamp"].notna()
    )
    return df[mask].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """같은 선박, 같은 시각 레코드 중복 제거 (마지막 값 유지)."""
    return df.drop_duplicates(subset=["mmsi", "timestamp"], keep="last").copy()


def sort_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """선박별로 시간순 정렬."""
    return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


def filter_short_trajectories(df: pd.DataFrame, min_points: int = 5) -> pd.DataFrame:
    """포인트 수가 min_points 미만인 선박(궤적)은 통계적으로 의미가 없으므로 제외."""
    counts = df.groupby("mmsi").size()
    valid_mmsi = counts[counts >= min_points].index
    return df[df["mmsi"].isin(valid_mmsi)].copy()


def clean_pipeline(df: pd.DataFrame, min_points: int = 5) -> pd.DataFrame:
    """전체 정제 파이프라인을 순서대로 적용."""
    df = remove_invalid_coordinates(df)
    df = remove_duplicates(df)
    df = sort_trajectories(df)
    df = filter_short_trajectories(df, min_points=min_points)
    return df
