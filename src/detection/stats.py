"""
통계적 임계값 계산 유틸리티.

기존 이상탐지는 "속도 40노트 초과", "침로변화 90도 초과"처럼 고정된 임의 임계값을
모든 선박에 일괄 적용했다. 하지만 실제 데이터로 검증해보니(트러블슈팅 로그 참고)
선박종류마다 정상 행동 범위가 크게 다르다 — 예인선/파일럿선은 원래 자주 급선회하고,
화물선/유조선은 거의 직선 항로를 유지한다.

이 모듈은 "고정 임계값"이 아니라 "같은 선박종류 집단 내에서 통계적으로 얼마나 벗어났는가
(z-score)"를 계산해서, 각 선박종류의 정상 행동 baseline 대비 이상 여부를 판단할 수 있게 한다.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def group_zscore(values: pd.Series, group_keys: pd.Series, min_group_size: int = 30) -> pd.Series:
    """그룹(선박종류)별 평균/표준편차 기준 z-score를 계산한다.

    그룹 표본 수가 min_group_size 미만이면 통계적으로 신뢰하기 어려우므로,
    전체 데이터(global) 평균/표준편차로 대체(fallback)한다.

    Args:
        values: z-score를 계산할 값(예: implied_speed_knots)
        group_keys: 그룹 기준(예: ship_type)
        min_group_size: 그룹별 통계를 신뢰하기 위한 최소 표본 수

    Returns:
        values와 같은 인덱스를 가지는 z-score Series
    """
    df = pd.DataFrame({"value": values, "group": group_keys})
    group_sizes = df.groupby("group")["value"].transform("count")

    group_mean = df.groupby("group")["value"].transform("mean")
    group_std = df.groupby("group")["value"].transform("std")

    global_mean = df["value"].mean()
    global_std = df["value"].std()

    # 그룹 표본이 부족하거나 표준편차가 0에 가까우면(값이 거의 고정된 그룹) 전체 통계로 대체
    use_global = (group_sizes < min_group_size) | group_std.isna() | (group_std < 1e-6)
    mean_used = group_mean.where(~use_global, global_mean)
    std_used = group_std.where(~use_global, global_std)
    std_used = std_used.replace(0, np.nan)

    return (df["value"] - mean_used) / std_used


def percentile_rank(values: pd.Series) -> pd.Series:
    """각 값이 전체 분포에서 상위 몇 %에 해당하는지 반환한다 (0~100, 낮을수록 극단값)."""
    return 100 - values.rank(pct=True) * 100
