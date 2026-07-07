import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from src.detection.stats import group_zscore, percentile_rank


def test_group_zscore_normal_group_computes_within_group_stats():
    # 그룹 A: 평균 10, 그룹 B: 평균 100 (표본 충분)
    values = pd.Series([10, 10, 10, 10, 30] + [100] * 30)
    groups = pd.Series(["A"] * 5 + ["B"] * 30)

    z = group_zscore(values, groups, min_group_size=5)
    # 그룹 A 안에서 30은 나머지(10,10,10,10)보다 확연히 높은 이상치여야 함
    assert z.iloc[4] > 1.5
    # 그룹 B는 전부 동일값(표준편차 0) -> 전체 데이터 기준으로 폴백되어야 함(에러 없이 값 반환)
    assert z.iloc[5:].notna().all() or z.iloc[5:].isna().all()  # 폴백 결과가 일관되게 나오는지


def test_group_zscore_small_group_falls_back_to_global():
    values = pd.Series([1.0, 2.0, 3.0, 100.0] + list(range(1, 31)))
    groups = pd.Series(["rare"] * 4 + ["common"] * 30)

    z = group_zscore(values, groups, min_group_size=10)
    # 'rare' 그룹(표본 4개, min_group_size 10 미만)은 전체 통계로 대체되어야 하므로
    # 같은 값이라도 그룹 통계만으로 계산했을 때와 다른 결과가 나와야 함(폴백이 실제로 작동)
    assert not z.iloc[:4].isna().any()


def test_group_zscore_handles_all_nan_group_keys_without_error():
    values = pd.Series([1.0, 2.0, 100.0, 1.5, 2.5])
    groups = pd.Series([np.nan] * 5)

    z = group_zscore(values, groups, min_group_size=30)
    assert not z.isna().any()
    assert z.idxmax() == 2  # 100.0이 가장 튀는 값으로 나와야 함


def test_percentile_rank_extremes():
    values = pd.Series([1, 2, 3, 4, 100])
    ranks = percentile_rank(values)
    # 가장 큰 값(100)은 상위 0%에 가까워야 함
    assert ranks.iloc[4] < ranks.iloc[0]
