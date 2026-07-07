"""
다크쉽 헌터 대시보드.

실행: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from src.preprocessing.loader import load_ais_csv
from src.preprocessing.clean import clean_pipeline
from src.detection.routes import extract_waypoints
from src.detection.anomaly import detect_dark_gaps, detect_kinematic_jumps, detect_route_deviation
from src.ai_layer.explainer import explain_anomaly


@st.cache_data(show_spinner=False)
def _load_and_clean(file_bytes_or_path):
    df_raw = load_ais_csv(file_bytes_or_path)
    return clean_pipeline(df_raw)


@st.cache_data(show_spinner=False)
def _cached_waypoints(df: pd.DataFrame, eps_km: float, min_samples: int = 10):
    return extract_waypoints(df, eps_km=eps_km, min_samples=min_samples)


@st.cache_data(show_spinner=False)
def _cached_dark_gaps(df: pd.DataFrame, gap_minutes: float):
    return detect_dark_gaps(df, max_gap_minutes=gap_minutes)


@st.cache_data(show_spinner=False)
def _cached_jumps(df: pd.DataFrame):
    return detect_kinematic_jumps(df)


@st.cache_data(show_spinner=False)
def _cached_deviations(df: pd.DataFrame, waypoints: pd.DataFrame, threshold_km: float):
    return detect_route_deviation(df, waypoints, threshold_km=threshold_km)

st.set_page_config(page_title="다크쉽 헌터", page_icon="🚢", layout="wide")
st.title("🚢 다크쉽 헌터 — AIS 이상항적 탐지")

st.markdown(
    "AIS(선박 자동식별시스템) 공개 데이터를 기반으로 정상 항로를 학습하고, "
    "신호 중단·급변침로·항로 이탈 등의 이상 징후를 탐지합니다."
)

uploaded = st.file_uploader("AIS CSV 파일 업로드", type=["csv"])
default_path = Path(__file__).resolve().parents[2] / "data" / "raw"

if uploaded is not None:
    df = _load_and_clean(uploaded)
elif (default_path / "ais_data.csv").exists():
    df = _load_and_clean(str(default_path / "ais_data.csv"))
    st.info(f"업로드 파일이 없어 기본 데이터({default_path / 'ais_data.csv'})를 사용합니다.")
else:
    st.warning("CSV를 업로드하거나 data/raw/ais_data.csv를 준비해주세요.")
    st.stop()

st.success(f"정제 완료: {df['mmsi'].nunique()}척, {len(df)}개 레코드")

col1, col2 = st.columns(2)
with col1:
    eps_km = st.slider("Waypoint 클러스터 반경 (km)", 0.5, 10.0, 1.5, 0.5)
with col2:
    gap_minutes = st.slider("신호중단 기준 (분)", 10, 120, 60, 10)

waypoints = _cached_waypoints(df, eps_km)
st.write(f"학습된 waypoint 수: {len(waypoints)}")

# 지도(st_folium)는 팬/줌/클릭 시마다 스크립트를 재실행시키는 컴포넌트라서,
# 캐싱 없이는 지도를 조작할 때마다 정제/항로학습/이상탐지가 전부 처음부터
# 다시 돌아 "매번 새로 시작"하는 것처럼 보이는 문제가 있었음 — 위 캐싱으로 해결.
dark_gaps = _cached_dark_gaps(df, gap_minutes)
jumps = _cached_jumps(df)
deviations = _cached_deviations(df, waypoints, 15.0)

tab1, tab2, tab3 = st.tabs(["🗺️ 지도", "⚠️ 이상 리스트", "📊 통계"])

with tab1:
    center_lat, center_lon = df["lat"].mean(), df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    for _, wp in waypoints.iterrows():
        folium.CircleMarker(
            location=[wp["lat"], wp["lon"]],
            radius=3 + min(wp["point_count"] / 50, 10),
            color="gray",
            fill=True,
            fill_opacity=0.4,
            popup=f"Waypoint (포인트 {wp['point_count']}개)",
        ).add_to(m)

    for mmsi, group in df.groupby("mmsi"):
        coords = group[["lat", "lon"]].values.tolist()
        folium.PolyLine(coords, color="blue", weight=1, opacity=0.5, popup=f"MMSI {mmsi}").add_to(m)

    st_folium(m, width=1200, height=600, returned_objects=[])

with tab2:
    st.subheader(f"🔇 신호 중단 ({len(dark_gaps)}건)")
    st.dataframe(dark_gaps, width='stretch')

    st.subheader(f"⚡ 급변 침로/속도 ({len(jumps)}건)")
    st.dataframe(jumps, width='stretch')

    st.subheader(f"📍 항로 이탈 ({len(deviations)}건)")
    st.dataframe(deviations, width='stretch')

    if st.button("선택된 이상 항적 AI 설명 생성 (상위 5건)"):
        combined = pd.concat(
            [dark_gaps.head(2), jumps.head(2), deviations.head(1)], ignore_index=True, sort=False
        )
        for _, row in combined.iterrows():
            with st.expander(f"MMSI {row.get('mmsi')} — 이상 상세"):
                st.write(explain_anomaly(row.dropna().to_dict()))

with tab3:
    st.metric("선박 수", df["mmsi"].nunique())
    st.metric("총 레코드 수", len(df))
    st.metric("신호중단 건수", len(dark_gaps))
    st.metric("급변 건수", len(jumps))
    st.metric("항로이탈 건수", len(deviations))
