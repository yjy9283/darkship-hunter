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

st.set_page_config(page_title="다크쉽 헌터", page_icon="🚢", layout="wide")
st.title("🚢 다크쉽 헌터 — AIS 이상항적 탐지")

st.markdown(
    "AIS(선박 자동식별시스템) 공개 데이터를 기반으로 정상 항로를 학습하고, "
    "신호 중단·급변침로·항로 이탈 등의 이상 징후를 탐지합니다."
)

uploaded = st.file_uploader("AIS CSV 파일 업로드", type=["csv"])
default_path = Path(__file__).resolve().parents[2] / "data" / "raw"

if uploaded is not None:
    df_raw = load_ais_csv(uploaded)
elif (default_path / "ais_data.csv").exists():
    df_raw = load_ais_csv(default_path / "ais_data.csv")
    st.info(f"업로드 파일이 없어 기본 데이터({default_path / 'ais_data.csv'})를 사용합니다.")
else:
    st.warning("CSV를 업로드하거나 data/raw/ais_data.csv를 준비해주세요.")
    st.stop()

with st.spinner("데이터 정제 중..."):
    df = clean_pipeline(df_raw)
st.success(f"정제 완료: {df['mmsi'].nunique()}척, {len(df)}개 레코드")

col1, col2 = st.columns(2)
with col1:
    eps_km = st.slider("Waypoint 클러스터 반경 (km)", 0.5, 10.0, 1.5, 0.5)
with col2:
    gap_minutes = st.slider("신호중단 기준 (분)", 10, 120, 60, 10)

with st.spinner("정상 항로 학습 중 (DBSCAN)..."):
    waypoints = extract_waypoints(df, eps_km=eps_km, min_samples=10)
st.write(f"학습된 waypoint 수: {len(waypoints)}")

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

    st_folium(m, width=1200, height=600)

with tab2:
    dark_gaps = detect_dark_gaps(df, max_gap_minutes=gap_minutes)
    jumps = detect_kinematic_jumps(df)
    deviations = detect_route_deviation(df, waypoints, threshold_km=15.0)

    st.subheader(f"🔇 신호 중단 ({len(dark_gaps)}건)")
    st.dataframe(dark_gaps, use_container_width=True)

    st.subheader(f"⚡ 급변 침로/속도 ({len(jumps)}건)")
    st.dataframe(jumps, use_container_width=True)

    st.subheader(f"📍 항로 이탈 ({len(deviations)}건)")
    st.dataframe(deviations, use_container_width=True)

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
