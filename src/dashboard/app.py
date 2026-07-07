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
from src.preprocessing.vessel_types import get_vessel_type_name, translate_reason
from src.detection.routes import extract_waypoints, build_corridor_edges
from src.detection.anomaly import (
    detect_dark_gaps, detect_kinematic_jumps_statistical, detect_route_deviation_corridor, score_with_isolation_forest,
)
from src.ai_layer.explainer import explain_anomaly, answer_question


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
def _cached_jumps(df: pd.DataFrame, ship_type_map: pd.Series, z_thresh: float = 3.0):
    return detect_kinematic_jumps_statistical(df, ship_type_map, z_thresh=z_thresh)


@st.cache_data(show_spinner=False)
def _cached_corridors(df: pd.DataFrame, waypoints: pd.DataFrame):
    return build_corridor_edges(df, waypoints, min_vessel_count=2)


@st.cache_data(show_spinner=False)
def _cached_deviations(df: pd.DataFrame, waypoints: pd.DataFrame, corridors: pd.DataFrame, threshold_km: float):
    return detect_route_deviation_corridor(df, waypoints, corridors, threshold_km=threshold_km)


@st.cache_data(show_spinner=False)
def _cached_ml_score(df: pd.DataFrame, waypoints: pd.DataFrame):
    return score_with_isolation_forest(df, waypoints)


# --- 표시용 컬럼명 한글 매핑 (원본 데이터/로직은 영문 그대로, 화면 표시만 한글화) ---
DARK_GAP_COLS_KO = {
    "mmsi": "MMSI", "gap_start": "중단 시작", "gap_end": "중단 종료",
    "gap_minutes": "중단시간(분)", "lat_before": "위도(전)", "lon_before": "경도(전)",
    "lat_after": "위도(후)", "lon_after": "경도(후)", "ship_type_name": "선박종류",
}
JUMP_COLS_KO = {
    "mmsi": "MMSI", "timestamp": "시각", "lat": "위도", "lon": "경도",
    "implied_speed_knots": "역산속도(노트)", "speed_zscore": "속도 Z-점수", "speed_percentile": "속도 상위(%)",
    "course_change_deg": "침로변화(도)", "course_zscore": "침로 Z-점수", "course_percentile": "침로 상위(%)",
    "reason": "사유", "ship_type_name": "선박종류",
}
DEVIATION_COLS_KO = {
    "mmsi": "MMSI", "timestamp": "시각", "lat": "위도", "lon": "경도",
    "distance_to_route_km": "항로이탈거리(km)", "ship_type_name": "선박종류",
}


def _add_ship_type_and_translate(df_result: pd.DataFrame, ship_type_map: pd.Series, is_jump: bool = False) -> pd.DataFrame:
    """이상탐지 결과에 한글 선박종류 컬럼을 붙이고, jump의 경우 reason도 한글화."""
    out = df_result.copy()
    if "mmsi" in out.columns and len(out) > 0:
        out["ship_type_name"] = out["mmsi"].map(ship_type_map).apply(get_vessel_type_name)
    if is_jump and "reason" in out.columns:
        out["reason"] = out["reason"].apply(translate_reason)
    return out


st.set_page_config(page_title="다크쉽 헌터", page_icon="🚢", layout="wide")
st.title("🚢 다크쉽 헌터 — AIS 이상항적 탐지")

st.markdown(
    "AIS(선박 자동식별시스템) 공개 데이터를 기반으로 정상 항로를 학습하고, "
    "신호 중단·급변침로·항로 이탈 등의 이상 징후를 탐지합니다."
)

with st.expander("ℹ️ 각 이상 유형은 무엇을 기준으로 판단하나요? (꼭 읽어보세요)"):
    st.markdown(
        """
- **🔇 신호 중단**: 같은 선박의 AIS 신호가 설정한 시간(기본 60분) 이상 끊겼다가 다시 나타난 경우.
  국제표준(ITU-R M.1371)상 AIS는 항해 중 2~10초, **정박 중이어도 최대 3분** 간격으로 신호를
  보내도록 규정되어 있습니다. 60분 끊김은 정상 케이스 중 가장 느린 경우(3분)보다도 20배 긴
  시간이라 통계적 근거가 있는 기준입니다. 다만 GPS 음영구역·장비 문제로도 발생할 수 있어
  이것만으로 의심 선박이라 단정할 수 없습니다.
- **⚡ 급변침로/속도 (통계 기반)**: 고정된 임계값 대신, **같은 선박종류(예인선, 화물선 등) 집단
  안에서 평균·표준편차 대비 얼마나 벗어났는지(z-score)** 를 기준으로 판단합니다. 예를 들어
  예인선은 원래 자주 급선회하기 때문에 예인선끼리 비교하고, 화물선은 화물선끼리 비교합니다.
  |z-score| > 3(표준편차 3배 이상, 대략 상위 0.3% 이내)이면서 동시에 물리적으로도 의미 있는
  크기(속도 25노트 또는 침로변화 60도 이상)일 때만 이상으로 판단합니다. 표본이 적은 선박종류는
  전체 데이터 기준으로 자동 대체됩니다.
- **📍 항로 이탈 (corridor 기반)**: waypoint(밀집구역)뿐 아니라, 실제 여러 선박이 오간
  waypoint 사이 이동 구간(corridor)까지 학습해서, 거기서 15km 이상 떨어진 경우만 이탈로 판단.
  waypoint 점만 보던 이전 방식보다 오탐이 약 22% 줄었지만, 학습된 corridor 범위 밖(원양 등)의
  항해는 여전히 오탐될 수 있습니다.

모든 탐지는 **통계적으로 드문 신호**일 뿐, 실제 불법/이상 행위를 확정하는 것이 아닙니다.
        """
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

ship_type_map = df.groupby("mmsi")["ship_type"].first()

# 지도(st_folium)는 팬/줌/클릭 시마다 스크립트를 재실행시키는 컴포넌트라서,
# 캐싱 없이는 지도를 조작할 때마다 정제/항로학습/이상탐지가 전부 처음부터
# 다시 돌아 "매번 새로 시작"하는 것처럼 보이는 문제가 있었음 — 캐싱으로 해결.
dark_gaps = _cached_dark_gaps(df, gap_minutes)
jumps = _cached_jumps(df, ship_type_map)
corridors = _cached_corridors(df, waypoints)
deviations = _cached_deviations(df, waypoints, corridors, 15.0)

dark_gaps_ko = _add_ship_type_and_translate(dark_gaps, ship_type_map)
jumps_ko = _add_ship_type_and_translate(jumps, ship_type_map, is_jump=True)
deviations_ko = _add_ship_type_and_translate(deviations, ship_type_map)

# --- 필터 ---
st.markdown("### 🔍 필터")
fcol1, fcol2 = st.columns(2)
all_ship_types = sorted(set(dark_gaps_ko.get("ship_type_name", pd.Series(dtype=str))).union(
    jumps_ko.get("ship_type_name", pd.Series(dtype=str))
).union(deviations_ko.get("ship_type_name", pd.Series(dtype=str))))
with fcol1:
    selected_types = st.multiselect("선박종류 필터", options=all_ship_types, default=[])
with fcol2:
    mmsi_search = st.text_input("MMSI 검색 (일부 입력 가능)", value="")


def _apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    out = d
    if selected_types and "ship_type_name" in out.columns:
        out = out[out["ship_type_name"].isin(selected_types)]
    if mmsi_search:
        out = out[out["mmsi"].astype(str).str.contains(mmsi_search, na=False)]
    return out


dark_gaps_f = _apply_filters(dark_gaps_ko)
jumps_f = _apply_filters(jumps_ko)
deviations_f = _apply_filters(deviations_ko)

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 지도", "⚠️ 이상 리스트", "📊 통계", "💬 AI에게 물어보기"])

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

    for _, edge in corridors.iterrows():
        folium.PolyLine(
            [[edge["lat_a"], edge["lon_a"]], [edge["lat_b"], edge["lon_b"]]],
            color="green",
            weight=3,
            opacity=0.6,
            popup=f"Corridor ({edge['vessel_count']}척 이용)",
        ).add_to(m)

    for mmsi, group in df.groupby("mmsi"):
        coords = group[["lat", "lon"]].values.tolist()
        folium.PolyLine(coords, color="#3388ff", weight=1, opacity=0.4, popup=f"MMSI {mmsi}").add_to(m)

    # 이상 항적 강조 표시 (색상 구분): 신호중단=빨강, 급변=주황, 항로이탈=보라
    for _, row in dark_gaps_f.iterrows():
        folium.CircleMarker(
            location=[row["lat_after"], row["lon_after"]], radius=6, color="red", fill=True,
            fill_opacity=0.8, popup=f"신호중단 MMSI {row['mmsi']} ({row['gap_minutes']:.0f}분)",
        ).add_to(m)
    for _, row in jumps_f.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=5, color="orange", fill=True,
            fill_opacity=0.8, popup=f"급변 MMSI {row['mmsi']} ({row['reason']})",
        ).add_to(m)
    deviations_sample = deviations_f.sample(min(300, len(deviations_f)), random_state=1) if len(deviations_f) > 0 else deviations_f
    for _, row in deviations_sample.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=4, color="purple", fill=True,
            fill_opacity=0.6, popup=f"이탈 MMSI {row['mmsi']} ({row['distance_to_route_km']}km)",
        ).add_to(m)

    st.caption("🔴 신호중단 · 🟠 급변침로/속도 · 🟣 항로이탈 (최대 300건 샘플) · ⚪ waypoint · 🟢 corridor(학습된 정상 이동구간) · 파란선 항적")
    st_folium(m, width=1200, height=600, returned_objects=[])

with tab2:
    st.subheader(f"🔇 신호 중단 ({len(dark_gaps_f)}건 / 전체 {len(dark_gaps)}건)")
    st.dataframe(dark_gaps_f.rename(columns=DARK_GAP_COLS_KO), width='stretch')

    st.subheader(f"⚡ 급변 침로/속도 ({len(jumps_f)}건 / 전체 {len(jumps)}건)")
    st.dataframe(jumps_f.rename(columns=JUMP_COLS_KO), width='stretch')

    st.subheader(f"📍 항로 이탈 ({len(deviations_f)}건 / 전체 {len(deviations)}건)")
    st.dataframe(deviations_f.rename(columns=DEVIATION_COLS_KO), width='stretch')

    if st.button("선택된 이상 항적 AI 설명 생성 (상위 5건)"):
        combined = pd.concat(
            [dark_gaps_f.head(2), jumps_f.head(2), deviations_f.head(1)], ignore_index=True, sort=False
        )
        for _, row in combined.iterrows():
            with st.expander(f"MMSI {row.get('mmsi')} ({row.get('ship_type_name', '미상')}) — 이상 상세"):
                st.write(explain_anomaly(row.dropna().to_dict()))

with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("선박 수", df["mmsi"].nunique())
    m2.metric("총 레코드 수", len(df))
    m3.metric("신호중단 건수", len(dark_gaps))
    m4.metric("급변 건수", len(jumps))

    st.markdown("#### 선박종류별 이상탐지율 (척당 평균 건수)")
    total_by_type = ship_type_map.apply(get_vessel_type_name).value_counts()
    jump_by_type = jumps_ko["ship_type_name"].value_counts() if len(jumps_ko) else pd.Series(dtype=int)
    stat_rows = []
    for t, total in total_by_type.items():
        j = jump_by_type.get(t, 0)
        stat_rows.append({"선박종류": t, "전체척수": total, "급변건수": j, "척당급변율": round(j / total, 2)})
    stat_df = pd.DataFrame(stat_rows).sort_values("척당급변율", ascending=False)
    st.dataframe(stat_df, width='stretch')
    st.bar_chart(stat_df.set_index("선박종류")["척당급변율"])

    st.markdown("#### 🤖 종합 ML 이상탐지 (IsolationForest)")
    st.caption(
        "속도·침로변화·항로이탈도 3가지 지표를 종합해서, 개별 규칙으로는 애매해도 "
        "여러 지표가 복합적으로 이상한 지점을 잡아냅니다. 상위 15개 선박(이상 포인트 많은 순)."
    )
    ml_scored = _cached_ml_score(df, waypoints)
    ml_by_vessel = (
        ml_scored[ml_scored["is_anomaly"]].groupby("mmsi").size().sort_values(ascending=False).head(15)
    )
    ml_table = pd.DataFrame({"MMSI": ml_by_vessel.index, "ML이상포인트수": ml_by_vessel.values})
    ml_table["선박종류"] = ml_table["MMSI"].map(ship_type_map).apply(get_vessel_type_name)
    st.dataframe(ml_table, width='stretch')

with tab4:
    st.markdown("현재 로드된 데이터에 대해 자유롭게 질문해보세요. (예: '가장 이상한 선박이 뭐야?', '신호중단이 가장 긴 배는?')")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_q = st.chat_input("질문을 입력하세요")
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("user"):
            st.write(user_q)

        context = f"""
- 전체 선박 수: {df['mmsi'].nunique()}척, 총 레코드 {len(df)}개
- 신호중단 탐지: {len(dark_gaps)}건 (기준: {gap_minutes}분 이상)
- 급변침로/속도 탐지: {len(jumps)}건
- 항로이탈 탐지: {len(deviations)}건 (기준: waypoint로부터 15km 이상)
- 신호중단 상위 5건 (MMSI, 중단시간(분)): {dark_gaps.nlargest(5, 'gap_minutes')[['mmsi','gap_minutes']].to_dict('records') if len(dark_gaps) else '없음'}
- 급변 상위 5건 (MMSI, 역산속도, 침로변화, 사유): {jumps.head(5)[['mmsi','implied_speed_knots','course_change_deg','reason']].to_dict('records') if len(jumps) else '없음'}
"""
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                answer = answer_question(user_q, context)
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))
