# 🚢 DarkShip Hunter

> AIS(선박 자동식별시스템) 공개 데이터를 기반으로, 선박의 이상 항적을 탐지하고 AI가 자연어로 설명해주는 프로젝트

## 문제의식

선박은 AIS 신호로 위치·속도·침로를 실시간 송출하지만, 불법조업·밀수·해적 활동 등의 이유로 일부러 신호를 꺼버리는 "다크 베슬(Dark Vessel)" 현상이 실제 해양 보안 연구에서 다뤄지는 문제입니다. 이 프로젝트는 공개 AIS 데이터를 활용해 이런 이상 항적을 탐지하는 파이프라인을 직접 구현합니다.

## 데이터 출처

- **NOAA MarineCadastre (공식 공개 데이터)**: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html
  - 미국 연안 전체 하루치 AIS 데이터 (2024-01-01, 약 730만 행)
  - 이 중 샌프란시스코 베이 지역(위경도 필터링)만 추출해 사용: 약 17.4만 행, 428척
  - 컬럼: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType 등
- (참고) Kaggle "AIS Dataset" (eminserkanerdonmez)도 검토했으나, 실제 다운로드해보니 위경도/시간 컬럼이 빠진
  스냅샷 형태라 이 프로젝트 목적(위치 기반 이상탐지)에 맞지 않아 NOAA 공식 데이터로 전환함.

## 실행 결과 (2024-01-01 SF Bay 데이터 기준)

| 지표 | 결과 |
|---|---|
| 정제 후 레코드 | 174,221행 / 428척 |
| 학습된 waypoint | 14개 (eps=1.5km, min_samples=10) |
| 신호중단 (60분 이상) | 62건 (41척) |
| 급변침로/속도 | 802건 (161척) |
| 항로 이탈 (15km 기준) | 약 6.6% (서브샘플 기준) |

## 튜닝 과정에서 발견한 한계점 (정직하게 명시)

1. **정박 중 COG 노이즈 문제**: 처음엔 급변침로가 9,569건으로 과다 탐지됐는데, 원인은 정박/저속 상태에서
   COG(침로) 값 자체가 GPS 노이즈로 흔들리는 것이었음. 저속(2노트 미만) 구간은 침로 급변 판정에서
   제외하도록 수정해 802건으로 정상화.
2. **점 클러스터 기반 항로 모델의 한계**: waypoint를 "정박지/밀집구역"의 점으로만 학습하다 보니,
   두 항구 사이를 정상적으로 항해하는 구간도 "가장 가까운 hotspot에서 멀다"는 이유로 이탈로
   오탐될 수 있음. 실제 항로(선, 회랑)를 학습하는 방식(TREAD 등)보다 단순화된 접근이라는 점을 인지하고 있음.
3. **밀집구역 메모리 이슈**: DBSCAN을 전체 포인트에 그대로 적용하면 정박지처럼 극도로 밀집된 구역에서
   메모리 사용량이 급증(10만행에 3.2GB)해서, waypoint 학습용으로는 대표 샘플(최대 3만 포인트)만 사용하도록 조정.
4. **Groq reasoning 모델의 빈 응답 이슈**: `openai/gpt-oss-120b`는 reasoning 모델이라 답변 전에
   내부 사고 과정에도 토큰을 소비함. `max_tokens`가 낮으면(300) reasoning만 소비하고 실제 답변이
   빈 문자열로 반환되는 경우가 실사용 테스트에서 확인됨 (reasoning 68~104 토큰 관측).
   `max_tokens`를 500으로 늘리고, 빈 응답 시 조용히 실패하는 대신 규칙 기반 폴백으로
   안전하게 대체하는 로직을 추가함 (`tests/test_explainer.py`에서 mock으로 검증).

## Groq 연동 실사용 테스트 결과

- SF Bay 데이터의 실제 이상 항적(신호중단, 급변침로)으로 Groq API 호출 성공 확인
- 4~5건 정상 응답 확인, 1건 빈 응답 케이스 발견 및 원인 규명(위 4번 항목) 후 수정 완료

## 아키텍처

```
AIS 원본 데이터
  → 전처리 (노이즈 제거, 궤적 재구성)
  → 정상 항로 학습 (DBSCAN)
  → 이상탐지 (신호중단 / 급변침로 / 항로이탈)
  → AI 설명 생성 (Groq)
  → 지도 시각화 (Streamlit + Folium)
```

## 기술 스택

- Python, pandas, numpy, scikit-learn, haversine
- Groq API (gpt-oss-120b)
- Streamlit, Folium

## 한계점 (정직하게 명시)

- 실시간 스트리밍이 아닌 히스토리컬 공개 데이터 사용
- 좁은 해역 데이터로 검증, 라벨링된 정답 없이 비지도 학습 방식 채택
- 탐지 성능은 정성적 사례 분석으로 검증 (정량적 벤치마크 아님)

## 로드맵

- [ ] 데이터 전처리 & EDA
- [ ] 정상 항로 클러스터링 (DBSCAN)
- [ ] 이상탐지 엔진
- [ ] AI 설명 레이어 (Groq)
- [ ] 대시보드 (Streamlit)

## 참고 자료

- TREAD (Traffic Route Extraction and Anomaly Detection) 논문 방법론
- [LeoPits/Vessels-anomaly-detection-with-AIS-data](https://github.com/LeoPits/Vessels-anomaly-detection-with-AIS-data) (구조 참고용)
