# 🚢 DarkShip Hunter

> AIS(선박 자동식별시스템) 공개 데이터를 기반으로, 선박의 이상 항적을 탐지하고 AI가 자연어로 설명해주는 프로젝트

## 문제의식

선박은 AIS 신호로 위치·속도·침로를 실시간 송출하지만, 불법조업·밀수·해적 활동 등의 이유로 일부러 신호를 꺼버리는 "다크 베슬(Dark Vessel)" 현상이 실제 해양 보안 연구에서 다뤄지는 문제입니다. 이 프로젝트는 공개 AIS 데이터를 활용해 이런 이상 항적을 탐지하는 파이프라인을 직접 구현합니다.

## 데이터 출처

- Kaggle "AIS Dataset" (eminserkanerdonmez) — 2022년 1~3월 카테가트 해협(덴마크-스웨덴) 실제 선박 통항 데이터
- 백업: [NOAA MarineCadastre](https://marinecadastre.gov/accessais)

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
