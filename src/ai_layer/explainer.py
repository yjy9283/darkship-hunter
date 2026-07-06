"""
탐지된 이상 항적을 Groq LLM에 전달해 자연어 리포트를 생성하는 모듈.

Groq는 OpenAI SDK와 호환되므로 openai 클라이언트에 base_url만 바꿔서 사용한다.
API 키가 없는 환경(로컬 개발/테스트)에서도 코드가 죽지 않도록,
키가 없으면 규칙 기반 폴백 설명을 반환한다.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # 2026-07 기준 Groq 권장 모델. deprecation 발생 시 groq 콘솔에서 최신 모델명 확인 필요


def _get_client():
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _fallback_explanation(anomaly: dict) -> str:
    """API 키가 없을 때 사용하는 규칙 기반 설명 (개발/테스트용)."""
    parts = [f"[규칙 기반 폴백] 선박 {anomaly.get('mmsi', '?')}에서 이상 징후 감지."]

    if "gap_minutes" in anomaly:
        parts.append(f"AIS 신호가 {anomaly['gap_minutes']:.0f}분간 끊겼습니다.")

    reason = anomaly.get("reason", "")
    if "implausible_speed" in reason and "implied_speed_knots" in anomaly:
        parts.append(f"역산 속도 {anomaly['implied_speed_knots']}노트로 비정상적으로 높습니다.")
    if "sharp_course_change" in reason and "course_change_deg" in anomaly:
        parts.append(f"침로가 {anomaly['course_change_deg']}도 급변했습니다.")

    if "distance_to_route_km" in anomaly:
        parts.append(f"정상 항로에서 {anomaly['distance_to_route_km']}km 이탈했습니다.")

    return " ".join(parts)


def explain_anomaly(anomaly: dict, model: str = DEFAULT_MODEL) -> str:
    """단일 이상 항적 레코드(dict)를 받아 자연어 설명을 생성한다.

    Args:
        anomaly: detect_* 함수들이 반환하는 DataFrame의 한 행 (dict로 변환된 것)
        model: Groq에서 서빙하는 모델명

    Returns:
        한국어 자연어 설명 문자열
    """
    client = _get_client()
    if client is None:
        return _fallback_explanation(anomaly)

    prompt = f"""다음은 선박 AIS(자동식별시스템) 데이터에서 탐지된 이상 항적 정보입니다.
해양 모니터링 담당자가 바로 이해할 수 있도록 2~3문장의 한국어로 간결하게 설명해주세요.
왜 이게 이상 징후로 판단됐는지, 어떤 조치를 취하면 좋을지 포함해주세요.

이상 항적 데이터:
{anomaly}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def explain_anomalies_batch(anomalies: list[dict], model: str = DEFAULT_MODEL) -> list[str]:
    """여러 이상 항적을 순차적으로 설명 생성 (배치 처리시 Groq 무료 티어 rate limit 고려해 사용)."""
    return [explain_anomaly(a, model=model) for a in anomalies]
