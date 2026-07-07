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

    reason = str(anomaly.get("reason", ""))
    is_speed_reason = any(k in reason for k in ["implausible_speed", "statistical_speed_outlier", "속도"])
    is_course_reason = any(k in reason for k in ["sharp_course_change", "statistical_course_outlier", "침로"])

    if is_speed_reason and "implied_speed_knots" in anomaly:
        parts.append(f"역산 속도 {anomaly['implied_speed_knots']}노트로 비정상적으로 높습니다.")
        if "speed_zscore" in anomaly:
            parts.append(f"(같은 선박종류 평균 대비 z-score {anomaly['speed_zscore']})")
    if is_course_reason and "course_change_deg" in anomaly:
        parts.append(f"침로가 {anomaly['course_change_deg']}도 급변했습니다.")
        if "course_zscore" in anomaly:
            parts.append(f"(같은 선박종류 평균 대비 z-score {anomaly['course_zscore']})")

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

    prompt = f"""다음은 선박 AIS(자동식별시스템) 데이터에서 규칙 기반 임계값을 넘어 탐지된
이상 항적 후보입니다. 이건 확정된 이상 행위가 아니라 "추가 확인이 필요한 신호"일 뿐입니다.

해양 모니터링 담당자에게 2~3문장의 한국어로 간결하게 설명해주세요. 다음 원칙을 반드시 지켜주세요:
- 왜 이 임계값을 넘었는지 수치를 근거로 담백하게 설명할 것
- "위조", "불법", "즉시 신고", "밀수" 등 단정적이거나 자극적인 표현은 쓰지 말 것
- GPS 오차, 통신 음영구역, 접안/정박 중 조작 등 정상적인 설명 가능성도 함께 언급할 것
- 마지막에 "추가 확인이 필요하다" 정도의 톤으로 마무리할 것 (신고/조치를 지시하지 말 것)

이상 항적 데이터:
{anomaly}
"""
    # gpt-oss-120b는 reasoning 모델이라 답변 전에 내부 사고 과정에도 토큰을 씀.
    # max_tokens가 너무 낮으면 reasoning만 소비하고 실제 답변(content)이 빈 문자열로
    # 잘리는 경우가 실사용 테스트에서 확인됨 (reasoning 68~104 토큰 관측) -> 여유를 둠.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
    except Exception as e:
        # 네트워크 차단, 인증 실패, rate limit 등 API 호출 자체가 실패하는 경우
        # 예외를 그대로 노출해 앱을 죽이는 대신 폴백 설명으로 안전하게 대체
        return _fallback_explanation(anomaly) + f" (참고: AI 호출 실패로 규칙 기반 설명으로 대체됨 — {type(e).__name__})"

    content = response.choices[0].message.content
    if not content or not content.strip():
        # reasoning 토큰 소비 등으로 빈 응답이 온 경우, 침묵하는 대신 폴백으로 안전하게 대체
        return _fallback_explanation(anomaly) + " (참고: AI 응답이 비어 있어 규칙 기반 설명으로 대체됨)"
    return content.strip()


def explain_anomalies_batch(anomalies: list[dict], model: str = DEFAULT_MODEL) -> list[str]:
    """여러 이상 항적을 순차적으로 설명 생성 (배치 처리시 Groq 무료 티어 rate limit 고려해 사용)."""
    return [explain_anomaly(a, model=model) for a in anomalies]


def answer_question(question: str, context: str, model: str = DEFAULT_MODEL) -> str:
    """대시보드 챗봇에서 사용자의 자유 질문에 대해, 현재 로드된 데이터 요약(context)을
    바탕으로 답변을 생성한다. API 키가 없으면 안내 메시지로 대체한다.
    """
    client = _get_client()
    if client is None:
        return (
            "지금은 Groq API 키가 설정되어 있지 않아 AI 답변을 생성할 수 없어요. "
            "`.env`에 GROQ_API_KEY를 설정하면 이 질문에 답변할 수 있습니다."
        )

    prompt = f"""당신은 선박 AIS 이상탐지 대시보드의 보조 챗봇입니다.
아래는 현재 대시보드에 로드된 데이터의 요약 정보입니다.

{context}

이 정보를 바탕으로 사용자의 질문에 한국어로 간결하고 정확하게 답변하세요.
데이터에 없는 내용은 추측하지 말고 "현재 데이터로는 알 수 없다"고 답하세요.
단정적인 판단(불법, 위조 등)은 피하고, 어떤 근거로 그렇게 판단되는지 설명하세요.

질문: {question}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
    except Exception as e:
        return f"AI 호출에 실패했어요 ({type(e).__name__}). 네트워크 상태나 API 키를 확인해주세요."

    content = response.choices[0].message.content
    if not content or not content.strip():
        return "AI 응답이 비어 있어요. 잠시 후 다시 시도해주세요."
    return content.strip()
