"""AIS VesselType 코드(국제표준)를 한글 표기로 변환하는 유틸리티."""

from __future__ import annotations

VESSEL_TYPE_NAMES: dict[int, str] = {
    30: "어선",
    31: "예인선",
    32: "예인선(대형)",
    33: "준설선",
    34: "다이빙작업선",
    35: "군함",
    36: "범선",
    37: "레저보트",
    40: "고속선",
    50: "파일럿선",
    51: "수색구조선",
    52: "예인/터그선",
    53: "항만관제선",
    54: "방제선(오염대응)",
    55: "법집행선",
    60: "여객선",
    61: "여객선",
    62: "여객선",
    63: "여객선",
    64: "여객선",
    65: "여객선",
    66: "여객선",
    67: "여객선",
    68: "여객선",
    69: "여객선",
    70: "화물선",
    71: "화물선(위험물A)",
    72: "화물선(위험물B)",
    73: "화물선(위험물C)",
    74: "화물선(위험물D)",
    80: "유조선",
    81: "유조선(위험물A)",
    82: "유조선(위험물B)",
    83: "유조선(위험물C)",
    84: "유조선(위험물D)",
    90: "기타선박",
}


def get_vessel_type_name(code) -> str:
    """VesselType 코드를 한글명으로 변환. 알 수 없는 코드는 '코드N'으로 표기."""
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return "미상"
    return VESSEL_TYPE_NAMES.get(code_int, f"코드{code_int}")


REASON_NAMES: dict[str, str] = {
    "implausible_speed": "비정상 속도",
    "sharp_course_change": "급선회",
}


def translate_reason(reason: str) -> str:
    """detect_kinematic_jumps의 reason 필드(콤마 구분 영문 코드)를 한글로 변환."""
    if not reason:
        return ""
    parts = [REASON_NAMES.get(r.strip(), r.strip()) for r in reason.split(",")]
    return ", ".join(parts)
