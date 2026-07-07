"""
explainer.py의 안전장치 테스트: Groq가 reasoning 토큰 소모로 빈 content를
반환하는 경우, 조용히 실패하지 않고 폴백 설명으로 대체되는지 검증한다.
(실제 Groq 네트워크 호출 없이 mock으로 재현)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ai_layer.explainer import explain_anomaly


def _make_mock_response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_response


@patch("src.ai_layer.explainer._get_client")
def test_explain_anomaly_falls_back_when_content_is_empty(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("")
    mock_get_client.return_value = mock_client

    result = explain_anomaly({"mmsi": "123", "gap_minutes": 70})

    assert "규칙 기반 폴백" in result
    assert "비어" in result  # 빈 응답이었다는 안내가 포함되는지


@patch("src.ai_layer.explainer._get_client")
def test_explain_anomaly_returns_content_when_present(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("정상적인 AI 설명입니다.")
    mock_get_client.return_value = mock_client

    result = explain_anomaly({"mmsi": "123", "gap_minutes": 70})

    assert result == "정상적인 AI 설명입니다."


@patch("src.ai_layer.explainer._get_client")
def test_explain_anomaly_uses_max_tokens_500(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("설명")
    mock_get_client.return_value = mock_client

    explain_anomaly({"mmsi": "123", "gap_minutes": 70})

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 500


@patch("src.ai_layer.explainer._get_client")
def test_explain_anomaly_falls_back_when_api_call_raises(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("network blocked")
    mock_get_client.return_value = mock_client

    result = explain_anomaly({"mmsi": "123", "gap_minutes": 70})

    assert "규칙 기반 폴백" in result
    assert "AI 호출 실패" in result
