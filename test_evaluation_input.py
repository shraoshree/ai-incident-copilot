"""Input parity checks independent of database and Gemini availability."""
import pytest
from pydantic import ValidationError

from evaluation import case_query
from evaluation_robust import score_rows


def test_structured_cases_use_formatted_redacted_input_for_bm25(monkeypatch):
    import evaluation_robust
    queries = []
    monkeypatch.setattr(evaluation_robust.BM25, "search",
                        lambda self, query, top_k: queries.append(query) or [])
    case = {"incident": {"service": "payments", "alert": "slow requests",
                         "severity": "critical", "description": "token=secret-value"},
            "split": "test", "expected_title": None}
    score_rows([case], "bm25", [])
    assert queries == ["Service: payments\nAlert: slow requests\nDescription: token=[REDACTED]"]


@pytest.mark.parametrize("field,value", [("service", ""), ("alert", "a" * 301),
                                         ("description", "d" * 10001), ("severity", "")])
def test_evaluation_rejects_invalid_api_payload(field, value):
    payload = {"service": "payments", "alert": "slow", "severity": "critical", "description": "slow"}
    payload[field] = value
    with pytest.raises(ValidationError):
        case_query({"incident": payload})
