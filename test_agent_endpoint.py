"""HTTP-level tests for the agent endpoint, without external model calls."""
from unittest.mock import patch
from fastapi.testclient import TestClient
import main


def test_agent_endpoint_persists_validated_analysis(tmp_path, monkeypatch):
    store = tmp_path / "incidents.json"
    store.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(main, "INCIDENTS_FILE", str(store))
    payload = {
        "service": "billing",
        "alert": "High API latency",
        "severity": "critical",
        "description": "Database connection pool exhausted",
    }
    output = {
        "text": '{"likely_cause":"Pool saturation","investigation_steps":["Inspect pool metrics"],"immediate_action":"Reduce request pressure"}',
        "tool_trace": [{"tool": "search_incident_knowledge", "result_count": 1}],
    }
    with patch.object(main, "get_gemini_client", return_value=object()), \
         patch.object(main, "investigate_with_tools", return_value=output):
        response = TestClient(main.app).post("/incident/agent", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["incident"]["analysis_source"] == "gemini_agent"
    assert body["incident"]["ai_analysis"]["likely_cause"] == "Pool saturation"
    assert body["tool_trace"][0]["result_count"] == 1
    assert len(main.load_incidents()) == 1


def test_agent_endpoint_falls_back_on_agent_error(tmp_path, monkeypatch):
    store = tmp_path / "incidents.json"
    store.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(main, "INCIDENTS_FILE", str(store))
    payload = {
        "service": "billing",
        "alert": "Latency",
        "severity": "high",
        "description": "Timeouts",
    }
    with patch.object(main, "get_gemini_client", return_value=object()), \
         patch.object(main, "investigate_with_tools", side_effect=RuntimeError("agent unavailable")):
        response = TestClient(main.app).post("/incident/agent", json=payload)
    assert response.status_code == 200
    assert response.json()["incident"]["analysis_source"] == "fallback"
    assert response.json()["incident"]["ai_analysis"]["likely_cause"].startswith("Root cause not confirmed")
