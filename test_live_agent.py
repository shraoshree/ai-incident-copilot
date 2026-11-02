"""Opt-in live test: run only with a private GEMINI_API_KEY."""
import os

import pytest
from google import genai

from agent_tools import investigate_with_tools


@pytest.mark.live
def test_real_model_searches_historical_knowledge():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not configured")
    knowledge = (
        "Incident: payments-api connection exhaustion\n"
        "Symptoms: HTTP 503 spikes after database pool saturation.\n"
        "Mitigation: inspect active connections and pool limits."
    )
    incident = {
        "service": "payments-api",
        "alert": "HTTP 503 errors",
        "severity": "high",
        "description": "Connection pool exhaustion after traffic spike",
    }
    result = investigate_with_tools(
        genai.Client(api_key=key),
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        incident,
        knowledge,
    )
    assert any(
        event["tool"] == "search_incident_knowledge" and event["result_count"] > 0
        for event in result["tool_trace"]
    ), "Model did not retrieve relevant historical evidence"
    assert result["text"].strip(), "Model returned no investigation"
    from main import IncidentAnalysis
    import json
    raw = result["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    IncidentAnalysis.model_validate(json.loads(raw))
