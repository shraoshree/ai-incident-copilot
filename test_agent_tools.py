"""Offline tests for bounded, read-only investigation tools."""
import pytest
from pydantic import ValidationError
from agent_tools import search_incident_knowledge, investigate_with_tools


KNOWLEDGE = (
    "Incident: API latency\nSymptoms: database connections exhausted\n"
    "\n\nIncident: Queue backlog\nSymptoms: worker queue growing"
)


def test_retrieval_returns_ranked_evidence():
    result = search_incident_knowledge("database connections exhausted", KNOWLEDGE)
    assert result["count"] == 1
    assert "API latency" in result["matches"][0]["excerpt"]


def test_no_match_does_not_fabricate_evidence():
    assert search_incident_knowledge("unrelated service issue", KNOWLEDGE)["count"] == 0


def test_query_length_guard():
    with pytest.raises(ValidationError):
        search_incident_knowledge("x", KNOWLEDGE)


class FakeResponse:
    function_calls = []
    text = '{"likely_cause":"unknown","investigation_steps":[],"immediate_action":"inspect"}'


class FakeModels:
    def generate_content(self, **kwargs):
        return FakeResponse()


class FakeClient:
    models = FakeModels()


def test_no_tool_call_returns_text_and_empty_trace():
    result = investigate_with_tools(FakeClient(), "fake-model", {"alert": "latency"}, KNOWLEDGE)
    assert result["tool_trace"] == []
    assert "likely_cause" in result["text"]
