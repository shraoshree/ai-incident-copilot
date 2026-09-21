from datetime import datetime, timezone

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


# --------------------------------------------------
# Basic API Tests
# --------------------------------------------------

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "AI Incident Copilot"
    }


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Incident Copilot is running"
    }


# --------------------------------------------------
# Semantic Retrieval Tests
# --------------------------------------------------

def test_semantic_retrieval(monkeypatch):
    expected_knowledge = (
        "Incident: Payment Service High Latency\n"
        "Symptoms: p99 API latency above 2000ms.\n"
        "Possible cause: Database connection pool exhaustion."
    )

    def fake_create_embedding(text):
        return [0.1, 0.2, 0.3]

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def execute(self, query, params):
            pass

        def fetchone(self):
            return (expected_knowledge, 0.18)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(
        main,
        "create_embedding",
        fake_create_embedding
    )

    monkeypatch.setattr(
        main.psycopg,
        "connect",
        lambda database_url: FakeConnection()
    )

    monkeypatch.setattr(
        main,
        "register_vector",
        lambda connection: None
    )

    incident = main.Incident(
        service="checkout-backend",
        alert="Requests slowing down",
        severity="critical",
        description=(
            "DB connections appear saturated and requests "
            "are waiting for available connections"
        )
    )

    result = main.find_relevant_knowledge(incident)

    assert result == expected_knowledge


def test_retrieval_dependency_failure(monkeypatch):
    def fake_create_embedding(text):
        raise RuntimeError("Embedding provider unavailable")

    monkeypatch.setattr(
        main,
        "create_embedding",
        fake_create_embedding
    )

    incident = main.Incident(
        service="checkout-backend",
        alert="Requests slowing down",
        severity="critical",
        description="Database connections are saturated"
    )

    result = main.find_relevant_knowledge(incident)

    assert result == "No relevant past incident found."


# --------------------------------------------------
# Gemini Fallback + Incident Ingestion
# --------------------------------------------------

def test_incident_saved_when_ai_falls_back(monkeypatch):
    historical_knowledge = (
        "Incident: Payment Service High Latency\n"
        "Possible cause: Database connection pool exhaustion."
    )

    fallback_analysis = main.IncidentAnalysis(
        likely_cause="AI provider unavailable",
        investigation_steps=[
            "Check application logs",
            "Review monitoring metrics"
        ],
        immediate_action="Continue manual investigation"
    )

    monkeypatch.setattr(
        main,
        "find_relevant_knowledge",
        lambda incident: historical_knowledge
    )

    monkeypatch.setattr(
        main,
        "analyze_incident",
        lambda incident, knowledge: (
            fallback_analysis,
            "fallback"
        )
    )

    def fake_save_incident(
        incident,
        relevant_knowledge,
        ai_analysis,
        analysis_source
    ):
        return {
            "incident_id": "INC-test1234",
            "timestamp": (
                datetime.now(timezone.utc).isoformat()
            ),
            "service": incident.service,
            "alert": incident.alert,
            "severity": incident.severity,
            "description": incident.description,
            "status": "open",
            "relevant_knowledge": relevant_knowledge,
            "analysis_source": analysis_source,
            "ai_analysis": ai_analysis.model_dump()
        }

    monkeypatch.setattr(
        main,
        "save_incident",
        fake_save_incident
    )

    response = client.post(
        "/incident",
        json={
            "service": "checkout-backend",
            "alert": "Requests slowing down",
            "severity": "critical",
            "description": "Database connections are saturated"
        }
    )

    assert response.status_code == 200

    body = response.json()

    assert body["message"] == "Incident analyzed and saved"
    assert body["incident"]["analysis_source"] == "fallback"
    assert body["incident"]["status"] == "open"
    assert (
        body["incident"]["relevant_knowledge"]
        == historical_knowledge
    )


# --------------------------------------------------
# Status Transition Tests
# --------------------------------------------------

def test_valid_status_transition(monkeypatch):
    incident_id = "INC-test1234"

    class FakeCursor:
        def __init__(self):
            self.last_query = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def execute(self, query, params):
            self.last_query = query

        def fetchone(self):
            if "SELECT status" in self.last_query:
                return ("open",)

            return (
                incident_id,
                datetime.now(timezone.utc),
                "checkout-backend",
                "Requests slowing down",
                "critical",
                "Database connections are saturated",
                "investigating",
                "Historical incident",
                "fallback",
                {
                    "likely_cause": "AI provider unavailable",
                    "investigation_steps": [
                        "Check application logs"
                    ],
                    "immediate_action": (
                        "Continue manual investigation"
                    )
                }
            )

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

    monkeypatch.setattr(
        main.psycopg,
        "connect",
        lambda database_url: FakeConnection()
    )

    response = client.patch(
        f"/incidents/{incident_id}/status",
        json={
            "status": "investigating"
        }
    )

    assert response.status_code == 200
    assert (
        response.json()["incident"]["status"]
        == "investigating"
    )


def test_rejected_backward_transition(monkeypatch):
    incident_id = "INC-test1234"

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def execute(self, query, params):
            pass

        def fetchone(self):
            return ("resolved",)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

    monkeypatch.setattr(
        main.psycopg,
        "connect",
        lambda database_url: FakeConnection()
    )

    response = client.patch(
        f"/incidents/{incident_id}/status",
        json={
            "status": "open"
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Invalid status transition: "
            "resolved -> open"
        )
    }