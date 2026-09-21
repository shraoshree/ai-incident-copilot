import main
from fastapi.testclient import TestClient


client = TestClient(main.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "AI Incident Copilot"
    }


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI Incident Copilot is running"


def test_knowledge_retrieval():
    incident = main.Incident(
        service="payment-service",
        alert="High API latency",
        severity="critical",
        description="p99 latency increased to 2800ms"
    )

    result = main.find_relevant_knowledge(incident)

    assert "Payment Service High Latency" in result
    assert "Database connection pool exhaustion" in result


def test_no_matching_knowledge():
    incident = main.Incident(
        service="unknown-service",
        alert="Completely unrelated problem",
        severity="low",
        description="xyzabc unusual event"
    )

    result = main.find_relevant_knowledge(incident)

    assert result == "No relevant past incident found."