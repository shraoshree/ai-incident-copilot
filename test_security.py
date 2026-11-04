"""Security regressions: auth, rate limits, and redaction."""
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from security import FixedWindowLimiter, redact_sensitive

KEY = {"X-API-Key": "test-only-copilot-secret"}


def test_auth_required_on_incident_routes():
    client = TestClient(main.app)
    assert client.get("/incidents").status_code == 401
    assert client.get("/incidents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health").status_code != 401


def test_auth_fail_closed_without_config(monkeypatch):
    monkeypatch.delenv("COPILOT_API_KEY", raising=False)
    client = TestClient(main.app)
    assert client.get("/incidents", headers=KEY).status_code == 503


def test_redaction_of_common_secrets_and_identifiers():
    raw = "password=hunter2 api_key=ABC123 Authorization: Bearer abc.DEF token=xyz user@example.com 192.168.0.10"
    clean = redact_sensitive(raw)
    for sensitive in ("hunter2", "ABC123", "abc.DEF", "user@example.com", "192.168.0.10"):
        assert sensitive not in clean
    assert "[REDACTED]" in clean
    assert "json-secret-123" not in redact_sensitive('{"password": "json-secret-123"}')


def test_input_redacted_before_retrieval_and_persistence():
    client = TestClient(main.app, headers=KEY)
    with patch.object(main, "retrieve", return_value=([], 0.5)) as retrieve, \
         patch.object(main, "analyze_incident", return_value=(main.fallback("demo"), "fallback")), \
         patch.object(main, "save_incident", return_value={"incident_id": "INC-test"}) as save:
        response = client.post("/incident", json={
            "service": "payments", "alert": "password=secret123", "severity": "high",
            "description": "contact admin@example.com and Bearer abc.def"
        })
    assert response.status_code == 200, response.text
    incident = retrieve.call_args.args
    stored = save.call_args.args[0]
    assert "secret123" not in str(incident)
    assert "secret123" not in stored.alert
    assert "admin@example.com" not in stored.description
    assert "abc.def" not in stored.description


def test_request_size_limit():
    client = TestClient(main.app, headers=KEY)
    assert client.post("/incident", content="x" * 32769, headers={"content-type": "application/json"}).status_code == 413


def test_limiter_enforces_quota_and_resets():
    limiter = FixedWindowLimiter()
    assert limiter.allow(("local", "incident"), limit=2)[0]
    assert limiter.allow(("local", "incident"), limit=2)[0]
    allowed, retry = limiter.allow(("local", "incident"), limit=2)
    assert not allowed and retry > 0
    assert limiter.allow(("different", "incident"), limit=2)[0]
