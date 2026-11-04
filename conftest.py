"""Test-only auth setting. Production must supply its own secret."""
import pytest

@pytest.fixture(autouse=True)
def demo_auth(monkeypatch):
    monkeypatch.setenv("COPILOT_API_KEY", "test-only-copilot-secret")
