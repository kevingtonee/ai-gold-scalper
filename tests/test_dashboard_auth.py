"""Dashboard control authentication tests."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.trading_dashboard import TradingDashboard


@pytest.fixture()
def dashboard(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "test-dashboard-secret")
    monkeypatch.setenv("AI_SERVER_API_KEY", "test-api-key")
    return TradingDashboard(debug=True)


def test_control_requires_api_key(dashboard):
    client = dashboard.app.test_client()
    response = client.post("/api/control/emergency_stop", json={})
    assert response.status_code == 401


def test_control_with_api_key(dashboard):
    client = dashboard.app.test_client()
    response = client.post(
        "/api/control/emergency_stop",
        json={},
        headers={"X-API-Key": "test-api-key"},
    )
    assert response.status_code == 200
    assert response.get_json()["success"] is True
