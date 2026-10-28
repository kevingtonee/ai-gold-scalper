"""EA/server risk-management contract tests."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.enhanced_ai_server_consolidated import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture()
def api_key(monkeypatch):
    monkeypatch.setenv("AI_SERVER_API_KEY", "test-secret-key")
    return "test-secret-key"


@pytest.fixture()
def auth_headers(api_key):
    return {"X-API-Key": api_key}


def _valid_payload():
    return {
        "entry_price": 2000.0,
        "stop_loss_price": 1990.0,
        "side": "BUY",
        "symbol": "XAUUSD",
        "confidence": 80,
        "account_balance": 10000.0,
    }


def test_old_ea_payload_fails(client, auth_headers):
    response = client.post(
        "/risk-management",
        json={"confidence": 0.75, "account_balance": 10000.0, "risk_percent": 0.02},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_corrected_payload_succeeds(client, auth_headers):
    response = client.post("/risk-management", json=_valid_payload(), headers=auth_headers)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["valid"] is True
    assert payload["signal"] == "BUY"
    assert payload["lot_size"] == 0.1


def test_missing_api_key(client):
    response = client.post("/risk-management", json=_valid_payload())
    assert response.status_code == 401


def test_invalid_api_key(client, auth_headers):
    bad = dict(auth_headers)
    bad["X-API-Key"] = "wrong"
    response = client.post("/risk-management", json=_valid_payload(), headers=bad)
    assert response.status_code == 401


def test_risk_rejection_invalid_stop(client, auth_headers):
    payload = _valid_payload()
    payload["stop_loss_price"] = 2010.0
    response = client.post("/risk-management", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert response.get_json()["valid"] is False


def test_non_finite_values_rejected(client, auth_headers):
    payload = _valid_payload()
    payload["entry_price"] = float("nan")
    response = client.post("/risk-management", json=payload, headers=auth_headers)
    assert response.status_code == 400
