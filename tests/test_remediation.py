"""Remediation regression tests for the AI Gold Scalper server."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import pytest

from core.enhanced_ai_server_consolidated import (
    ModelInferenceEngine,
    app,
    cache,
    load_config,
)


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


def _valid_risk_payload():
    return {
        "entry_price": 2000.0,
        "stop_loss_price": 1990.0,
        "side": "BUY",
        "symbol": "XAUUSD",
        "confidence": 80,
    }


def test_risk_management_valid_request(client, auth_headers):
    response = client.post("/risk-management", json=_valid_risk_payload(), headers=auth_headers)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["valid"] is True
    assert payload["signal"] == "BUY"
    assert payload["lot_size"] == 0.1
    assert payload["sl"] == 1990.0
    assert "tp" in payload
    assert "risk" in payload


def test_risk_management_missing_fields(client, auth_headers):
    response = client.post("/risk-management", json={"symbol": "XAUUSD"}, headers=auth_headers)
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_risk_management_invalid_side(client, auth_headers):
    payload = _valid_risk_payload()
    payload["side"] = "HOLD"
    response = client.post("/risk-management", json=payload, headers=auth_headers)
    assert response.status_code == 400


def test_risk_management_malformed_numbers(client, auth_headers):
    payload = _valid_risk_payload()
    payload["entry_price"] = "not-a-number"
    response = client.post("/risk-management", json=payload, headers=auth_headers)
    assert response.status_code == 400

    payload = _valid_risk_payload()
    payload["entry_price"] = float("nan")
    response = client.post("/risk-management", json=payload, headers=auth_headers)
    assert response.status_code == 400


def test_risk_management_invalid_stop_direction(client, auth_headers):
    payload = _valid_risk_payload()
    payload["stop_loss_price"] = 2010.0
    response = client.post("/risk-management", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert response.get_json()["valid"] is False


def test_risk_management_risk_limit_rejected(client, auth_headers):
    import core.enhanced_ai_server_consolidated as server

    original_open_positions = server.risk_manager.open_positions
    server.risk_manager.open_positions = 5
    try:
        response = client.post("/risk-management", json=_valid_risk_payload(), headers=auth_headers)
    finally:
        server.risk_manager.open_positions = original_open_positions
    assert response.status_code == 400
    assert response.get_json()["valid"] is False

def test_risk_management_internal_exception(client, auth_headers, monkeypatch):
    import core.enhanced_ai_server_consolidated as server

    monkeypatch.setattr(server, "RiskManager", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internal detail")))
    response = client.post("/risk-management", json=_valid_risk_payload(), headers=auth_headers)
    assert response.status_code == 500
    payload = response.get_json()
    assert "secret internal detail" not in str(payload)
    assert "details" not in payload


def test_unauthorized_request(client):
    os.environ["AI_SERVER_API_KEY"] = "secret"
    try:
        response = client.post("/risk-management", json=_valid_risk_payload())
    finally:
        del os.environ["AI_SERVER_API_KEY"]
    assert response.status_code == 401
    assert response.get_json()["error"] == "Unauthorized"


def test_authorized_request(client, auth_headers):
    response = client.post("/risk-management", json=_valid_risk_payload(), headers=auth_headers)
    assert response.status_code == 200


def test_missing_api_key_configuration(client, monkeypatch):
    monkeypatch.delenv("AI_SERVER_API_KEY", raising=False)
    response = client.post("/risk-management", json=_valid_risk_payload())
    assert response.status_code == 401


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "healthy"
    assert "models_loaded" in payload


def test_models_status_endpoint(client, auth_headers):
    response = client.get("/models/status", headers=auth_headers)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert "loaded_models" in payload["models"]


def test_ai_signal_invalid_payload(client, auth_headers):
    response = client.post("/ai_signal", json="not-an-object", headers=auth_headers)
    assert response.status_code == 400


def test_cache_behavior():
    cache.clear()
    assert cache.get("remediation-test") is None
    cache.set("remediation-test", {"value": 42}, timeout=60)
    assert cache.get("remediation-test") == {"value": 42}


def test_dashboard_import_and_construction():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dashboard.trading_dashboard import TradingDashboard

    dashboard = TradingDashboard(debug=True)
    assert dashboard.app is not None
    assert dashboard.socketio is not None
def test_config_loading_nested_and_flat(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({
        "risk": {
            "default_account_balance": 12345.0,
            "risk_percent_per_trade": 0.02,
        },
        "symbols": {
            "XAUUSD": {"contract_size": 100.0, "tick_size": 0.01, "tick_value": 1.0,
                       "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
                       "stop_level_min_points": 10.0}
        },
        "ai": {"api_key": "YOUR_OPENAI_API_KEY_HERE"},
    }), encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(nested))
    assert load_config() is True
    from core.enhanced_ai_server_consolidated import CONFIG

    assert CONFIG["risk"]["account_balance"] == 12345.0
    assert CONFIG["use_gpt4"] is False

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({
        "account_balance": 5000.0,
        "openai_api_key": "sk-test-flat-key",
    }), encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(flat))
    assert load_config() is True
    assert CONFIG["risk"]["account_balance"] == 5000.0
    assert CONFIG["use_gpt4"] is True


def test_config_loading_missing_and_malformed(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "does-not-exist.json"))
    assert load_config() is False

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(bad))
    assert load_config() is False


def test_model_path_and_hash_security(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    monkeypatch.setenv("AI_MODEL_DIR", str(trusted))
    engine = ModelInferenceEngine()

    model_file = trusted / "model.pkl"
    joblib.dump({"value": 1}, model_file)
    assert engine.load_model("m", str(model_file)) is False

    manifest = trusted / "model.manifest.json"
    manifest.write_text(json.dumps({
        "version": "1.0",
        "sha256": "0" * 64,
    }), encoding="utf-8")
    assert engine.load_model("m", str(model_file)) is False

    outside = tmp_path / "outside.pkl"
    joblib.dump({"value": 2}, outside)
    (tmp_path / "outside.manifest.json").write_text(json.dumps({
        "version": "1.0",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    assert engine.load_model("outside", str(outside)) is False

    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({
        "version": "1.0",
        "sha256": digest,
        "model_name": "m",
    }), encoding="utf-8")
    assert engine.load_model("m", str(model_file)) is True
    assert "m" in engine.models


def test_model_rejects_unsupported_extension(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    monkeypatch.setenv("AI_MODEL_DIR", str(trusted))
    engine = ModelInferenceEngine()
    bad = trusted / "model.txt"
    bad.write_text("not a model", encoding="utf-8")
    (trusted / "model.manifest.json").write_text(json.dumps({
        "version": "1.0",
        "sha256": hashlib.sha256(bad.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    assert engine.load_model("bad", str(bad)) is False

