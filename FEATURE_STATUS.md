# Feature Status — AI Gold Scalper

## Implemented and tested
- Flask AI server with `/health`, `/ai_signal`, `/models/status`, `/performance`, `/risk-management`.
- API key authentication via `X-API-Key` / Bearer token; fail-closed when unset.
- Trusted model loading with approved directory, manifest, SHA-256 hash, and optional HMAC signature.
- Broker-aware risk sizing (`RiskManager.calculate_position_size`) with tests.
- Persistent risk state store (SQLite) with day rollover.
- Trading dashboard construction with required secret key outside debug.
- Dashboard control endpoint authentication.

## Implemented but not integration-tested
- EA HTTP contract for `/risk-management` and `/ai_signal` (MQL5 implementation; compilation not verified).
- EA demo-account checks, auto-trading gate, spread/position/margin/volume validation (code present; not compiled).
- Model inference endpoint behavior with real trained artifacts (none present).

## Placeholders or unimplemented
- News ingestion, sentiment analysis, and news-to-signal integration.
- Learned pattern recognition beyond rule-based technical indicators.
- Model training pipeline, validation, and walk-forward testing.
- Realistic backtesting with transaction costs and out-of-sample evaluation.
- MT5 Python integration (EA is the bridge).
- Live/demo order execution verification (mocked only in tests; no broker calls).

## Unavailable because trained model artifacts are missing
- ML inference (`rf_classifier`, `rf_regressor`) is unavailable; `feature_list.pkl` is deliberately not loaded.
- GPT-4 signals are unavailable unless `OPENAI_API_KEY` is configured.

## Safety defaults
- `ENABLE_AUTO_TRADING=false`
- Real-account trading rejected in EA when `Require_Demo_Account=true`
- API and dashboard secrets loaded from environment only
