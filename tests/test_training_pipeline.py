"""Training pipeline smoke tests."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "scripts").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from training.automated_model_trainer import AutomatedModelTrainer, TrainingConfig


def _small_market_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=300, freq="1min")
    base = 2000.0
    returns = np.random.default_rng(42).normal(0, 0.0005, size=300)
    close = base * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0003,
            "low": close * 0.9997,
            "close": close,
            "volume": 500,
        },
        index=idx,
    )


def test_training_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "trained").mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(min_samples=10, hyperopt_trials=0, cv_folds=2, validation_split=0.2)
    trainer = AutomatedModelTrainer(cfg)
    trainer._load_market_data = lambda lookback_days=30: _small_market_data()

    from training.automated_model_trainer import ModelTrainer

    ModelTrainer.optimize_hyperparameters = lambda self, X, y, config: self.default_params

    X, y = trainer._prepare_training_data()
    assert not X.empty
    assert len(X) == len(y)

    mt = ModelTrainer("ridge")
    res = mt.train(X.tail(200), y.loc[X.tail(200).index], cfg)
    assert res.success is True

    trained_dir = tmp_path / "models" / "trained"
    assert (trained_dir / "feature_list.json").exists()
    feature_list = json.loads((trained_dir / "feature_list.json").read_text())
    assert len(feature_list) == X.shape[1]
    manifests = list(trained_dir.glob("*.manifest.json"))
    assert manifests
    manifest = json.loads(manifests[0].read_text())
    assert "sha256" in manifest and "version" in manifest
