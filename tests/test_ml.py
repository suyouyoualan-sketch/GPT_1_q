import math
from pathlib import Path

import app.ml as ml
from app.strategy import backtest


def synthetic_candles(count: int = 800) -> list[list[float]]:
    start = 1_700_000_000_000
    rows = []
    for index in range(count):
        close = 100 + 0.02 * index + 2 * math.sin(index / 20)
        open_price = close - 0.15 * math.sin(index / 5)
        rows.append(
            [
                start + index * 3_600_000,
                open_price,
                max(open_price, close) + 0.4,
                min(open_price, close) - 0.4,
                close,
                1_000 + 100 * math.sin(index / 7),
            ]
        )
    return rows


def test_lightgbm_training_produces_reusable_strategy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ml, "MODEL_DIR", Path(tmp_path).resolve())
    ml._load_booster.cache_clear()
    rows = synthetic_candles()

    trained = ml.train_lightgbm(
        rows,
        experiment_id=7,
        artifact_token="abcdef123456",
        horizon=12,
        fee_bps=10,
        slippage_bps=5,
        entry_threshold_pct=0.1,
        exit_threshold_pct=0,
    )

    assert trained["sample_count"] > 600
    assert trained["oos_sample_count"] > 250
    assert len(trained["metrics"]["folds"]) == 3
    assert all(fold["purged_samples"] == 12 for fold in trained["metrics"]["folds"])
    assert (tmp_path / trained["artifact"]).is_file()

    rule = {
        "ml_model": {
            "model_type": "lightgbm",
            "artifact": trained["artifact"],
            "feature_version": ml.FEATURE_VERSION,
            "horizon": 12,
            "entry_threshold_pct": 0.1,
            "exit_threshold_pct": 0,
        }
    }
    result = backtest(rows, rule, initial_cash=1_000, order_quote=500)
    assert result["trades"]
    assert result["trades"][0]["decision"] == "模型买入开仓"
    assert all(trade["reason"] == "ml_prediction" for trade in result["trades"])

    ml._load_booster.cache_clear()
