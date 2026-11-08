from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re
from typing import Any

import lightgbm as lgb
import numpy as np

from .exchange import compact_ohlcv


MODEL_DIR = (Path(__file__).resolve().parent.parent / "data/models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_VERSION = "ohlcv_v1"
LOOKBACK = 50
FEATURE_NAMES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "sma_distance_10",
    "sma_distance_20",
    "sma_distance_50",
    "momentum_6",
    "momentum_12",
    "momentum_24",
    "volatility_12",
    "volatility_24",
    "rsi_14",
    "zscore_20",
    "volume_ratio_20",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "candle_range",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]
ARTIFACT_PATTERN = re.compile(r"^ml_[0-9]+_[a-f0-9]{12}\.txt$")


class MLTrainingError(ValueError):
    pass


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _feature_row(candles: list[list[float]], index: int) -> list[float]:
    if index < LOOKBACK - 1:
        raise MLTrainingError(f"计算特征至少需要 {LOOKBACK} 根 K 线")
    closes = [float(row[4]) for row in candles]
    volumes = [float(row[5]) for row in candles]
    close = closes[index]
    row = candles[index]
    values: list[float] = []

    for period in (1, 3, 6, 12, 24):
        values.append((_safe_ratio(close, closes[index - period]) - 1) * 100)
    for period in (10, 20, 50):
        average = sum(closes[index - period + 1 : index + 1]) / period
        values.append((_safe_ratio(close, average) - 1) * 100)
    for period in (6, 12, 24):
        values.append((_safe_ratio(close, closes[index - period]) - 1) * 100)
    for period in (12, 24):
        returns = [
            (_safe_ratio(closes[i], closes[i - 1]) - 1) * 100
            for i in range(index - period + 1, index + 1)
        ]
        values.append(float(np.std(returns)))

    changes = [
        closes[i] - closes[i - 1] for i in range(index - 13, index + 1)
    ]
    average_gain = sum(max(change, 0.0) for change in changes) / 14
    average_loss = sum(max(-change, 0.0) for change in changes) / 14
    values.append(
        100.0
        if average_loss == 0
        else 100 - 100 / (1 + average_gain / average_loss)
    )

    close_window = closes[index - 19 : index + 1]
    mean = sum(close_window) / 20
    deviation = float(np.std(close_window))
    values.append(_safe_ratio(close - mean, deviation))
    average_volume = sum(volumes[index - 19 : index + 1]) / 20
    values.append(_safe_ratio(volumes[index], average_volume))

    open_price, high, low = float(row[1]), float(row[2]), float(row[3])
    scale = open_price or close or 1.0
    values.extend(
        [
            (close - open_price) / scale * 100,
            (high - max(open_price, close)) / scale * 100,
            (min(open_price, close) - low) / scale * 100,
            (high - low) / scale * 100,
        ]
    )

    from datetime import datetime, timezone

    moment = datetime.fromtimestamp(float(row[0]) / 1_000, tz=timezone.utc)
    values.extend(
        [
            math.sin(2 * math.pi * moment.hour / 24),
            math.cos(2 * math.pi * moment.hour / 24),
            math.sin(2 * math.pi * moment.weekday() / 7),
            math.cos(2 * math.pi * moment.weekday() / 7),
        ]
    )
    return values


def build_dataset(
    candles: list[list[float]], horizon: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if horizon < 1:
        raise MLTrainingError("预测周期必须至少为 1")
    indices = list(range(LOOKBACK - 1, len(candles) - horizon))
    if len(indices) < 600:
        raise MLTrainingError(
            f"有效训练样本只有 {len(indices)} 个，至少需要 600 个；请扩大训练区间"
        )
    x = np.asarray([_feature_row(candles, index) for index in indices], dtype=float)
    closes = [float(row[4]) for row in candles]
    y = np.asarray(
        [(closes[index + horizon] / closes[index] - 1) * 100 for index in indices],
        dtype=float,
    )
    return x, y, indices


def _train_booster(x: np.ndarray, y: np.ndarray) -> lgb.Booster:
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbosity": -1,
        "num_threads": 2,
    }
    return lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=180)


def _prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if len(actual) > 1 and np.std(actual) and np.std(predicted)
        else 0.0
    )
    return {
        "mae_pct": round(float(np.mean(np.abs(errors))), 4),
        "rmse_pct": round(float(np.sqrt(np.mean(errors**2))), 4),
        "direction_accuracy_pct": round(
            float(np.mean(np.sign(actual) == np.sign(predicted)) * 100), 2
        ),
        "correlation": round(correlation, 4),
    }


def _simulate_oos(
    candles: list[list[float]],
    candle_indices: list[int],
    predictions: np.ndarray,
    entry_threshold_pct: float,
    exit_threshold_pct: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    initial_cash = 10_000.0
    cash = initial_cash
    position = 0.0
    entry_price: float | None = None
    peak = initial_cash
    max_drawdown = 0.0
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, float | int]] = []
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    first_execution_index = min(candle_indices[-1], candle_indices[0] + 1)
    first_price = float(candles[first_execution_index][1])

    for sample_index, prediction in zip(candle_indices, predictions, strict=True):
        execution_index = min(sample_index + 1, len(candles) - 1)
        candle = candles[execution_index]
        timestamp = int(candle[0])
        market_price = float(candle[1])
        if position == 0 and prediction >= entry_threshold_pct:
            price = market_price * (1 + slippage_rate)
            fee = cash * fee_rate
            position = (cash - fee) / price
            cash = 0.0
            entry_price = price
            trades.append(
                {
                    "timestamp": timestamp,
                    "side": "buy",
                    "price": round(price, 8),
                    "amount": round(position, 10),
                    "prediction_pct": round(float(prediction), 4),
                    "reason": "ml_prediction",
                    "action": "open",
                    "decision": "模型买入开仓",
                    "candle": {
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    },
                }
            )
        elif position > 0 and prediction <= exit_threshold_pct:
            price = market_price * (1 - slippage_rate)
            amount = position
            cash = amount * price * (1 - fee_rate)
            position = 0.0
            entry_price = None
            trades.append(
                {
                    "timestamp": timestamp,
                    "side": "sell",
                    "price": round(price, 8),
                    "amount": round(amount, 10),
                    "prediction_pct": round(float(prediction), 4),
                    "reason": "ml_prediction",
                    "action": "close",
                    "decision": "模型卖出平仓",
                    "candle": {
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    },
                }
            )
        equity = cash + position * market_price
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
        equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": round(equity, 4),
                "benchmark_equity": round(
                    initial_cash * market_price / first_price, 4
                ),
            }
        )

    last_index = min(candle_indices[-1] + 1, len(candles) - 1)
    last_price = float(candles[last_index][4])
    final_equity = cash + position * last_price
    return_pct = (final_equity / initial_cash - 1) * 100
    benchmark_return_pct = (last_price / first_price - 1) * 100
    closed = [trade for trade in trades if trade["side"] == "sell"]
    wins = 0
    buy_price: float | None = None
    for trade in trades:
        if trade["side"] == "buy":
            buy_price = float(trade["price"])
        elif buy_price is not None:
            wins += int(float(trade["price"]) > buy_price)
            buy_price = None

    step = max(1, len(equity_curve) // 160)
    sampled_curve = equity_curve[::step]
    if sampled_curve[-1] != equity_curve[-1]:
        sampled_curve.append(equity_curve[-1])
    price_rows = candles[first_execution_index : last_index + 1]
    return {
        "initial_cash": initial_cash,
        "final_equity": round(final_equity, 4),
        "return_pct": round(return_pct, 4),
        "benchmark_return_pct": round(benchmark_return_pct, 4),
        "excess_return_pct": round(return_pct - benchmark_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "closed_trades": len(closed),
        "win_rate_pct": round(wins / len(closed) * 100, 2) if closed else 0.0,
        "open_position": round(position, 10),
        "trades": trades,
        "equity_curve": sampled_curve,
        "price_candles": compact_ohlcv(price_rows),
        "entry_price": round(entry_price, 8) if entry_price else None,
    }


def train_lightgbm(
    candles: list[list[float]],
    *,
    experiment_id: int,
    artifact_token: str,
    horizon: int,
    fee_bps: float,
    slippage_bps: float,
    entry_threshold_pct: float,
    exit_threshold_pct: float,
) -> dict[str, Any]:
    x, gross_targets, candle_indices = build_dataset(candles, horizon)
    round_trip_cost_pct = 2 * (fee_bps + slippage_bps) / 100
    targets = gross_targets - round_trip_cost_pct
    sample_count = len(targets)
    boundaries = [int(sample_count * ratio) for ratio in (0.55, 0.70, 0.85, 1.0)]
    oos_predictions: list[np.ndarray] = []
    oos_actual: list[np.ndarray] = []
    oos_indices: list[int] = []
    folds: list[dict[str, Any]] = []

    for fold_number in range(3):
        train_end = boundaries[fold_number]
        test_end = boundaries[fold_number + 1]
        train_stop = train_end - horizon
        if train_stop < 300:
            raise MLTrainingError("扣除预测隔离区后，训练样本不足 300 个")
        if test_end - train_end < 50:
            raise MLTrainingError("每个滚动验证区间至少需要 50 个样本")
        booster = _train_booster(x[:train_stop], targets[:train_stop])
        predicted = booster.predict(x[train_end:test_end])
        actual = targets[train_end:test_end]
        oos_predictions.append(np.asarray(predicted, dtype=float))
        oos_actual.append(actual)
        oos_indices.extend(candle_indices[train_end:test_end])
        folds.append(
            {
                "fold": fold_number + 1,
                "train_samples": train_stop,
                "test_samples": test_end - train_end,
                "purged_samples": horizon,
                **_prediction_metrics(actual, np.asarray(predicted, dtype=float)),
            }
        )

    all_oos_predictions = np.concatenate(oos_predictions)
    all_oos_actual = np.concatenate(oos_actual)
    prediction_metrics = _prediction_metrics(all_oos_actual, all_oos_predictions)
    simulation = _simulate_oos(
        candles,
        oos_indices,
        all_oos_predictions,
        entry_threshold_pct,
        exit_threshold_pct,
        fee_bps,
        slippage_bps,
    )

    final_booster = _train_booster(x, targets)
    artifact = f"ml_{experiment_id}_{artifact_token}.txt"
    if not ARTIFACT_PATTERN.fullmatch(artifact):
        raise MLTrainingError("模型文件名无效")
    final_booster.save_model(str(MODEL_DIR / artifact))
    importance = final_booster.feature_importance(importance_type="gain")
    total_importance = float(np.sum(importance)) or 1.0
    feature_importance = sorted(
        [
            {
                "feature": name,
                "importance_pct": round(float(value) / total_importance * 100, 2),
            }
            for name, value in zip(FEATURE_NAMES, importance, strict=True)
        ],
        key=lambda item: item["importance_pct"],
        reverse=True,
    )
    return {
        "artifact": artifact,
        "sample_count": sample_count,
        "oos_sample_count": len(all_oos_actual),
        "feature_names": FEATURE_NAMES,
        "feature_importance": feature_importance,
        "metrics": {
            **prediction_metrics,
            "round_trip_cost_pct": round(round_trip_cost_pct, 4),
            "folds": folds,
            "backtest": simulation,
        },
    }


def validate_model_config(config: dict[str, Any], require_artifact: bool = True) -> None:
    if config.get("model_type") != "lightgbm":
        raise MLTrainingError(f"不支持的模型类型: {config.get('model_type')}")
    if config.get("feature_version") != FEATURE_VERSION:
        raise MLTrainingError("模型特征版本不匹配")
    artifact = str(config.get("artifact", ""))
    if not ARTIFACT_PATTERN.fullmatch(artifact):
        raise MLTrainingError("模型文件标识无效")
    path = (MODEL_DIR / artifact).resolve()
    if path.parent != MODEL_DIR:
        raise MLTrainingError("模型文件路径无效")
    if require_artifact and not path.is_file():
        raise MLTrainingError("模型文件不存在，请重新训练")
    if int(config.get("horizon", 0)) < 1:
        raise MLTrainingError("模型预测周期无效")
    try:
        entry = float(config.get("entry_threshold_pct"))
        exit_value = float(config.get("exit_threshold_pct"))
        slippage = float(config.get("slippage_bps", 0))
    except (TypeError, ValueError) as exc:
        raise MLTrainingError("模型阈值或滑点配置无效") from exc
    if not all(math.isfinite(value) for value in (entry, exit_value, slippage)):
        raise MLTrainingError("模型阈值或滑点必须是有限数值")
    if exit_value >= entry:
        raise MLTrainingError("模型卖出阈值必须低于买入阈值")
    if slippage < 0 or slippage > 1_000:
        raise MLTrainingError("模型滑点必须在 0 到 1000 bps 之间")


@lru_cache(maxsize=32)
def _load_booster(artifact: str) -> lgb.Booster:
    validate_model_config(
        {
            "model_type": "lightgbm",
            "feature_version": FEATURE_VERSION,
            "artifact": artifact,
            "horizon": 1,
            "entry_threshold_pct": 0.25,
            "exit_threshold_pct": 0.0,
        }
    )
    return lgb.Booster(model_file=str(MODEL_DIR / artifact))


def prediction_series(
    candles: list[list[float]], config: dict[str, Any]
) -> list[float | None]:
    validate_model_config(config)
    result: list[float | None] = [None] * len(candles)
    if len(candles) < LOOKBACK:
        return result
    indices = list(range(LOOKBACK - 1, len(candles)))
    x = np.asarray([_feature_row(candles, index) for index in indices], dtype=float)
    predictions = _load_booster(str(config["artifact"])).predict(x)
    for index, prediction in zip(indices, predictions, strict=True):
        result[index] = float(prediction)
    return result


def model_signal(candles: list[list[float]], config: dict[str, Any], index: int) -> str:
    predictions = prediction_series(candles, config)
    prediction = predictions[index]
    if prediction is None:
        return "hold"
    if prediction >= float(config.get("entry_threshold_pct", 0.25)):
        return "buy"
    if prediction <= float(config.get("exit_threshold_pct", 0.0)):
        return "sell"
    return "hold"

