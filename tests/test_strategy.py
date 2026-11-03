import pytest

from app.strategy import StrategyError, _compute_series, backtest, strategy_signal


def candles(prices: list[float]) -> list[list[float]]:
    return [
        [1_700_000_000_000 + i * 60_000, p, p + 1, p - 1, p, 10]
        for i, p in enumerate(prices)
    ]


def threshold_rule() -> dict:
    return {
        "entry": {
            "left": {"field": "close"},
            "op": "crosses_above",
            "right": {"value": 100},
        },
        "exit": {
            "left": {"field": "close"},
            "op": "crosses_below",
            "right": {"value": 100},
        },
    }


def test_strategy_detects_crossings() -> None:
    assert strategy_signal(candles([98, 99, 101]), threshold_rule()) == "buy"
    assert strategy_signal(candles([102, 101, 99]), threshold_rule()) == "sell"
    assert strategy_signal(candles([98, 99, 99.5]), threshold_rule()) == "hold"


def test_backtest_executes_round_trip_and_fees() -> None:
    prices = [98] * 20 + [101, 103, 105, 99, 97]
    result = backtest(
        candles(prices),
        threshold_rule(),
        initial_cash=1_000,
        order_quote=500,
        fee_bps=10,
    )
    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert [trade["action"] for trade in result["trades"]] == ["open", "close"]
    assert [trade["decision"] for trade in result["trades"]] == [
        "买入开仓",
        "策略卖出平仓",
    ]
    assert result["closed_trades"] == 1
    assert result["final_equity"] < 1_000
    assert result["open_position"] == 0
    assert "benchmark_return_pct" in result
    assert "excess_return_pct" in result
    assert all("benchmark_equity" in point for point in result["equity_curve"])


def test_invalid_indicator_is_rejected() -> None:
    rule = threshold_rule()
    rule["entry"]["left"] = {"indicator": "magic", "period": 2}
    with pytest.raises(StrategyError, match="不支持的指标"):
        strategy_signal(candles([98, 99, 101]), rule)


def test_alpha_factor_values() -> None:
    rows = candles([100, 102, 105])
    rows[0][5], rows[1][5], rows[2][5] = 10, 10, 30

    momentum = _compute_series(rows, {"indicator": "momentum", "period": 2})
    volatility = _compute_series(rows, {"indicator": "volatility", "period": 2})
    zscore = _compute_series(rows, {"indicator": "zscore", "period": 2})
    volume_ratio = _compute_series(rows, {"indicator": "volume_ratio", "period": 2})

    assert momentum[-1] == pytest.approx(5)
    assert volatility[-1] == pytest.approx(0.470588, rel=1e-5)
    assert zscore[-1] == pytest.approx(1)
    assert volume_ratio[-1] == pytest.approx(1.5)


def test_zscore_alpha_can_trigger_a_buy_signal() -> None:
    rule = {
        "entry": {
            "left": {"indicator": "zscore", "period": 3},
            "op": "<",
            "right": {"value": -1},
        },
        "exit": {
            "left": {"indicator": "zscore", "period": 3},
            "op": ">",
            "right": {"value": 0},
        },
    }
    assert strategy_signal(candles([100, 95, 90]), rule) == "buy"


def test_stop_loss_closes_position() -> None:
    prices = [98] * 20 + [101, 100, 96, 95]
    result = backtest(
        candles(prices),
        threshold_rule(),
        initial_cash=1_000,
        order_quote=500,
        stop_loss_pct=3,
    )
    assert result["trades"][-1]["side"] == "sell"
    assert result["trades"][-1]["reason"] == "stop_loss"
    assert result["trades"][-1]["decision"] == "止损平仓"

