import pytest

from app.strategy import StrategyError, backtest, strategy_signal


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
    assert result["closed_trades"] == 1
    assert result["final_equity"] < 1_000
    assert result["open_position"] == 0


def test_invalid_indicator_is_rejected() -> None:
    rule = threshold_rule()
    rule["entry"]["left"] = {"indicator": "magic", "period": 2}
    with pytest.raises(StrategyError, match="不支持的指标"):
        strategy_signal(candles([98, 99, 101]), rule)


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

