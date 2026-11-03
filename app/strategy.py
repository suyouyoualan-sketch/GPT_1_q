from dataclasses import dataclass
from typing import Any


class StrategyError(ValueError):
    pass


def _series(candles: list[list[float]], operand: dict[str, Any]) -> list[float | None]:
    count = len(candles)
    if "value" in operand:
        return [float(operand["value"])] * count
    if "field" in operand:
        fields = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}
        field = operand["field"]
        if field not in fields:
            raise StrategyError(f"不支持的价格字段: {field}")
        return [float(row[fields[field]]) for row in candles]

    indicator = operand.get("indicator")
    period = int(operand.get("period", 0))
    if period < 2:
        raise StrategyError("指标 period 必须至少为 2")
    closes = [float(row[4]) for row in candles]

    if indicator == "sma":
        values: list[float | None] = [None] * count
        running = 0.0
        for i, value in enumerate(closes):
            running += value
            if i >= period:
                running -= closes[i - period]
            if i >= period - 1:
                values[i] = running / period
        return values

    if indicator == "ema":
        values = [None] * count
        if count < period:
            return values
        seed = sum(closes[:period]) / period
        values[period - 1] = seed
        multiplier = 2 / (period + 1)
        for i in range(period, count):
            seed = (closes[i] - seed) * multiplier + seed
            values[i] = seed
        return values

    if indicator == "rsi":
        values = [None] * count
        if count <= period:
            return values
        gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, count)]
        losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, count)]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        values[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        for i in range(period + 1, count):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            values[i] = (
                100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
            )
        return values

    raise StrategyError(f"不支持的指标: {indicator}")


def _condition(
    candles: list[list[float]], condition: dict[str, Any], index: int
) -> bool:
    if not {"left", "op", "right"}.issubset(condition):
        raise StrategyError("每个条件都需要 left、op 和 right")
    left = _series(candles, condition["left"])
    right = _series(candles, condition["right"])
    current_left, current_right = left[index], right[index]
    if current_left is None or current_right is None:
        return False
    op = condition["op"]
    if op == ">":
        return current_left > current_right
    if op == "<":
        return current_left < current_right
    if op == ">=":
        return current_left >= current_right
    if op == "<=":
        return current_left <= current_right
    if op in {"crosses_above", "crosses_below"}:
        if index < 1 or left[index - 1] is None or right[index - 1] is None:
            return False
        if op == "crosses_above":
            return left[index - 1] <= right[index - 1] and current_left > current_right
        return left[index - 1] >= right[index - 1] and current_left < current_right
    raise StrategyError(f"不支持的比较操作: {op}")


def evaluate_rule(candles: list[list[float]], rule: dict[str, Any], index: int) -> bool:
    if "all" in rule:
        items = rule["all"]
        if not isinstance(items, list) or not items:
            raise StrategyError("all 必须是非空条件数组")
        return all(evaluate_rule(candles, item, index) for item in items)
    if "any" in rule:
        items = rule["any"]
        if not isinstance(items, list) or not items:
            raise StrategyError("any 必须是非空条件数组")
        return any(evaluate_rule(candles, item, index) for item in items)
    return _condition(candles, rule, index)


def strategy_signal(candles: list[list[float]], config: dict[str, Any], index: int = -1) -> str:
    if len(candles) < 3:
        raise StrategyError("K 线数量不足")
    index = index if index >= 0 else len(candles) - 1
    if "entry" not in config or "exit" not in config:
        raise StrategyError("策略需要 entry 和 exit 规则")
    if evaluate_rule(candles, config["entry"], index):
        return "buy"
    if evaluate_rule(candles, config["exit"], index):
        return "sell"
    return "hold"


@dataclass
class BacktestTrade:
    timestamp: int
    side: str
    price: float
    amount: float
    reason: str


def backtest(
    candles: list[list[float]],
    config: dict[str, Any],
    initial_cash: float = 10_000.0,
    order_quote: float = 1_000.0,
    fee_bps: float = 10.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> dict[str, Any]:
    if len(candles) < 20:
        raise StrategyError("回测至少需要 20 根 K 线")
    cash = float(initial_cash)
    position = 0.0
    entry_price: float | None = None
    fee_rate = fee_bps / 10_000
    trades: list[BacktestTrade] = []
    equity_curve: list[dict[str, float | int]] = []
    peak = initial_cash
    max_drawdown = 0.0
    first_price = float(candles[0][4])

    for i in range(1, len(candles)):
        timestamp, price = int(candles[i][0]), float(candles[i][4])
        reason = "rule"
        signal = strategy_signal(candles, config, i)
        if position > 0 and entry_price:
            if stop_loss_pct and price <= entry_price * (1 - stop_loss_pct / 100):
                signal, reason = "sell", "stop_loss"
            elif take_profit_pct and price >= entry_price * (1 + take_profit_pct / 100):
                signal, reason = "sell", "take_profit"

        if signal == "buy" and position == 0:
            budget = min(order_quote, cash)
            if budget > 0:
                fee = budget * fee_rate
                amount = (budget - fee) / price
                cash -= budget
                position = amount
                entry_price = price
                trades.append(BacktestTrade(timestamp, "buy", price, amount, reason))
        elif signal == "sell" and position > 0:
            amount = position
            cash += amount * price * (1 - fee_rate)
            position = 0.0
            entry_price = None
            trades.append(BacktestTrade(timestamp, "sell", price, amount, reason))

        equity = cash + position * price
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0)
        if i % max(1, len(candles) // 120) == 0 or i == len(candles) - 1:
            equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": round(equity, 4),
                    "benchmark_equity": round(initial_cash * price / first_price, 4),
                }
            )

    final_price = float(candles[-1][4])
    final_equity = cash + position * final_price
    return_pct = (final_equity / initial_cash - 1) * 100
    benchmark_return_pct = (final_price / first_price - 1) * 100
    sells = [trade for trade in trades if trade.side == "sell"]
    wins = 0
    buy_price = None
    for trade in trades:
        if trade.side == "buy":
            buy_price = trade.price
        elif buy_price is not None and trade.price > buy_price:
            wins += 1
            buy_price = None

    return {
        "initial_cash": initial_cash,
        "final_equity": round(final_equity, 4),
        "return_pct": round(return_pct, 4),
        "benchmark_return_pct": round(benchmark_return_pct, 4),
        "excess_return_pct": round(return_pct - benchmark_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "closed_trades": len(sells),
        "win_rate_pct": round(wins / len(sells) * 100, 2) if sells else 0.0,
        "open_position": round(position, 10),
        "trades": [trade.__dict__ for trade in trades],
        "equity_curve": equity_curve,
    }


DEFAULT_RULE: dict[str, Any] = {
    "entry": {
        "all": [
            {
                "left": {"indicator": "sma", "period": 20},
                "op": "crosses_above",
                "right": {"indicator": "sma", "period": 50},
            }
        ]
    },
    "exit": {
        "any": [
            {
                "left": {"indicator": "sma", "period": 20},
                "op": "crosses_below",
                "right": {"indicator": "sma", "period": 50},
            }
        ]
    },
}

