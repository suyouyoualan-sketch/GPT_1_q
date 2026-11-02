from sqlalchemy import select

from .db import SessionLocal
from .models import StrategyTemplate


BUILTIN_TEMPLATES = [
    {
        "slug": "classic-sma-20-50",
        "name": "经典双均线",
        "description": "最适合入门的趋势策略。规则直观、交易频率较低，适合先理解趋势跟随。",
        "category": "趋势",
        "entry_summary": "SMA 20 上穿 SMA 50",
        "exit_summary": "SMA 20 下穿 SMA 50",
        "preset": {"template": "sma_cross", "fast_period": 20, "slow_period": 50},
        "rule": {
            "entry": {"all": [{"left": {"indicator": "sma", "period": 20}, "op": "crosses_above", "right": {"indicator": "sma", "period": 50}}]},
            "exit": {"any": [{"left": {"indicator": "sma", "period": 20}, "op": "crosses_below", "right": {"indicator": "sma", "period": 50}}]},
        },
    },
    {
        "slug": "classic-ema-12-26",
        "name": "EMA 快速趋势",
        "description": "使用反应更快的指数均线，能较早发现趋势变化，同时也更容易遇到假信号。",
        "category": "趋势",
        "entry_summary": "EMA 12 上穿 EMA 26",
        "exit_summary": "EMA 12 下穿 EMA 26",
        "preset": {"template": "ema_cross", "fast_period": 12, "slow_period": 26},
        "rule": {
            "entry": {"all": [{"left": {"indicator": "ema", "period": 12}, "op": "crosses_above", "right": {"indicator": "ema", "period": 26}}]},
            "exit": {"any": [{"left": {"indicator": "ema", "period": 12}, "op": "crosses_below", "right": {"indicator": "ema", "period": 26}}]},
        },
    },
    {
        "slug": "classic-rsi-14",
        "name": "RSI 超买超卖",
        "description": "在震荡行情中寻找价格过度下跌后的反弹机会，单边趋势中需要谨慎使用。",
        "category": "震荡",
        "entry_summary": "RSI(14) 低于 30",
        "exit_summary": "RSI(14) 高于 70",
        "preset": {"template": "rsi_reversal", "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70},
        "rule": {
            "entry": {"left": {"indicator": "rsi", "period": 14}, "op": "<", "right": {"value": 30}},
            "exit": {"left": {"indicator": "rsi", "period": 14}, "op": ">", "right": {"value": 70}},
        },
    },
    {
        "slug": "classic-price-sma-50",
        "name": "价格突破均线",
        "description": "价格重新站上中期均线时跟随，跌回均线下方时离场，规则简洁清楚。",
        "category": "突破",
        "entry_summary": "收盘价上穿 SMA 50",
        "exit_summary": "收盘价下穿 SMA 50",
        "preset": {"template": "price_ma", "ma_period": 50},
        "rule": {
            "entry": {"left": {"field": "close"}, "op": "crosses_above", "right": {"indicator": "sma", "period": 50}},
            "exit": {"left": {"field": "close"}, "op": "crosses_below", "right": {"indicator": "sma", "period": 50}},
        },
    },
    {
        "slug": "classic-trend-rsi-filter",
        "name": "趋势 + RSI 过滤",
        "description": "双均线确认趋势，同时用 RSI 避免在已经过热时追高，是更稳健的组合示例。",
        "category": "组合",
        "entry_summary": "SMA 20 上穿 SMA 50，且 RSI(14) 低于 65",
        "exit_summary": "SMA 20 下穿 SMA 50，或 RSI(14) 高于 75",
        "preset": None,
        "rule": {
            "entry": {"all": [
                {"left": {"indicator": "sma", "period": 20}, "op": "crosses_above", "right": {"indicator": "sma", "period": 50}},
                {"left": {"indicator": "rsi", "period": 14}, "op": "<", "right": {"value": 65}},
            ]},
            "exit": {"any": [
                {"left": {"indicator": "sma", "period": 20}, "op": "crosses_below", "right": {"indicator": "sma", "period": 50}},
                {"left": {"indicator": "rsi", "period": 14}, "op": ">", "right": {"value": 75}},
            ]},
        },
    },
]


def seed_builtin_templates() -> None:
    with SessionLocal() as db:
        existing = set(db.scalars(select(StrategyTemplate.slug).where(StrategyTemplate.slug.is_not(None))))
        for definition in BUILTIN_TEMPLATES:
            if definition["slug"] not in existing:
                db.add(StrategyTemplate(**definition, is_builtin=True))
        db.commit()
