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
    {
        "slug": "alpha-momentum-20",
        "name": "动量 Alpha",
        "description": "衡量最近一段时间的涨跌幅，趋势足够强时进入，动量转负时离场。适合用来认识最直观的 Alpha 因子。",
        "category": "Alpha因子",
        "entry_summary": "20 周期动量高于 3%",
        "exit_summary": "20 周期动量低于 0%",
        "preset": {
            "template": "alpha_momentum",
            "momentum_period": 20,
            "momentum_entry": 3,
            "momentum_exit": 0,
        },
        "rule": {
            "entry": {
                "left": {"indicator": "momentum", "period": 20},
                "op": ">",
                "right": {"value": 3},
            },
            "exit": {
                "left": {"indicator": "momentum", "period": 20},
                "op": "<",
                "right": {"value": 0},
            },
        },
    },
    {
        "slug": "alpha-zscore-20",
        "name": "Z-score 均值回归 Alpha",
        "description": "把价格偏离近期均值的程度标准化。价格明显偏低时进入，回到均值附近时离场，适合震荡市场。",
        "category": "Alpha因子",
        "entry_summary": "价格 Z-score 低于 -2",
        "exit_summary": "价格 Z-score 高于 0",
        "preset": {
            "template": "alpha_zscore",
            "zscore_period": 20,
            "zscore_entry": -2,
            "zscore_exit": 0,
        },
        "rule": {
            "entry": {
                "left": {"indicator": "zscore", "period": 20},
                "op": "<",
                "right": {"value": -2},
            },
            "exit": {
                "left": {"indicator": "zscore", "period": 20},
                "op": ">",
                "right": {"value": 0},
            },
        },
    },
    {
        "slug": "alpha-multi-factor",
        "name": "量价低波动 Alpha",
        "description": "同时检查上涨动量、成交量放大和波动率上限。条件更严格，用来演示如何把多个因子组合成一个信号。",
        "category": "Alpha因子",
        "entry_summary": "动量 > 3%、量比 > 1.5、波动率 < 2%",
        "exit_summary": "动量 < 0% 或波动率 > 4%",
        "preset": {
            "template": "alpha_multi",
            "momentum_period": 20,
            "momentum_entry": 3,
            "momentum_exit": 0,
            "volume_period": 20,
            "volume_ratio_entry": 1.5,
            "volatility_period": 20,
            "volatility_max": 2,
            "volatility_exit": 4,
        },
        "rule": {
            "entry": {
                "all": [
                    {
                        "left": {"indicator": "momentum", "period": 20},
                        "op": ">",
                        "right": {"value": 3},
                    },
                    {
                        "left": {"indicator": "volume_ratio", "period": 20},
                        "op": ">",
                        "right": {"value": 1.5},
                    },
                    {
                        "left": {"indicator": "volatility", "period": 20},
                        "op": "<",
                        "right": {"value": 2},
                    },
                ]
            },
            "exit": {
                "any": [
                    {
                        "left": {"indicator": "momentum", "period": 20},
                        "op": "<",
                        "right": {"value": 0},
                    },
                    {
                        "left": {"indicator": "volatility", "period": 20},
                        "op": ">",
                        "right": {"value": 4},
                    },
                ]
            },
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
