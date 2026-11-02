import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import Base
from app.engine import TradingEngine
from app.models import Strategy
from app.schemas import StrategyCreate
from app.strategy import DEFAULT_RULE


def strategy(**overrides) -> Strategy:
    values = {
        "name": "allocation-test",
        "mode": "paper",
        "rule": DEFAULT_RULE,
        "initial_cash": 10_000,
        "allocation_pct": 20,
        "initial_allocation": 2_000,
        "paper_cash": 2_000,
    }
    values.update(overrides)
    return Strategy(**values)


def test_dynamic_allocation_rebalances_before_new_entry() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        dynamic = strategy(
            allocation_mode="dynamic_target",
            paper_cash=3_000,
            rebalance_band_pct=5,
        )
        fixed = strategy(
            name="fixed",
            allocation_mode="fixed_once",
            allocation_pct=30,
            initial_allocation=3_000,
            paper_cash=3_000,
        )
        db.add_all([dynamic, fixed])
        db.commit()

        TradingEngine._refresh_dynamic_allocation(db, dynamic)

        # Portfolio equity is 11,000 after the dynamic sleeve's 1,000 profit.
        assert dynamic.paper_cash == pytest.approx(2_200)
        assert dynamic.capital_transfers == pytest.approx(-800)
        assert dynamic.last_rebalanced_at is not None


def test_fixed_allocation_never_moves_capital() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        fixed = strategy(allocation_mode="fixed_once", paper_cash=2_600)
        db.add(fixed)
        db.commit()

        TradingEngine._refresh_dynamic_allocation(db, fixed)

        assert fixed.paper_cash == 2_600
        assert fixed.capital_transfers == 0


def test_dynamic_allocation_requires_paper_mode() -> None:
    with pytest.raises(ValidationError, match="动态目标比例目前只允许在模拟盘使用"):
        StrategyCreate(
            name="unsafe-live-dynamic",
            mode="live",
            exchange_account_id=1,
            allocation_mode="dynamic_target",
            allocation_pct=20,
            initial_cash=10_000,
            order_quote=100,
            rule=DEFAULT_RULE,
        )

