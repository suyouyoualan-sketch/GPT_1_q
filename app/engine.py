import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .exchange import ExchangeService
from .models import ExchangeAccount, OrderRecord, Strategy
from .strategy import strategy_signal


class TradingEngine:
    def __init__(self) -> None:
        self.tasks: dict[int, asyncio.Task[None]] = {}

    def start(self, strategy_id: int) -> None:
        current = self.tasks.get(strategy_id)
        if current and not current.done():
            return
        self.tasks[strategy_id] = asyncio.create_task(
            self._run(strategy_id), name=f"strategy-{strategy_id}"
        )

    async def stop(self, strategy_id: int) -> None:
        task = self.tasks.pop(strategy_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def stop_all(self) -> None:
        for strategy_id in list(self.tasks):
            await self.stop(strategy_id)

    def start_enabled(self) -> None:
        with SessionLocal() as db:
            ids = db.scalars(select(Strategy.id).where(Strategy.enabled.is_(True))).all()
        for strategy_id in ids:
            self.start(strategy_id)

    async def _run(self, strategy_id: int) -> None:
        while True:
            delay = 60
            try:
                delay = await self.tick(strategy_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep an isolated bot failure from killing the service
                with SessionLocal() as db:
                    strategy = db.get(Strategy, strategy_id)
                    if not strategy or not strategy.enabled:
                        return
                    strategy.last_error = str(exc)[:1_000]
                    strategy.last_run_at = datetime.now(timezone.utc)
                    delay = strategy.poll_seconds
                    db.commit()
            await asyncio.sleep(max(delay, 10))

    async def tick(self, strategy_id: int) -> int:
        with SessionLocal() as db:
            strategy = db.get(Strategy, strategy_id)
            if not strategy or not strategy.enabled:
                return 86_400
            account = (
                db.get(ExchangeAccount, strategy.exchange_account_id)
                if strategy.exchange_account_id
                else None
            )
            snapshot = {
                "exchange_id": strategy.exchange_id,
                "symbol": strategy.symbol,
                "timeframe": strategy.timeframe,
                "rule": strategy.rule,
            }

        candles = await ExchangeService.fetch_ohlcv(
            snapshot["exchange_id"],
            snapshot["symbol"],
            snapshot["timeframe"],
            limit=300,
            account=account if strategy.mode != "paper" else None,
        )
        signal = strategy_signal(candles, snapshot["rule"])
        price = float(candles[-1][4])

        with SessionLocal() as db:
            strategy = db.get(Strategy, strategy_id)
            if not strategy or not strategy.enabled:
                return 86_400
            effective_signal, reason = self._risk_signal(strategy, signal, price)
            if effective_signal in {"buy", "sell"} and effective_signal != strategy.last_signal:
                if strategy.mode == "paper":
                    self._paper_order(db, strategy, effective_signal, price, reason)
                else:
                    await self._exchange_order(db, strategy, effective_signal, reason)
            strategy.last_signal = effective_signal
            strategy.last_error = None
            strategy.last_run_at = datetime.now(timezone.utc)
            delay = strategy.poll_seconds
            db.commit()
            return delay

    @staticmethod
    def _risk_signal(strategy: Strategy, signal: str, price: float) -> tuple[str, str]:
        if strategy.position_amount > 0 and strategy.entry_price:
            if strategy.stop_loss_pct and price <= strategy.entry_price * (
                1 - strategy.stop_loss_pct / 100
            ):
                return "sell", "stop_loss"
            if strategy.take_profit_pct and price >= strategy.entry_price * (
                1 + strategy.take_profit_pct / 100
            ):
                return "sell", "take_profit"
        return signal, "rule"

    @staticmethod
    def _paper_order(
        db: Session, strategy: Strategy, side: str, price: float, reason: str
    ) -> None:
        fee_rate = strategy.fee_bps / 10_000
        if side == "buy" and strategy.position_amount == 0:
            TradingEngine._refresh_dynamic_allocation(db, strategy)
            budget = min(strategy.order_quote, strategy.paper_cash)
            if budget <= 0:
                raise RuntimeError("模拟账户可用资金不足")
            amount = budget * (1 - fee_rate) / price
            strategy.paper_cash -= budget
            strategy.position_amount = amount
            strategy.entry_price = price
        elif side == "sell" and strategy.position_amount > 0:
            amount = strategy.position_amount
            strategy.paper_cash += amount * price * (1 - fee_rate)
            strategy.position_amount = 0.0
            strategy.entry_price = None
        else:
            return
        db.add(
            OrderRecord(
                strategy_id=strategy.id,
                mode="paper",
                symbol=strategy.symbol,
                side=side,
                amount=amount,
                price=price,
                status="closed",
                reason=reason,
            )
        )

    @staticmethod
    def _refresh_dynamic_allocation(db: Session, strategy: Strategy) -> None:
        if strategy.allocation_mode != "dynamic_target":
            return
        paper_strategies = list(
            db.scalars(select(Strategy).where(Strategy.mode == "paper"))
        )
        portfolio_base = max(
            (item.initial_cash for item in paper_strategies),
            default=strategy.initial_cash,
        )
        strategy_performance = sum(
            (
                item.paper_cash
                + item.position_amount * float(item.entry_price or 0)
                - item.initial_allocation
                - item.capital_transfers
            )
            for item in paper_strategies
        )
        portfolio_equity = max(portfolio_base + strategy_performance, 0)
        target = portfolio_equity * strategy.allocation_pct / 100
        current = strategy.paper_cash
        drift_pct = abs(current - target) / target * 100 if target else 0
        if drift_pct < strategy.rebalance_band_pct:
            return
        transfer = target - current
        strategy.paper_cash = target
        strategy.capital_transfers += transfer
        strategy.last_rebalanced_at = datetime.now(timezone.utc)

    @staticmethod
    async def _exchange_order(
        db: Session, strategy: Strategy, side: str, reason: str
    ) -> None:
        if strategy.mode == "live" and not settings.allow_live_trading:
            raise RuntimeError("实盘交易被禁用；需设置 ALLOW_LIVE_TRADING=true")
        account = db.get(ExchangeAccount, strategy.exchange_account_id)
        if not account or not account.enabled:
            raise RuntimeError("交易所账户不存在或已禁用")
        if strategy.mode == "sandbox" and not account.sandbox:
            raise RuntimeError("测试网策略只能使用测试网账户")
        if strategy.mode == "live" and account.sandbox:
            raise RuntimeError("实盘策略不能使用测试网账户")
        if side == "buy" and strategy.position_amount > 0:
            return
        if side == "sell" and strategy.position_amount <= 0:
            return

        result = await ExchangeService.market_order(
            account,
            strategy.symbol,
            side,
            strategy.order_quote,
            strategy.position_amount,
        )
        if side == "buy":
            strategy.position_amount = result["amount"]
            strategy.entry_price = result["price"]
        else:
            strategy.position_amount = 0.0
            strategy.entry_price = None
        db.add(
            OrderRecord(
                strategy_id=strategy.id,
                exchange_order_id=result["id"],
                mode=strategy.mode,
                symbol=strategy.symbol,
                side=side,
                amount=result["amount"],
                price=result["price"],
                status=result["status"],
                reason=reason,
                raw=result["raw"],
            )
        )


engine = TradingEngine()

