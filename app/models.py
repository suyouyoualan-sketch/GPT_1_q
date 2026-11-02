from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExchangeAccount(Base):
    __tablename__ = "exchange_accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    exchange_id: Mapped[str] = mapped_column(String(30), index=True)
    api_key_enc: Mapped[str] = mapped_column(Text)
    api_secret_enc: Mapped[str] = mapped_column(Text)
    passphrase_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_hint: Mapped[str] = mapped_column(String(12))
    sandbox: Mapped[bool] = mapped_column(Boolean, default=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    strategies: Mapped[list["Strategy"]] = relationship(back_populates="account")


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    exchange_id: Mapped[str] = mapped_column(String(30), default="binance")
    exchange_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("exchange_accounts.id", ondelete="SET NULL"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(30), default="BTC/USDT")
    timeframe: Mapped[str] = mapped_column(String(10), default="1h")
    mode: Mapped[str] = mapped_column(String(12), default="paper")
    poll_seconds: Mapped[int] = mapped_column(Integer, default=60)
    order_quote: Mapped[float] = mapped_column(Float, default=100.0)
    initial_cash: Mapped[float] = mapped_column(Float, default=10_000.0)
    allocation_mode: Mapped[str] = mapped_column(String(20), default="fixed_once")
    allocation_pct: Mapped[float] = mapped_column(Float, default=10.0)
    rebalance_band_pct: Mapped[float] = mapped_column(Float, default=5.0)
    initial_allocation: Mapped[float] = mapped_column(Float, default=1_000.0)
    capital_transfers: Mapped[float] = mapped_column(Float, default=0.0)
    last_rebalanced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paper_cash: Mapped[float] = mapped_column(Float, default=10_000.0)
    position_amount: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee_bps: Mapped[float] = mapped_column(Float, default=10.0)
    rule: Mapped[dict[str, Any]] = mapped_column(JSON)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    last_signal: Mapped[str] = mapped_column(String(12), default="hold")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    account: Mapped[ExchangeAccount | None] = relationship(back_populates="strategies")
    orders: Mapped[list["OrderRecord"]] = relationship(back_populates="strategy")


class OrderRecord(Base):
    __tablename__ = "order_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id", ondelete="CASCADE"))
    exchange_order_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    mode: Mapped[str] = mapped_column(String(12))
    symbol: Mapped[str] = mapped_column(String(30))
    side: Mapped[str] = mapped_column(String(8))
    amount: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="closed")
    reason: Mapped[str | None] = mapped_column(String(80), nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    strategy: Mapped[Strategy] = relationship(back_populates="orders")


class StrategyTemplate(Base):
    __tablename__ = "strategy_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30), default="自定义")
    entry_summary: Mapped[str] = mapped_column(String(180))
    exit_summary: Mapped[str] = mapped_column(String(180))
    rule: Mapped[dict[str, Any]] = mapped_column(JSON)
    preset: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

