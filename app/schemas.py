from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SUPPORTED_EXCHANGES = {"binance", "okx", "bybit", "kraken"}


class AccountCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    exchange_id: str
    api_key: str = Field(min_length=3)
    api_secret: str = Field(min_length=3)
    passphrase: str | None = None
    sandbox: bool = True

    @model_validator(mode="after")
    def supported_exchange(self) -> "AccountCreate":
        if self.exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(f"暂不支持交易所: {self.exchange_id}")
        return self


class AccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    exchange_id: str
    api_key_hint: str
    sandbox: bool
    enabled: bool
    created_at: datetime


class StrategyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    exchange_id: str = "binance"
    exchange_account_id: int | None = None
    symbol: str = Field(default="BTC/USDT", pattern=r"^[A-Z0-9]+/[A-Z0-9]+$")
    timeframe: str = Field(default="1h", pattern=r"^[1-9][0-9]*[mhdwM]$")
    mode: Literal["paper", "sandbox", "live"] = "paper"
    poll_seconds: int = Field(default=60, ge=10, le=86_400)
    order_quote: float = Field(default=100.0, gt=0)
    initial_cash: float = Field(default=10_000.0, gt=0)
    allocation_mode: Literal["fixed_once", "dynamic_target"] = "fixed_once"
    allocation_pct: float = Field(default=10.0, gt=0, le=100)
    rebalance_band_pct: float = Field(default=5.0, ge=0, le=50)
    stop_loss_pct: float | None = Field(default=3.0, gt=0, le=100)
    take_profit_pct: float | None = Field(default=6.0, gt=0)
    fee_bps: float = Field(default=10.0, ge=0, le=1_000)
    rule: dict[str, Any]

    @model_validator(mode="after")
    def validate_mode(self) -> "StrategyCreate":
        if self.exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(f"暂不支持交易所: {self.exchange_id}")
        if self.mode != "paper" and self.exchange_account_id is None:
            raise ValueError("测试网或实盘策略必须选择交易所账户")
        if self.mode != "paper" and self.allocation_mode == "dynamic_target":
            raise ValueError("动态目标比例目前只允许在模拟盘使用")
        if self.order_quote > self.initial_cash * self.allocation_pct / 100:
            raise ValueError("每次投入不能超过分配给这个策略的资金")
        return self


class StrategyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    exchange_id: str
    exchange_account_id: int | None
    symbol: str
    timeframe: str
    mode: str
    poll_seconds: int
    order_quote: float
    initial_cash: float
    allocation_mode: str
    allocation_pct: float
    rebalance_band_pct: float
    initial_allocation: float
    capital_transfers: float
    last_rebalanced_at: datetime | None
    paper_cash: float
    position_amount: float
    entry_price: float | None
    stop_loss_pct: float | None
    take_profit_pct: float | None
    fee_bps: float
    rule: dict[str, Any]
    enabled: bool
    last_signal: str
    last_error: str | None
    last_run_at: datetime | None
    created_at: datetime


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int
    exchange_order_id: str | None
    mode: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    reason: str | None
    created_at: datetime


class TemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="我保存的策略规则", max_length=500)
    category: str = Field(default="我的策略", min_length=1, max_length=30)
    entry_summary: str = Field(min_length=1, max_length=180)
    exit_summary: str = Field(min_length=1, max_length=180)
    rule: dict[str, Any]
    preset: dict[str, Any] | None = None


class TemplateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    category: str
    entry_summary: str
    exit_summary: str
    rule: dict[str, Any]
    preset: dict[str, Any] | None
    is_builtin: bool
    created_at: datetime

