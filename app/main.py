from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, init_db
from .engine import engine
from .exchange import ExchangeService
from .models import ExchangeAccount, OrderRecord, Strategy, StrategyTemplate
from .schemas import (
    AccountCreate,
    AccountOut,
    OrderOut,
    StrategyCreate,
    StrategyOut,
    TemplateCreate,
    TemplateOut,
)
from .security import encrypt_secret, require_admin
from .strategy import DEFAULT_RULE, StrategyError, backtest, strategy_signal


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    engine.start_enabled()
    yield
    await engine.stop_all()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

Db = Annotated[Session, Depends(get_db)]
Admin = Annotated[None, Depends(require_admin)]


def _validate_rule(rule: dict[str, Any]) -> None:
    candles = [
        [1_700_000_000_000 + i * 60_000, 100 + i, 102 + i, 99 + i, 101 + i, 10]
        for i in range(120)
    ]
    strategy_signal(candles, rule)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": settings.app_name,
        "live_trading_enabled": settings.allow_live_trading,
    }


@app.get("/api/meta", dependencies=[Depends(require_admin)])
async def meta() -> dict[str, Any]:
    return {
        "exchanges": ["binance", "okx", "bybit", "kraken"],
        "live_trading_enabled": settings.allow_live_trading,
        "default_rule": DEFAULT_RULE,
        "dsl": {
            "indicators": [
                "sma",
                "ema",
                "rsi",
                "momentum",
                "volatility",
                "zscore",
                "volume_ratio",
            ],
            "fields": ["open", "high", "low", "close", "volume"],
            "operators": [">", "<", ">=", "<=", "crosses_above", "crosses_below"],
        },
    }


@app.get("/api/overview", dependencies=[Depends(require_admin)])
async def overview(db: Db) -> dict[str, Any]:
    return {
        "accounts": db.scalar(select(func.count()).select_from(ExchangeAccount)) or 0,
        "strategies": db.scalar(select(func.count()).select_from(Strategy)) or 0,
        "running": db.scalar(
            select(func.count()).select_from(Strategy).where(Strategy.enabled.is_(True))
        )
        or 0,
        "orders": db.scalar(select(func.count()).select_from(OrderRecord)) or 0,
    }


@app.get("/api/accounts", response_model=list[AccountOut], dependencies=[Depends(require_admin)])
async def list_accounts(db: Db) -> list[ExchangeAccount]:
    return list(db.scalars(select(ExchangeAccount).order_by(ExchangeAccount.id.desc())))


@app.post(
    "/api/accounts",
    response_model=AccountOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_account(payload: AccountCreate, db: Db) -> ExchangeAccount:
    account = ExchangeAccount(
        name=payload.name,
        exchange_id=payload.exchange_id,
        api_key_enc=encrypt_secret(payload.api_key),
        api_secret_enc=encrypt_secret(payload.api_secret),
        passphrase_enc=encrypt_secret(payload.passphrase) if payload.passphrase else None,
        api_key_hint=f"••••{payload.api_key[-4:]}",
        sandbox=payload.sandbox,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


@app.post("/api/accounts/{account_id}/test", dependencies=[Depends(require_admin)])
async def test_account(account_id: int, db: Db) -> dict[str, Any]:
    account = db.get(ExchangeAccount, account_id)
    if not account:
        raise HTTPException(404, "账户不存在")
    try:
        return await ExchangeService.test_account(account)
    except Exception as exc:
        raise HTTPException(400, f"连接失败: {exc}") from exc


@app.delete("/api/accounts/{account_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_account(account_id: int, db: Db) -> None:
    account = db.get(ExchangeAccount, account_id)
    if not account:
        raise HTTPException(404, "账户不存在")
    used = db.scalar(
        select(func.count()).select_from(Strategy).where(Strategy.exchange_account_id == account_id)
    )
    if used:
        raise HTTPException(409, "该账户仍被策略使用，请先删除相关策略")
    db.delete(account)
    db.commit()


@app.get("/api/strategies", response_model=list[StrategyOut], dependencies=[Depends(require_admin)])
async def list_strategies(db: Db) -> list[Strategy]:
    return list(db.scalars(select(Strategy).order_by(Strategy.id.desc())))


@app.post(
    "/api/strategies",
    response_model=StrategyOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_strategy(payload: StrategyCreate, db: Db) -> Strategy:
    try:
        _validate_rule(payload.rule)
    except StrategyError as exc:
        raise HTTPException(422, f"策略规则无效: {exc}") from exc
    if payload.exchange_account_id:
        account = db.get(ExchangeAccount, payload.exchange_account_id)
        if not account:
            raise HTTPException(422, "所选交易所账户不存在")
        if account.exchange_id != payload.exchange_id:
            raise HTTPException(422, "策略交易所与账户交易所不一致")
        if payload.mode == "sandbox" and not account.sandbox:
            raise HTTPException(422, "测试网策略必须选择测试网账户")
        if payload.mode == "live" and account.sandbox:
            raise HTTPException(422, "实盘策略必须选择实盘账户")
    allocated = payload.initial_cash * payload.allocation_pct / 100
    strategy = Strategy(
        **payload.model_dump(),
        paper_cash=allocated,
        initial_allocation=allocated,
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return strategy


@app.post("/api/strategies/{strategy_id}/start", response_model=StrategyOut, dependencies=[Depends(require_admin)])
async def start_strategy(strategy_id: int, db: Db) -> Strategy:
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    if strategy.mode == "live" and not settings.allow_live_trading:
        raise HTTPException(409, "实盘交易未启用；请先设置 ALLOW_LIVE_TRADING=true")
    strategy.enabled = True
    strategy.last_error = None
    db.commit()
    engine.start(strategy.id)
    return strategy


@app.post("/api/strategies/{strategy_id}/stop", response_model=StrategyOut, dependencies=[Depends(require_admin)])
async def stop_strategy(strategy_id: int, db: Db) -> Strategy:
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    strategy.enabled = False
    db.commit()
    await engine.stop(strategy.id)
    return strategy


@app.post("/api/strategies/{strategy_id}/backtest", dependencies=[Depends(require_admin)])
async def run_backtest(
    strategy_id: int,
    db: Db,
    limit: int = Query(default=500, ge=100, le=1_500),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
) -> dict[str, Any]:
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    if (start is None) != (end is None):
        raise HTTPException(422, "开始日期和结束日期必须同时填写")
    if start and end and start > end:
        raise HTTPException(422, "开始日期不能晚于结束日期")
    try:
        if start and end:
            start_ms = int(
                datetime.combine(start, time.min, tzinfo=timezone.utc).timestamp()
                * 1_000
            )
            end_ms = int(
                datetime.combine(
                    end + timedelta(days=1), time.min, tzinfo=timezone.utc
                ).timestamp()
                * 1_000
            )
            candles = await ExchangeService.fetch_ohlcv_range(
                strategy.exchange_id,
                strategy.symbol,
                strategy.timeframe,
                start_ms,
                end_ms,
            )
        else:
            candles = await ExchangeService.fetch_ohlcv(
                strategy.exchange_id, strategy.symbol, strategy.timeframe, limit=limit
            )
        result = backtest(
            candles,
            strategy.rule,
            initial_cash=strategy.initial_cash,
            order_quote=strategy.order_quote,
            fee_bps=strategy.fee_bps,
            stop_loss_pct=strategy.stop_loss_pct,
            take_profit_pct=strategy.take_profit_pct,
        )
        result["candle_count"] = len(candles)
        result["data_start"] = datetime.fromtimestamp(
            candles[0][0] / 1_000, tz=timezone.utc
        ).date().isoformat()
        result["data_end"] = datetime.fromtimestamp(
            candles[-1][0] / 1_000, tz=timezone.utc
        ).date().isoformat()
        return result
    except Exception as exc:
        raise HTTPException(400, f"回测失败: {exc}") from exc


@app.delete("/api/strategies/{strategy_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_strategy(strategy_id: int, db: Db) -> None:
    strategy = db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(404, "策略不存在")
    strategy.enabled = False
    db.commit()
    await engine.stop(strategy.id)
    db.delete(strategy)
    db.commit()


@app.get("/api/orders", response_model=list[OrderOut], dependencies=[Depends(require_admin)])
async def list_orders(
    db: Db, limit: int = Query(default=100, ge=1, le=500)
) -> list[OrderRecord]:
    return list(
        db.scalars(select(OrderRecord).order_by(OrderRecord.id.desc()).limit(limit))
    )


@app.get(
    "/api/library",
    response_model=list[TemplateOut],
    dependencies=[Depends(require_admin)],
)
async def list_library(db: Db) -> list[StrategyTemplate]:
    return list(
        db.scalars(
            select(StrategyTemplate).order_by(
                StrategyTemplate.is_builtin.desc(), StrategyTemplate.id.desc()
            )
        )
    )


@app.post(
    "/api/library",
    response_model=TemplateOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_library_item(payload: TemplateCreate, db: Db) -> StrategyTemplate:
    try:
        _validate_rule(payload.rule)
    except StrategyError as exc:
        raise HTTPException(422, f"策略规则无效: {exc}") from exc
    item = StrategyTemplate(**payload.model_dump(), is_builtin=False)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.delete(
    "/api/library/{template_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_library_item(template_id: int, db: Db) -> None:
    item = db.get(StrategyTemplate, template_id)
    if not item:
        raise HTTPException(404, "策略模板不存在")
    if item.is_builtin:
        raise HTTPException(409, "经典策略不能删除")
    db.delete(item)
    db.commit()

