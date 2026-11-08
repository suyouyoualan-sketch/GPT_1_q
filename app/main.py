import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, init_db
from .engine import engine
from .exchange import ExchangeService, compact_ohlcv
from .ml import FEATURE_VERSION, train_lightgbm
from .models import ExchangeAccount, MLExperiment, OrderRecord, Strategy, StrategyTemplate
from .schemas import (
    AccountCreate,
    AccountOut,
    MLExperimentOut,
    MLTrainRequest,
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
        "ml_experiments": db.scalar(select(func.count()).select_from(MLExperiment))
        or 0,
    }


@app.get("/api/market", dependencies=[Depends(require_admin)])
async def market_snapshot(
    exchange_id: str = Query(default="binance"),
    symbol: str = Query(default="BTC/USDT", pattern=r"^[A-Z0-9]+/[A-Z0-9]+$"),
    timeframe: str = Query(default="1h", pattern=r"^[1-9][0-9]*[mhdwM]$"),
    limit: int = Query(default=240, ge=60, le=500),
) -> dict[str, Any]:
    try:
        candles = await ExchangeService.fetch_ohlcv(
            exchange_id, symbol, timeframe, limit=limit
        )
        if not candles:
            raise ValueError("交易所没有返回 K 线")
        first_close = float(candles[0][4])
        last_close = float(candles[-1][4])
        return {
            "exchange_id": exchange_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "last_price": last_close,
            "period_change_pct": round(
                (last_close / first_close - 1) * 100 if first_close else 0, 4
            ),
            "period_high": max(float(row[2]) for row in candles),
            "period_low": min(float(row[3]) for row in candles),
            "period_volume": sum(float(row[5]) for row in candles),
            "candle_count": len(candles),
            "candles": compact_ohlcv(candles, max_points=limit),
        }
    except Exception as exc:
        raise HTTPException(400, f"行情加载失败: {exc}") from exc


@app.get(
    "/api/ml/experiments",
    response_model=list[MLExperimentOut],
    dependencies=[Depends(require_admin)],
)
async def list_ml_experiments(db: Db) -> list[MLExperiment]:
    return list(db.scalars(select(MLExperiment).order_by(MLExperiment.id.desc())))


@app.post(
    "/api/ml/train",
    response_model=MLExperimentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def train_ml_experiment(payload: MLTrainRequest, db: Db) -> MLExperiment:
    experiment = MLExperiment(
        **payload.model_dump(),
        status="training",
        feature_version=FEATURE_VERSION,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    try:
        start_ms = int(
            datetime.combine(payload.train_start, time.min, tzinfo=timezone.utc).timestamp()
            * 1_000
        )
        end_ms = int(
            datetime.combine(
                payload.train_end + timedelta(days=1), time.min, tzinfo=timezone.utc
            ).timestamp()
            * 1_000
        )
        candles = await ExchangeService.fetch_ohlcv_range(
            payload.exchange_id,
            payload.symbol,
            payload.timeframe,
            start_ms,
            end_ms,
            max_candles=30_000,
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1_000)
        candles = [row for row in candles if int(row[0]) + 3_600_000 <= now_ms]
        if len(candles) < 700:
            raise ValueError(
                f"交易所只返回了 {len(candles)} 根 K 线，至少需要 700 根"
            )
        result = await asyncio.to_thread(
            train_lightgbm,
            candles,
            experiment_id=experiment.id,
            artifact_token=secrets.token_hex(6),
            horizon=payload.horizon,
            fee_bps=payload.fee_bps,
            slippage_bps=payload.slippage_bps,
            entry_threshold_pct=payload.entry_threshold_pct,
            exit_threshold_pct=payload.exit_threshold_pct,
        )
        experiment.status = "ready"
        experiment.candle_count = len(candles)
        experiment.sample_count = result["sample_count"]
        experiment.oos_sample_count = result["oos_sample_count"]
        experiment.artifact = result["artifact"]
        experiment.feature_names = result["feature_names"]
        experiment.feature_importance = result["feature_importance"]
        experiment.metrics = result["metrics"]
        experiment.error = None
        db.commit()
        db.refresh(experiment)
        return experiment
    except Exception as exc:
        experiment.status = "failed"
        experiment.error = str(exc)[:1_000]
        db.commit()
        raise HTTPException(400, f"模型训练失败: {exc}") from exc


@app.post(
    "/api/ml/experiments/{experiment_id}/library",
    response_model=TemplateOut,
    dependencies=[Depends(require_admin)],
)
async def save_ml_to_library(experiment_id: int, db: Db) -> StrategyTemplate:
    experiment = db.get(MLExperiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "训练记录不存在")
    if experiment.status != "ready" or not experiment.artifact:
        raise HTTPException(409, "模型尚未训练完成")
    slug = f"ml-{experiment.artifact.removesuffix('.txt')}"
    existing = db.scalar(select(StrategyTemplate).where(StrategyTemplate.slug == slug))
    if existing:
        return existing
    rule = {
        "ml_model": {
            "experiment_id": experiment.id,
            "model_type": experiment.model_type,
            "artifact": experiment.artifact,
            "feature_version": experiment.feature_version,
            "exchange_id": experiment.exchange_id,
            "symbol": experiment.symbol,
            "timeframe": experiment.timeframe,
            "training_end": experiment.train_end.isoformat(),
            "horizon": experiment.horizon,
            "training_fee_bps": experiment.fee_bps,
            "slippage_bps": experiment.slippage_bps,
            "entry_threshold_pct": experiment.entry_threshold_pct,
            "exit_threshold_pct": experiment.exit_threshold_pct,
        }
    }
    try:
        _validate_rule(rule)
    except StrategyError as exc:
        raise HTTPException(422, f"模型策略无效: {exc}") from exc
    item = StrategyTemplate(
        slug=slug,
        name=experiment.name,
        description=(
            f"{experiment.symbol} · {experiment.timeframe} · LightGBM；"
            f"预测未来 {experiment.horizon} 根 K 线的扣费后收益。"
        ),
        category="机器学习",
        entry_summary=f"预测净收益 ≥ {experiment.entry_threshold_pct}% 时买入",
        exit_summary=f"预测净收益 ≤ {experiment.exit_threshold_pct}% 时平仓",
        rule=rule,
        preset=None,
        is_builtin=False,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.delete(
    "/api/ml/experiments/{experiment_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_ml_experiment(experiment_id: int, db: Db) -> None:
    experiment = db.get(MLExperiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "训练记录不存在")
    db.delete(experiment)
    db.commit()


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
    ml_config = payload.rule.get("ml_model")
    if ml_config and any(
        (
            ml_config.get("exchange_id") != payload.exchange_id,
            ml_config.get("symbol") != payload.symbol,
            ml_config.get("timeframe") != payload.timeframe,
        )
    ):
        raise HTTPException(422, "机器学习策略必须使用训练时的交易所、交易对和周期")
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
        ml_config = strategy.rule.get("ml_model")
        if ml_config and candles:
            training_end = date.fromisoformat(str(ml_config["training_end"]))
            backtest_start = datetime.fromtimestamp(
                candles[0][0] / 1_000, tz=timezone.utc
            ).date()
            if backtest_start <= training_end:
                raise ValueError(
                    "机器学习模型只能回测训练结束日期之后的行情；"
                    "训练区间的可靠表现请查看机器学习实验室中的样本外滚动验证"
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
        result["exchange_id"] = strategy.exchange_id
        result["symbol"] = strategy.symbol
        result["timeframe"] = strategy.timeframe
        result["price_candles"] = compact_ohlcv(candles)
        candles_by_timestamp = {int(row[0]): row for row in candles}
        for trade in result["trades"]:
            row = candles_by_timestamp.get(int(trade["timestamp"]))
            if row:
                trade["candle"] = {
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
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

