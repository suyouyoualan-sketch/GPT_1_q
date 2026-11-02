from typing import Any

import ccxt.async_support as ccxt

from .models import ExchangeAccount
from .schemas import SUPPORTED_EXCHANGES
from .security import decrypt_secret


class ExchangeService:
    @staticmethod
    def build(
        exchange_id: str,
        account: ExchangeAccount | None = None,
        sandbox: bool | None = None,
    ) -> Any:
        if exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(f"暂不支持交易所: {exchange_id}")
        exchange_class = getattr(ccxt, exchange_id)
        options: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 20_000,
        }
        if account:
            options.update(
                {
                    "apiKey": decrypt_secret(account.api_key_enc),
                    "secret": decrypt_secret(account.api_secret_enc),
                }
            )
            if account.passphrase_enc:
                options["password"] = decrypt_secret(account.passphrase_enc)
        exchange = exchange_class(options)
        use_sandbox = account.sandbox if account and sandbox is None else bool(sandbox)
        if use_sandbox:
            exchange.set_sandbox_mode(True)
        return exchange

    @classmethod
    async def test_account(cls, account: ExchangeAccount) -> dict[str, Any]:
        exchange = cls.build(account.exchange_id, account)
        try:
            markets = await exchange.load_markets()
            balance = await exchange.fetch_balance()
            nonzero = {
                code: float(total)
                for code, total in (balance.get("total") or {}).items()
                if total and float(total) != 0
            }
            return {
                "ok": True,
                "markets": len(markets),
                "nonzero_assets": nonzero,
                "sandbox": account.sandbox,
            }
        finally:
            await exchange.close()

    @classmethod
    async def fetch_ohlcv(
        cls,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        account: ExchangeAccount | None = None,
    ) -> list[list[float]]:
        exchange = cls.build(exchange_id, account, sandbox=account.sandbox if account else False)
        try:
            candles = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return [[float(v) for v in row] for row in candles]
        finally:
            await exchange.close()

    @classmethod
    async def market_order(
        cls,
        account: ExchangeAccount,
        symbol: str,
        side: str,
        quote_budget: float,
        position_amount: float = 0.0,
    ) -> dict[str, Any]:
        exchange = cls.build(account.exchange_id, account)
        try:
            await exchange.load_markets()
            ticker = await exchange.fetch_ticker(symbol)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            if price <= 0:
                raise RuntimeError("交易所未返回有效最新价格")
            raw_amount = quote_budget / price if side == "buy" else position_amount
            if raw_amount <= 0:
                raise RuntimeError("下单数量必须大于零")
            amount = float(exchange.amount_to_precision(symbol, raw_amount))
            if amount <= 0:
                raise RuntimeError("按交易所精度处理后的下单数量为零")
            order = await exchange.create_order(symbol, "market", side, amount)
            return {
                "id": str(order.get("id") or ""),
                "status": str(order.get("status") or "open"),
                "amount": float(order.get("filled") or order.get("amount") or amount),
                "price": float(order.get("average") or order.get("price") or price),
                "raw": {
                    "id": order.get("id"),
                    "status": order.get("status"),
                    "filled": order.get("filled"),
                    "average": order.get("average"),
                },
            }
        finally:
            await exchange.close()

