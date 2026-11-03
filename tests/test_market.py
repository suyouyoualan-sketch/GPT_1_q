import asyncio

from app.exchange import ExchangeService
from app.main import market_snapshot


def test_market_snapshot_returns_chart_and_summary(monkeypatch) -> None:
    rows = [
        [1_700_000_000_000 + i * 3_600_000, 100 + i, 102 + i, 99 + i, 101 + i, 10]
        for i in range(60)
    ]

    async def fake_fetch(*args, **kwargs):
        return rows

    monkeypatch.setattr(ExchangeService, "fetch_ohlcv", fake_fetch)

    result = asyncio.run(market_snapshot("binance", "BTC/USDT", "1h", 60))

    assert result["symbol"] == "BTC/USDT"
    assert result["last_price"] == 160
    assert result["period_high"] == 161
    assert result["period_low"] == 99
    assert result["period_volume"] == 600
    assert len(result["candles"]) == 60

