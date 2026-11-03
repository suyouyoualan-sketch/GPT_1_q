import asyncio

from app.exchange import ExchangeService


class FakeExchange:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.closed = False

    @staticmethod
    def parse_timeframe(_: str) -> int:
        return 60

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float]]:
        return [row for row in self.rows if row[0] >= since][:limit]

    async def close(self) -> None:
        self.closed = True


def test_fetch_ohlcv_range_filters_end_and_closes(monkeypatch) -> None:
    start = 1_700_000_000_000
    rows = [
        [start + i * 60_000, 100, 101, 99, 100 + i, 10]
        for i in range(6)
    ]
    fake = FakeExchange(rows)
    monkeypatch.setattr(ExchangeService, "build", lambda *args, **kwargs: fake)

    result = asyncio.run(
        ExchangeService.fetch_ohlcv_range(
            "binance",
            "BTC/USDT",
            "1m",
            start,
            start + 3 * 60_000,
        )
    )

    assert [row[0] for row in result] == [row[0] for row in rows[:3]]
    assert fake.closed is True
