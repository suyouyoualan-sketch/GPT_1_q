from app.library import BUILTIN_TEMPLATES
from app.strategy import strategy_signal


def test_builtin_library_rules_are_unique_and_valid() -> None:
    slugs = [item["slug"] for item in BUILTIN_TEMPLATES]
    assert len(slugs) == len(set(slugs))
    assert len(BUILTIN_TEMPLATES) >= 5

    candles = [
        [1_700_000_000_000 + i * 60_000, 100 + i, 102 + i, 99 + i, 101 + i, 10]
        for i in range(120)
    ]
    for item in BUILTIN_TEMPLATES:
        assert strategy_signal(candles, item["rule"]) in {"buy", "sell", "hold"}

