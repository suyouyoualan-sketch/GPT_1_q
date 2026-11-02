from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


if settings.database_url.startswith("sqlite:///./"):
    Path(settings.database_url.removeprefix("sqlite:///./")).parent.mkdir(
        parents=True, exist_ok=True
    )


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


async def get_db() -> AsyncGenerator[Session, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401
    from .library import seed_builtin_templates

    Base.metadata.create_all(bind=engine)
    _migrate_strategy_allocation_columns()
    seed_builtin_templates()


def _migrate_strategy_allocation_columns() -> None:
    """Keep existing SQLite development databases compatible without data loss."""
    columns = {column["name"] for column in inspect(engine).get_columns("strategies")}
    additions = {
        "allocation_mode": "VARCHAR(20) NOT NULL DEFAULT 'fixed_once'",
        "allocation_pct": "FLOAT NOT NULL DEFAULT 100",
        "rebalance_band_pct": "FLOAT NOT NULL DEFAULT 5",
        "initial_allocation": "FLOAT NOT NULL DEFAULT 0",
        "capital_transfers": "FLOAT NOT NULL DEFAULT 0",
        "last_rebalanced_at": "DATETIME NULL",
    }
    added_initial_allocation = "initial_allocation" not in columns
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE strategies ADD COLUMN {name} {definition}"))
        if added_initial_allocation:
            connection.execute(
                text(
                    "UPDATE strategies SET initial_allocation = initial_cash, "
                    "allocation_pct = 100"
                )
            )

