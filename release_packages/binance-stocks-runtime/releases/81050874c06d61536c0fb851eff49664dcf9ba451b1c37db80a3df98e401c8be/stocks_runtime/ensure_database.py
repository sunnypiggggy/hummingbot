from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit

from stocks_runtime.settings import dedicated_database_url


async def main() -> None:
    import asyncpg

    target = dedicated_database_url(
        os.environ["DATABASE_URL"], os.getenv("BINANCE_STOCKS_DATABASE_NAME", "hummingbot_stocks")
    )
    raw = target.replace("postgresql+asyncpg://", "postgresql://", 1)
    parsed = urlsplit(raw)
    database = parsed.path.lstrip("/")
    if not database.replace("_", "").isalnum():
        raise ValueError("unsafe PostgreSQL database name")
    admin = urlunsplit((parsed.scheme, parsed.netloc, "/postgres", parsed.query, parsed.fragment))
    connection = await asyncpg.connect(admin)
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", database)
        if not exists:
            await connection.execute(f'CREATE DATABASE "{database}"')
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
