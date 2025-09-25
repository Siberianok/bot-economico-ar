from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..data_sources import dolar, reserves
from ..data_sources.models import Quote, ReserveStatus


@dataclass(slots=True)
class SummaryData:
    dolar_quotes: list[Quote]
    oficial_quotes: list[Quote]
    reserves: ReserveStatus


class MetricsPipeline:
    async def fetch_summary(self) -> SummaryData:
        async with asyncio.TaskGroup() as tg:
            dolar_task = tg.create_task(dolar.fetch_dolar_quotes())
            oficial_task = tg.create_task(dolar.fetch_oficial_blue())
            reserves_task = tg.create_task(reserves.fetch_reserves())
        return SummaryData(
            dolar_quotes=dolar_task.result(),
            oficial_quotes=oficial_task.result(),
            reserves=reserves_task.result(),
        )
