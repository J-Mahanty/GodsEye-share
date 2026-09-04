"""GodsEye API — city-wide ANPR platform.

    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import router
from api.state import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.bootstrap()
    state.sim_task = asyncio.create_task(state.sim_loop())
    yield
    state.sim_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await state.sim_task


app = FastAPI(
    title="GodsEye",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "City-wide ANPR platform: a high-accuracy plate reader, spatial-temporal "
        f"trajectory reconstruction and macro traffic analytics over the {config.CITY_NAME} "
        "camera network."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "GodsEye", "docs": "/docs", "health": "/api/health"}
