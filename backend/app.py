"""
backend/app.py
--------------
FastAPI app serving the dashboard and the simulation API.

    python -m backend.app            # http://127.0.0.1:8000
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from backend.inference import get_engine  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_engine()
    except FileNotFoundError as e:
        # Keep serving the static dashboard; /api/simulate will report the problem.
        print(f"⚠ Inference engine not available: {e}")
    yield


app = FastAPI(title="Transit Demand & Frequency API", lifespan=lifespan)

# Only needed when the dashboard is opened from another origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SimulationRequest(BaseModel):
    day_index: int = Field(0, ge=0, le=366)
    bus_capacity: int = Field(config.BUS_CAPACITY, ge=10, le=300)
    fixed_headway: int = Field(config.FIXED_HEADWAY_MIN, ge=1, le=60)


@app.get("/api/health")
def health():
    try:
        engine = get_engine()
        return {"status": "ok", "days": len(engine.days), "stops": engine.data.stops}
    except FileNotFoundError as e:
        return {"status": "no-model", "detail": str(e)}


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    try:
        engine = get_engine()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    try:
        return engine.run_simulation(req.day_index, req.bus_capacity, req.fixed_headway)
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/initial_data")
def initial_data():
    path = config.DASHBOARD_DIR / "data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run: python dashboard/data_export.py")
    return json.loads(path.read_text())


app.mount("/", StaticFiles(directory=config.DASHBOARD_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)))
