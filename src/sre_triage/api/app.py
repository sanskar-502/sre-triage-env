from __future__ import annotations

import os
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app

from sre_triage.core.environment import SREEnvironment
from sre_triage.schemas import HealthResponse, ReadyResponse, ResetRequest, SREAction, SREObservation
from sre_triage.settings import settings
from sre_triage.telemetry.logging import APP_START_TIME, REQUEST_METRICS, configure_logging, get_logger, log_event


logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging(settings.log_level)

    env_instance = SREEnvironment()
    app = create_fastapi_app(
        env=lambda: env_instance,
        action_cls=SREAction,
        observation_cls=SREObservation,
    )
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) in {"/reset", "/health"}
            and ("POST" in getattr(route, "methods", set()) or "GET" in getattr(route, "methods", set()))
        )
    ]

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        if settings.require_api_key and request.url.path not in {"/", "/health", "/ready"}:
            token = request.headers.get("x-api-key")
            if token != settings.service_api_key:
                raise HTTPException(status_code=401, detail="Missing or invalid API key")

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        REQUEST_METRICS[f"{request.method} {request.url.path}"] += 1
        response.headers["x-request-id"] = request_id
        log_event(
            logger,
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "path": request.url.path},
        )

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "SRE Triage Simulator Running"}

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", service=settings.service_name)

    @app.get("/ready", response_model=ReadyResponse)
    async def ready():
        return ReadyResponse(
            status="ready",
            service=settings.service_name,
            require_api_key=settings.require_api_key,
        )

    @app.get("/metrics")
    async def metrics():
        uptime_seconds = round(time.time() - APP_START_TIME, 2)
        return {
            "service": settings.service_name,
            "uptime_seconds": uptime_seconds,
            "request_counts": dict(REQUEST_METRICS),
        }

    @app.post("/reset")
    async def reset_with_task(request: Request):
        try:
            body = ResetRequest(**(await request.json()))
        except Exception:
            body = ResetRequest()
        obs = env_instance.reset(difficulty=body.difficulty, task_id=body.task_id)
        return {"observation": obs.model_dump(), "reward": 0.0, "done": False}

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "sre_triage.api.app:app",
        host=os.getenv("HOST", settings.host),
        port=int(os.getenv("PORT", str(settings.port))),
        workers=int(os.getenv("WORKERS", str(settings.workers))),
    )
