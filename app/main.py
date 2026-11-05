import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
VLLM_MODEL = os.getenv(
    "VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "10"))
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "5"))
HEALTH_PROBE_INTERVAL = float(os.getenv("HEALTH_PROBE_INTERVAL", "15"))
MAX_CONCURRENT_REQUESTS = max(
    1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
)
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))

BACKENDS = ("ollama", "vllm")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("llmops.api")


# Prometheus metrics
HTTP_REQUESTS = Counter(
    "llmops_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "llmops_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

HTTP_ERRORS = Counter(
    "llmops_http_errors_total",
    "Total HTTP requests returning an error status",
    ["method", "endpoint", "status"],
)

BACKEND_UP = Gauge(
    "llmops_backend_up",
    "1 when the backend is reachable and serving the configured model",
    ["backend"],
)

INFLIGHT = Gauge(
    "llmops_inflight_requests",
    "Generation requests currently queued or running",
    ["backend"],
)

GENERATION_DURATION = Histogram(
    "llmops_generation_duration_seconds",
    "End-to-end model generation duration in seconds",
    ["backend"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

TIME_TO_FIRST_TOKEN = Histogram(
    "llmops_time_to_first_token_seconds",
    "Time until the first streamed token in seconds",
    ["backend"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

LLM_TOKENS = Counter(
    "llmops_tokens_total",
    "Tokens processed by the model backend",
    ["backend", "type"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=max(1, MAX_CONNECTIONS // 5),
    )
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
        limits=limits,
    )
    app.state.probe = httpx.AsyncClient(timeout=HEALTH_TIMEOUT)
    app.state.semaphores = {
        backend: asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        for backend in BACKENDS
    }

    # Publish zero-valued gauges immediately so dashboards are not blank
    # before the first probe or request.
    for backend in BACKENDS:
        INFLIGHT.labels(backend=backend).set(0)
        BACKEND_UP.labels(backend=backend).set(0)

    probe_task = None
    if HEALTH_PROBE_INTERVAL > 0:
        probe_task = asyncio.create_task(health_probe_loop(app))

    try:
        yield
    finally:
        if probe_task is not None:
            probe_task.cancel()
            with suppress(asyncio.CancelledError):
                await probe_task
        await app.state.probe.aclose()
        await app.state.http.aclose()


app = FastAPI(
    title="Local LLMOps API",
    version="2.1.0",
    description=(
        "One OpenAI-ish API for Ollama and vLLM inference, with "
        "streaming, token accounting, health probes and Prometheus metrics."
    ),
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=120, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GenerateResponse(BaseModel):
    request_id: str
    backend: str
    model: str
    response: str
    latency_seconds: float
    time_to_first_token_seconds: float | None = None
    usage: Usage = Field(default_factory=Usage)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled error request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    duration = time.perf_counter() - start

    # Use route templates to avoid high-cardinality path labels.
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or "unmatched"
    method = request.method
    status = str(response.status_code)

    HTTP_REQUESTS.labels(
        method=method,
        endpoint=endpoint,
        status=status,
    ).inc()

    # Streaming responses return before generation finishes, so the HTTP
    # duration would only measure time-to-headers. Generation duration and
    # TTFT are recorded by the stream handler instead.
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("text/event-stream"):
        HTTP_REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

    if response.status_code >= 400:
        HTTP_ERRORS.labels(
            method=method,
            endpoint=endpoint,
            status=status,
        ).inc()

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration * 1000,
    )

    return response


def model_is_available(configured: str, available: list[str]) -> bool:
    """Match a configured model against what the backend reports.

    Ollama may report ``qwen2.5:0.5b`` or ``qwen2.5:0.5b`` depending on how
    the model was pulled; a missing tag implicitly means ``:latest``.
    """
    for name in available:
        if name == configured:
            return True
        if ":" not in configured and name == f"{configured}:latest":
            return True
        if ":" not in name and configured == f"{name}:latest":
            return True
    return False


async def check_backend(client: httpx.AsyncClient, backend: str) -> dict:
    if backend == "ollama":
        url = f"{OLLAMA_URL}/api/tags"
        model = OLLAMA_MODEL
    else:
        url = f"{VLLM_URL}/v1/models"
        model = VLLM_MODEL

    available: list[str] = []

    try:
        result = await client.get(url, timeout=HEALTH_TIMEOUT)
        result.raise_for_status()
        data = result.json()

        if backend == "ollama":
            available = [
                item.get("name") or item.get("model") or ""
                for item in data.get("models", [])
            ]
        else:
            available = [
                item.get("id") or ""
                for item in data.get("data", [])
            ]
        available = [name for name in available if name]

        backend_status = (
            "ok" if model_is_available(model, available) else "model_missing"
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        backend_status = "unavailable"

    ready = backend_status == "ok"
    BACKEND_UP.labels(backend=backend).set(1 if ready else 0)

    return {
        "backend": backend,
        "backend_status": backend_status,
        "ready": ready,
        "configured_model": model,
        "available_models": available[:20],
    }


async def health_probe_loop(app: FastAPI):
    """Keep backend gauges fresh even when /health is not being called."""
    while True:
        try:
            results = await asyncio.gather(
                *(
                    check_backend(app.state.probe, backend)
                    for backend in BACKENDS
                )
            )
            logger.debug(
                "health_probe %s",
                {
                    item["backend"]: item["backend_status"]
                    for item in results
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("health_probe_failed")

        await asyncio.sleep(HEALTH_PROBE_INTERVAL)


def build_upstream_request(
    payload: GenerateRequest,
    backend: str,
    stream: bool,
):
    if backend == "ollama":
        body = {
            "model": OLLAMA_MODEL,
            "prompt": payload.prompt,
            "stream": stream,
            "options": {
                "temperature": payload.temperature,
                "num_predict": payload.max_tokens,
            },
        }
        return f"{OLLAMA_URL}/api/generate", body, OLLAMA_MODEL

    body = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": payload.prompt}],
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return f"{VLLM_URL}/v1/chat/completions", body, VLLM_MODEL


def usage_from_ollama(data: dict) -> Usage:
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def usage_from_openai(data: dict) -> Usage:
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(
        usage.get("total_tokens") or prompt_tokens + completion_tokens
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def record_usage(backend: str, usage: Usage) -> None:
    LLM_TOKENS.labels(backend=backend, type="prompt").inc(
        usage.prompt_tokens
    )
    LLM_TOKENS.labels(backend=backend, type="completion").inc(
        usage.completion_tokens
    )


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root():
    return {
        "service": "Local LLMOps API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "readiness": "/readyz",
        "liveness": "/livez",
        "metrics": "/metrics",
        "generation_endpoints": [
            "/generate/ollama",
            "/generate/vllm",
            "/generate/ollama/stream",
            "/generate/vllm/stream",
        ],
    }


@app.get("/livez")
async def liveness():
    return {"status": "alive"}


@app.get("/health")
async def health(request: Request):
    backends = await asyncio.gather(
        *(check_backend(request.app.state.http, b) for b in BACKENDS)
    )
    by_name = {item["backend"]: item for item in backends}
    healthy = all(item["ready"] for item in backends)

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "backends": by_name,
        },
    )


@app.get("/readyz")
async def readiness(request: Request):
    """Ready when at least one backend can serve its configured model."""
    backends = await asyncio.gather(
        *(check_backend(request.app.state.http, b) for b in BACKENDS)
    )
    by_name = {item["backend"]: item for item in backends}
    ready = any(item["ready"] for item in backends)

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "backends": by_name,
        },
    )


@app.get("/health/{backend}")
async def backend_health(request: Request, backend: str):
    if backend not in BACKENDS:
        raise HTTPException(
            404,
            "Unknown backend. Use ollama or vllm.",
        )

    result = await check_backend(request.app.state.http, backend)

    return JSONResponse(
        status_code=200 if result["ready"] else 503,
        content={
            "status": "ok" if result["ready"] else "degraded",
            **result,
        },
    )


async def generate_with_backend(
    payload: GenerateRequest,
    request: Request,
    backend: str,
):
    client = request.app.state.http
    request_id = request.state.request_id
    start = time.perf_counter()

    url, body, model = build_upstream_request(payload, backend, stream=False)
    semaphore = request.app.state.semaphores[backend]

    INFLIGHT.labels(backend=backend).inc()
    try:
        async with semaphore:
            result = await client.post(url, json=body)
            result.raise_for_status()
            data = result.json()

        if backend == "ollama":
            answer = data["response"]
            usage = usage_from_ollama(data)
        else:
            answer = data["choices"][0]["message"]["content"]
            usage = usage_from_openai(data)

        latency = time.perf_counter() - start

        GENERATION_DURATION.labels(backend=backend).observe(latency)
        record_usage(backend, usage)

        logger.info(
            "generation_complete request_id=%s backend=%s "
            "latency_ms=%.1f prompt_tokens=%d completion_tokens=%d",
            request_id,
            backend,
            latency * 1000,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return GenerateResponse(
            request_id=request_id,
            backend=backend,
            model=model,
            response=answer.strip(),
            latency_seconds=round(latency, 4),
            usage=usage,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            504,
            "Model backend timed out",
            headers={"X-Request-ID": request_id},
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "backend_http_error request_id=%s backend=%s status=%s",
            request_id,
            backend,
            exc.response.status_code,
        )
        raise HTTPException(
            502,
            "Model backend returned an error",
            headers={"X-Request-ID": request_id},
        )
    except httpx.RequestError:
        logger.warning(
            "backend_connection_error request_id=%s backend=%s",
            request_id,
            backend,
        )
        raise HTTPException(
            503,
            "Model backend is unavailable",
            headers={"X-Request-ID": request_id},
        )
    except (KeyError, IndexError, TypeError, ValueError):
        logger.exception(
            "invalid_backend_response request_id=%s backend=%s",
            request_id,
            backend,
        )
        raise HTTPException(
            502,
            "Invalid response from model backend",
            headers={"X-Request-ID": request_id},
        )
    finally:
        INFLIGHT.labels(backend=backend).dec()


async def stream_generation(
    payload: GenerateRequest,
    request: Request,
    backend: str,
):
    client = request.app.state.http
    request_id = request.state.request_id
    start = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    usage = Usage()
    url, body, model = build_upstream_request(payload, backend, stream=True)
    semaphore = request.app.state.semaphores[backend]

    INFLIGHT.labels(backend=backend).inc()
    try:
        async with semaphore:
            async with client.stream("POST", url, json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    logger.warning(
                        "backend_stream_error request_id=%s backend=%s "
                        "status=%s",
                        request_id,
                        backend,
                        response.status_code,
                    )
                    yield sse_event(
                        "error",
                        {
                            "request_id": request_id,
                            "detail": "Model backend returned an error",
                            "status": response.status_code,
                        },
                    )
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if backend == "ollama":
                        data = json.loads(line)
                        delta = data.get("response") or ""
                        if data.get("done"):
                            usage = usage_from_ollama(data)
                    else:
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:") :].strip()
                        if raw == "[DONE]":
                            break
                        data = json.loads(raw)
                        if data.get("usage"):
                            usage = usage_from_openai(data)
                        choices = data.get("choices") or []
                        delta = (
                            choices[0].get("delta", {}).get("content") or ""
                            if choices
                            else ""
                        )

                    if not delta:
                        continue

                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        TIME_TO_FIRST_TOKEN.labels(backend=backend).observe(
                            first_token_at - start
                        )

                    chunks.append(delta)
                    yield sse_event("token", {"delta": delta})

        latency = time.perf_counter() - start
        answer = "".join(chunks).strip()

        GENERATION_DURATION.labels(backend=backend).observe(latency)
        record_usage(backend, usage)

        logger.info(
            "generation_stream_complete request_id=%s backend=%s "
            "latency_ms=%.1f ttft_ms=%s completion_tokens=%d",
            request_id,
            backend,
            latency * 1000,
            (
                f"{(first_token_at - start) * 1000:.1f}"
                if first_token_at
                else "n/a"
            ),
            usage.completion_tokens,
        )

        yield sse_event(
            "done",
            {
                "request_id": request_id,
                "backend": backend,
                "model": model,
                "response": answer,
                "latency_seconds": round(latency, 4),
                "time_to_first_token_seconds": (
                    round(first_token_at - start, 4)
                    if first_token_at
                    else None
                ),
                "usage": usage.model_dump(),
            },
        )

    except httpx.TimeoutException:
        yield sse_event(
            "error",
            {
                "request_id": request_id,
                "detail": "Model backend timed out",
            },
        )
    except httpx.RequestError:
        logger.warning(
            "backend_stream_connection_error request_id=%s backend=%s",
            request_id,
            backend,
        )
        yield sse_event(
            "error",
            {
                "request_id": request_id,
                "detail": "Model backend is unavailable",
            },
        )
    except (KeyError, IndexError, TypeError, ValueError):
        logger.exception(
            "invalid_backend_stream_response request_id=%s backend=%s",
            request_id,
            backend,
        )
        yield sse_event(
            "error",
            {
                "request_id": request_id,
                "detail": "Invalid response from model backend",
            },
        )
    finally:
        INFLIGHT.labels(backend=backend).dec()


@app.post("/generate/ollama", response_model=GenerateResponse)
async def generate_ollama(payload: GenerateRequest, request: Request):
    return await generate_with_backend(payload, request, "ollama")


@app.post("/generate/vllm", response_model=GenerateResponse)
async def generate_vllm(payload: GenerateRequest, request: Request):
    return await generate_with_backend(payload, request, "vllm")


@app.post("/generate", response_model=GenerateResponse)
async def generate_default(
    payload: GenerateRequest,
    request: Request,
    backend: str = "ollama",
):
    if backend not in BACKENDS:
        raise HTTPException(
            400,
            "Unknown backend. Use ollama or vllm.",
            headers={"X-Request-ID": request.state.request_id},
        )

    return await generate_with_backend(payload, request, backend)


@app.post("/generate/{backend}/stream")
async def generate_stream(
    payload: GenerateRequest,
    request: Request,
    backend: str,
):
    if backend not in BACKENDS:
        raise HTTPException(
            404,
            "Unknown backend. Use ollama or vllm.",
            headers={"X-Request-ID": request.state.request_id},
        )

    return StreamingResponse(
        stream_generation(payload, request, backend),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request.state.request_id,
        },
    )
