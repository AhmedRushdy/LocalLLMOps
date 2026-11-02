import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as api


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def mock_response(data, status_code=200):
    request = httpx.Request(
        "GET",
        "http://testserver/mock",
    )
    return httpx.Response(
        status_code,
        json=data,
        request=request,
    )


OLLAMA_TAGS = {"models": [{"name": "qwen2.5:0.5b"}]}
VLLM_MODELS = {"data": [{"id": "Qwen/Qwen2.5-0.5B-Instruct"}]}


class FakeStream:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


def parse_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((event, data))
    return events


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/generate/ollama" in response.json()["generation_endpoints"]
    assert "/generate/vllm" in response.json()["generation_endpoints"]
    assert "/generate/ollama/stream" in response.json()["generation_endpoints"]


def test_livez(client):
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "Hello", "max_tokens": 9999},
        {"prompt": "Hello", "temperature": 3},
    ],
)
def test_invalid_request_returns_422(client, payload):
    response = client.post("/generate/ollama", json=payload)
    assert response.status_code == 422


def test_ollama_generation(client):
    api.app.state.http.post = AsyncMock(
        return_value=mock_response(
            {
                "response": "Ollama test answer",
                "prompt_eval_count": 7,
                "eval_count": 3,
            }
        )
    )

    response = client.post(
        "/generate/ollama",
        json={"prompt": "Hello", "max_tokens": 20},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "ollama"
    assert body["response"] == "Ollama test answer"
    assert body["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_vllm_generation(client):
    api.app.state.http.post = AsyncMock(
        return_value=mock_response(
            {
                "choices": [
                    {"message": {"content": "vLLM test answer"}}
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 4,
                    "total_tokens": 9,
                },
            }
        )
    )

    response = client.post(
        "/generate/vllm",
        json={"prompt": "Hello", "max_tokens": 20},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "vllm"
    assert body["response"] == "vLLM test answer"
    assert body["usage"]["total_tokens"] == 9


def test_request_id_matches_header(client):
    api.app.state.http.post = AsyncMock(
        return_value=mock_response({"response": "Test"})
    )

    response = client.post(
        "/generate/ollama",
        headers={"X-Request-ID": "phase4-test-123"},
        json={"prompt": "Hello"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "phase4-test-123"
    assert response.json()["request_id"] == "phase4-test-123"


def test_health_reports_both_backends(client):
    api.app.state.http.get = AsyncMock(
        side_effect=[
            mock_response(OLLAMA_TAGS),
            mock_response(VLLM_MODELS),
        ]
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["backends"]["ollama"]["backend_status"] == "ok"
    assert response.json()["backends"]["vllm"]["backend_status"] == "ok"


def test_health_reports_unavailable_backend(client):
    api.app.state.http.get = AsyncMock(
        side_effect=[
            httpx.ConnectError("Ollama unavailable"),
            mock_response(VLLM_MODELS),
        ]
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert (
        response.json()["backends"]["ollama"]["backend_status"]
        == "unavailable"
    )
    assert response.json()["backends"]["vllm"]["backend_status"] == "ok"


def test_health_detects_missing_model(client):
    api.app.state.http.get = AsyncMock(
        side_effect=[
            mock_response(OLLAMA_TAGS),
            mock_response({"data": [{"id": "some/other-model"}]}),
        ]
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["backends"]["vllm"]["backend_status"] == (
        "model_missing"
    )
    assert response.json()["backends"]["vllm"]["ready"] is False


def test_readyz_is_ready_when_one_backend_works(client):
    api.app.state.http.get = AsyncMock(
        side_effect=[
            mock_response(OLLAMA_TAGS),
            mock_response({"data": [{"id": "some/other-model"}]}),
        ]
    )

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readyz_is_not_ready_when_no_backend_works(client):
    api.app.state.http.get = AsyncMock(
        side_effect=[
            httpx.ConnectError("Ollama unavailable"),
            httpx.ConnectError("vLLM unavailable"),
        ]
    )

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_unknown_backend_returns_404(client):
    response = client.get("/health/unknown")
    assert response.status_code == 404


def test_unknown_generate_backend_returns_400(client):
    response = client.post(
        "/generate?backend=unknown",
        json={"prompt": "Hello"},
    )
    assert response.status_code == 400


def test_stream_ollama(client):
    lines = [
        '{"response": "Hello", "done": false}',
        '{"response": " world", "done": false}',
        (
            '{"response": "", "done": true, '
            '"prompt_eval_count": 5, "eval_count": 2}'
        ),
    ]
    api.app.state.http.stream = MagicMock(
        return_value=FakeStream(lines)
    )

    with client.stream(
        "POST",
        "/generate/ollama/stream",
        json={"prompt": "Hi"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )
        body = "".join(response.iter_text())

    events = parse_sse(body)
    tokens = [data["delta"] for event, data in events if event == "token"]
    done = [data for event, data in events if event == "done"][0]

    assert tokens == ["Hello", " world"]
    assert done["response"] == "Hello world"
    assert done["usage"]["completion_tokens"] == 2
    assert done["time_to_first_token_seconds"] is not None


def test_stream_vllm(client):
    lines = [
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        'data: {"choices": [{"delta": {"content": " there"}}]}',
        (
            'data: {"choices": [], "usage": {"prompt_tokens": 4, '
            '"completion_tokens": 2, "total_tokens": 6}}'
        ),
        "data: [DONE]",
    ]
    api.app.state.http.stream = MagicMock(
        return_value=FakeStream(lines)
    )

    with client.stream(
        "POST",
        "/generate/vllm/stream",
        json={"prompt": "Hi"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = parse_sse(body)
    tokens = [data["delta"] for event, data in events if event == "token"]
    done = [data for event, data in events if event == "done"][0]

    assert tokens == ["Hi", " there"]
    assert done["response"] == "Hi there"
    assert done["usage"]["total_tokens"] == 6


def test_stream_unknown_backend_returns_404(client):
    response = client.post(
        "/generate/unknown/stream",
        json={"prompt": "Hi"},
    )
    assert response.status_code == 404


def test_metrics_endpoint(client):
    # Exercise the labeled metrics so they are registered in the registry.
    api.app.state.http.post = AsyncMock(
        return_value=mock_response(
            {"response": "ok", "prompt_eval_count": 1, "eval_count": 1}
        )
    )
    client.post("/generate/ollama", json={"prompt": "Hello"})

    api.app.state.http.get = AsyncMock(
        side_effect=[
            mock_response(OLLAMA_TAGS),
            mock_response(VLLM_MODELS),
        ]
    )
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llmops_http_requests_total" in response.text
    assert "llmops_tokens_total" in response.text
    assert "llmops_backend_up" in response.text
    assert "llmops_time_to_first_token_seconds" in response.text
