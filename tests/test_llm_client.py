import httpx
import pytest

from src.generation.llm_client import LLMClient, LLMError

MSG = [{"role": "user", "content": "hi"}]


def make_client(handler, **kw):
    return LLMClient(model="m", ollama_host="http://ollama", http=httpx.Client(
        transport=httpx.MockTransport(handler)), **kw)


def test_ollama_ok():
    def h(req):
        assert req.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": " hello "}})
    r = make_client(h).chat(MSG)
    assert r.text == "hello" and r.model == "ollama:m"


def test_fallback_to_groq():
    def h(req):
        if req.url.host == "ollama":
            return httpx.Response(500)
        assert req.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"content": "g"}}]})
    r = make_client(h, groq_api_key="k").chat(MSG)
    assert r.text == "g" and r.model.startswith("groq:")


def test_no_fallback_raises():
    with pytest.raises(LLMError):
        make_client(lambda req: httpx.Response(500), groq_api_key="").chat(MSG)
