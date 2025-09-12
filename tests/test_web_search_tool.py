

from yamllm.tools.utility_tools import WebSearch, DuckDuckGoProvider


class FailingDDG:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, *args, **kwargs):
        raise RuntimeError("202 Ratelimit: simulated")


class StubSerpAPIProvider:
    def __init__(self):
        self.calls = 0

    def search(self, query: str, max_results: int = 5):
        self.calls += 1
        return [
            {"title": "result1", "snippet": "s1", "url": "http://example.com/1"},
            {"title": "result2", "snippet": "s2", "url": "http://example.com/2"},
        ][:max_results]


def test_web_search_fallback_to_next_provider(monkeypatch):
    # Monkeypatch DDGS used inside DuckDuckGoProvider to simulate rate limit
    import yamllm.tools.utility_tools as ut
    monkeypatch.setattr(ut, "DDGS", FailingDDG)

    # Inject a stub SerpAPI provider to avoid network
    stub = StubSerpAPIProvider()
    ws = WebSearch(providers=[DuckDuckGoProvider(), stub], timeout=1, max_retries=1)

    out = ws.execute("test", max_results=2)
    assert "results" in out and out["num_results"] == 2
    assert stub.calls == 1


def test_web_search_all_providers_fail(monkeypatch):
    import yamllm.tools.utility_tools as ut
    monkeypatch.setattr(ut, "DDGS", FailingDDG)

    class FailingProvider:
        def search(self, *args, **kwargs):
            raise RuntimeError("fail")

    ws = WebSearch(providers=[DuckDuckGoProvider(), FailingProvider()], timeout=1, max_retries=1)
    out = ws.execute("test", max_results=1)
    assert "error" in out

