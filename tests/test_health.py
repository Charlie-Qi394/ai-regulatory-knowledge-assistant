"""Basic backend health-check test."""

from fastapi.testclient import TestClient
import pytest

from backend.main import app


def test_health_check() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_rejects_empty_question() -> None:
    client = TestClient(app)
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question must not be empty."


def test_ask_returns_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ask_and_save(question: str) -> dict[str, object]:
        return {
            "answer": "Use approved label checks before release [Source 1].",
            "sources": [
                {
                    "source_id": 1,
                    "filename": "sample.txt",
                    "chunk_index": 0,
                    "page_number": None,
                    "distance": 0.2,
                    "similarity": 0.8,
                    "excerpt": "Label checks should be completed before release.",
                }
            ],
        }

    monkeypatch.setattr("backend.app.api.routes.ask_and_save", fake_ask_and_save)

    client = TestClient(app)
    response = client.post("/ask", json={"question": "What label checks are needed?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What label checks are needed?"
    assert body["answer"] == "Use approved label checks before release [Source 1]."
    assert body["sources"][0]["filename"] == "sample.txt"
