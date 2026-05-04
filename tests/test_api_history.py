"""API tests for query history endpoint."""

from fastapi.testclient import TestClient

from backend.main import app


def test_history_returns_list(monkeypatch) -> None:
    def fake_get_recent_history() -> list[dict[str, object]]:
        return []

    monkeypatch.setattr("backend.app.api.routes.get_recent_history", fake_get_recent_history)

    client = TestClient(app)
    response = client.get("/history")

    assert response.status_code == 200
    assert "history" in response.json()
    assert isinstance(response.json()["history"], list)


def test_history_returns_service_unavailable_when_database_is_unavailable(monkeypatch) -> None:
    def fake_get_recent_history() -> list[dict[str, object]]:
        raise RuntimeError("DATABASE_URL is not set.")

    monkeypatch.setattr("backend.app.api.routes.get_recent_history", fake_get_recent_history)

    client = TestClient(app)
    response = client.get("/history")

    assert response.status_code == 503
    assert response.json()["detail"] == "DATABASE_URL is not set."
