"""Small HTTP client for the Streamlit frontend."""

from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def get_api_base_url() -> str:
    """Return the FastAPI base URL for the frontend."""
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def ask_question(question: str) -> dict[str, Any]:
    """Send a question to the FastAPI `/ask` endpoint."""
    response = requests.post(
        f"{get_api_base_url()}/ask",
        json={"question": question},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_history() -> list[dict[str, Any]]:
    """Fetch recent query history from the FastAPI `/history` endpoint."""
    response = requests.get(f"{get_api_base_url()}/history", timeout=15)
    response.raise_for_status()
    return response.json().get("history", [])


def check_excel_file(filename: str, content: bytes) -> dict[str, Any]:
    """Send an Excel workbook to the FastAPI `/check-excel` endpoint."""
    response = requests.post(
        f"{get_api_base_url()}/check-excel",
        files={
            "file": (
                filename,
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
