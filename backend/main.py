"""FastAPI entry point for the AI Regulatory Knowledge Assistant."""

from fastapi import FastAPI

from backend.app.api.routes import router


app = FastAPI(
    title="AI Regulatory Knowledge Assistant",
    description="Portfolio RAG assistant for querying regulatory knowledge sources.",
    version="0.1.0",
)

app.include_router(router)
