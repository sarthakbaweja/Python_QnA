from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.schemas.chat import AskRequest, AskResponse, HealthResponse
from rag.pipeline.graph import graph
from rag.retriever.qdrant_client import get_qdrant_client

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("20/minute")
def ask(request: Request, body: AskRequest) -> AskResponse:
    try:
        config = {"configurable": {"thread_id": body.session_id}}
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=body.question)],
                "session_id": body.session_id,
                "context": [],
                "sources": [],
            },
            config,
        )
        last_ai = result["messages"][-1]
        return AskResponse(
            answer=last_ai.content,
            session_id=body.session_id,
            sources=result.get("sources", []),
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Pipeline error: unable to generate answer")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_status = "connected"
    try:
        get_qdrant_client().get_collections()
    except Exception:
        qdrant_status = "disconnected"
    return HealthResponse(status="ok", qdrant=qdrant_status, version="1.0.0")
