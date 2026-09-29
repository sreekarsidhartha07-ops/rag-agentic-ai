"""FastAPI interface.  Run: uvicorn app:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph(index_name=config.PINECONE_INDEX_NAME)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    confidence_score: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    state = {
        "question": request.query,
        "context": [],
        "scores": [],
        "answer": "",
        "score": 0.0,
    }
    result = graph.invoke(state)
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )
