# RAG Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation chatbot that answers **only** from the Agentic AI eBook.
Built with Python, LangGraph, Pinecone, OpenAI and FastAPI.

## Architecture

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000/200)
    -> OpenAI text-embedding-3-small (1536-d) -> Pinecone (cosine)

POST /chat -> FastAPI -> LangGraph:  START -> retrieve -> generate -> END
                                      |          |           |
                                  top-3 chunks   |     strict-context prompt (gpt-4o-mini)
                                  + similarity   |     + confidence score
                                                 +-> out-of-scope guard (best score < threshold => refuse)
```

| File | Responsibility |
|---|---|
| `src/config.py` | Env vars & constants (models, chunk size, top-k, threshold) |
| `src/ingestion.py` | Download/load PDF, chunk, create Pinecone index, upsert embeddings |
| `src/graph.py` | LangGraph state (`AgentState`), `retrieve` and `generate` nodes |
| `app.py` | FastAPI `/chat` endpoint |
| `tests_sample_queries.py` | 6 benchmark queries incl. an out-of-scope one |

**Grounding:** the prompt forbids outside knowledge, and if the best retrieved chunk's cosine
similarity is below `MIN_RELEVANCE` (0.25, in `src/config.py`) the LLM is skipped and the bot refuses.

**Confidence score:** the best cosine similarity among the retrieved chunks (0-1); `0.0` when the bot refuses.

## Setup

```bash
git clone <this-repo> && cd rag-agentic-ai
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in OPENAI_API_KEY and PINECONE_API_KEY
```

## Ingest the PDF

The script downloads the eBook automatically to `data/Ebook-Agentic-AI.pdf`
(or place the file there manually), creates the Pinecone index and upserts the chunks:

```bash
python -m src.ingestion
```

## Run the API

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```

Response:

```json
{
  "answer": "...",
  "retrieved_chunks": ["...", "...", "..."],
  "confidence_score": 0.62
}
```

## Test queries

With the server running:

```bash
python tests_sample_queries.py
```

Queries 1-5 should return grounded answers with retrieved chunks. The last query
("Who won the 2022 FIFA World Cup?") must return
`I cannot answer based on the provided document.` with confidence `0.0`.
