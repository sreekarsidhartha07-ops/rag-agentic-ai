"""Environment setup & constants."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_DRIVE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# Below this best-match cosine similarity the question is treated as out-of-scope
MIN_RELEVANCE = 0.25
REFUSAL_MESSAGE = "I cannot answer based on the provided document."
