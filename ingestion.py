"""PDF loading, splitting & Pinecone index setup.

Run:  python -m src.ingestion
"""
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src import config


def download_pdf(pdf_path: str = config.PDF_PATH) -> None:
    """Download the eBook from Google Drive if it is not already present."""
    if os.path.exists(pdf_path):
        return
    import gdown

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    print("Downloading PDF from Google Drive...")
    gdown.download(id=config.PDF_DRIVE_ID, output=pdf_path, quiet=False)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Could not download the PDF. Download it manually and save as {pdf_path}"
        )


def ensure_index(index_name: str) -> None:
    """Create the Pinecone index (1536 dims, cosine) if it does not exist."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)


def run_ingestion(pdf_path: str, index_name: str):
    download_pdf(pdf_path)

    # 1. Load document
    docs = PyPDFLoader(pdf_path).load()
    print(f"Loaded {len(docs)} pages")

    # 2. Chunk document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed & upsert into Pinecone
    ensure_index(index_name)
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=index_name
    )
    print("Ingestion complete.")
    return vector_store


if __name__ == "__main__":
    run_ingestion(config.PDF_PATH, config.PINECONE_INDEX_NAME)
