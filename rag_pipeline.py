"""
rag_pipeline.py — Document ingestion, embedding, and retrieval pipeline.

Responsibilities:
  1. Load .txt documents from the knowledge base directory.
  2. Split them into overlapping chunks.
  3. Embed with the configured provider (local or OpenAI).
  4. Persist / load a FAISS vector store.
  5. Expose a retriever for the agent to use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import settings


# ── Embedding factory ────────────────────────────────────────────────────────

def _get_embeddings():
    """Return the configured embedding model instance."""
    if settings.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY,
        )
    else:
        # Local, fully offline — HuggingFace sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


# ── Document loading ─────────────────────────────────────────────────────────

def load_documents(kb_dir: Path) -> List[Document]:
    """
    Load all .txt files from kb_dir recursively.
    Each file becomes one or more Document objects with path metadata.
    """
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=False,
    )
    docs = loader.load()

    # Enrich metadata with a human-readable source name
    for doc in docs:
        raw_source = doc.metadata.get("source", "")
        doc.metadata["source"] = Path(raw_source).stem.replace("_", " ").title()

    return docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for fine-grained retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n━", "\n─", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


# ── Vector store ─────────────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document], embeddings) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(store: FAISS, path: Path, embeddings) -> None:
    """Persist vector store to disk."""
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def load_vector_store(path: Path, embeddings) -> FAISS:
    """Load a previously saved FAISS vector store."""
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Encapsulates the full RAG pipeline: build once, query many times.

    Usage:
        pipeline = RAGPipeline()
        pipeline.build(force_rebuild=False)   # loads from disk if available
        retriever = pipeline.get_retriever()
    """

    def __init__(self) -> None:
        self._embeddings = None
        self._store: FAISS | None = None

    def _ensure_embeddings(self):
        if self._embeddings is None:
            self._embeddings = _get_embeddings()

    def build(self, force_rebuild: bool = False) -> None:
        """
        Build (or reload) the vector store.

        Args:
            force_rebuild: If True, always re-ingest documents even if a
                           cached vector store exists on disk.
        """
        self._ensure_embeddings()
        index_file = settings.VECTOR_STORE_PATH / "index.faiss"

        if not force_rebuild and index_file.exists():
            self._store = load_vector_store(
                settings.VECTOR_STORE_PATH, self._embeddings
            )
            return

        # ── Fresh ingestion ──────────────────────────────────────────────────
        docs = load_documents(settings.KNOWLEDGE_BASE_DIR)
        chunks = split_documents(docs)
        self._store = build_vector_store(chunks, self._embeddings)
        save_vector_store(self._store, settings.VECTOR_STORE_PATH, self._embeddings)

    def get_retriever(self):
        """Return a LangChain retriever tuned to the configured top-K."""
        if self._store is None:
            raise RuntimeError("Call build() before get_retriever().")
        return self._store.as_retriever(
            search_type="mmr",          # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": settings.TOP_K_RESULTS,
                "fetch_k": settings.TOP_K_RESULTS * 3,
                "lambda_mult": 0.7,     # balance relevance vs diversity
            },
        )

    def get_document_count(self) -> int:
        """Return the total number of chunks indexed."""
        if self._store is None:
            return 0
        return self._store.index.ntotal
