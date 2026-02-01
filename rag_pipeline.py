from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class RAGConfig:
    chroma_dir: str = "data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150


def read_pdfs_to_docs(uploaded_files) -> List[Document]:
    """
    Convertit des PDFs (Streamlit UploadedFile) en Documents LangChain.
    Ajoute metadata: source + page.
    """
    docs: List[Document] = []
    for f in uploaded_files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": getattr(f, "name", "uploaded.pdf"), "page": i + 1},
                    )
                )
    return docs


def chunk_docs(docs: List[Document], cfg: RAGConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def get_vectordb(cfg: RAGConfig) -> Chroma:
    """
    Charge ou prépare une base Chroma persistée.
    """
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    vectordb = Chroma(
        persist_directory=cfg.chroma_dir,
        embedding_function=embeddings,
    )
    return vectordb


def index_documents(vectordb: Chroma, chunks: List[Document]) -> int:
    """
    Ajoute des chunks à la base (persistée).
    """
    vectordb.add_documents(chunks)
    # persistance gérée par Chroma via persist_directory
    return len(chunks)


def as_retriever(vectordb: Chroma, k: int = 4):
    return vectordb.as_retriever(search_kwargs={"k": k})
