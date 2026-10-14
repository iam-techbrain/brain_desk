from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb

app = FastAPI(title="FastAPI + ChromaDB RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="demo_docs")

class DocItem(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: List[float]

class QueryItem(BaseModel):
    embedding: List[float]
    top_k: int = 2

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "FastAPI + ChromaDB Service Running",
        "total_documents": collection.count()
    }

@app.get("/chroma/documents")
def get_all_documents():
    """ChromaDB ke andar ke saare documents dekhne ke liye"""
    data = collection.get()
    return {
        "count": len(data["ids"]),
        "ids": data["ids"],
        "documents": data["documents"],
        "metadatas": data["metadatas"]
    }

@app.post("/chroma/add")
def add_document(item: DocItem):
    """ChromaDB me document add/upsert karne ke liye"""
    collection.upsert(
        ids=[item.id],
        documents=[item.text],
        embeddings=[item.embedding],
        metadatas=[item.metadata or {}]
    )
    return {"message": "Document added successfully", "id": item.id}

@app.post("/chroma/query")
def query_chroma(item: QueryItem):
    """Vector similarity search karne ke liye"""
    results = collection.query(
        query_embeddings=[item.embedding],
        n_results=item.top_k
    )
    return {
        "query_results": [
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    }
