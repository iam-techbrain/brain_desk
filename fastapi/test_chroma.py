import chromadb

# 1. Local disk par ChromaDB initialize karein (yeh './chroma_data' me chroma.sqlite3 banata hai)
client = chromadb.PersistentClient(path="./chroma_data")

# 2. Collection banayein
collection = client.get_or_create_collection(name="demo_docs")

# 3. Documents aur unke sample embeddings add karein (Instant test, bina internet delay ke)
print("--> Adding sample documents with vector embeddings...")
collection.upsert(
    documents=[
        "FastAPI is a modern web framework for Python.",
        "Express is a minimal backend framework for Node.js.",
        "React is a frontend UI library.",
        "ChromaDB is a vector database for RAG."
    ],
    # 4-dimensional sample vectors
    embeddings=[
        [0.1, 0.8, 0.2, 0.0],
        [0.1, 0.7, 0.3, 0.1],
        [0.9, 0.1, 0.2, 0.0],
        [0.0, 0.1, 0.1, 0.9]
    ],
    metadatas=[
        {"type": "backend", "author": "tiangolo"},
        {"type": "backend", "author": "openjs"},
        {"type": "frontend", "author": "meta"},
        {"type": "vector_db", "author": "chroma"}
    ],
    ids=["doc_fastapi", "doc_express", "doc_react", "doc_chromadb"]
)

# 4. Total count check karein
count = collection.count()
print(f"Total documents stored in ChromaDB: {count}")

# 5. ChromaDB me stored saare records dekhein
print("\n--> Fetching all stored data from ChromaDB:")
all_data = collection.get()
for i, (doc_id, doc, meta) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
    print(f"[{i+1}] ID: {doc_id} | Doc: {doc} | Metadata: {meta}")

# 6. Vector Similarity Search test karein
print("\n--> Querying for similar documents (using vector [0.05, 0.1, 0.15, 0.85]):")
results = collection.query(
    query_embeddings=[[0.05, 0.1, 0.15, 0.85]],
    n_results=2
)
print("Top Matches:")
for doc, doc_id, dist in zip(results['documents'][0], results['ids'][0], results['distances'][0]):
    print(f"  -> ID: {doc_id} (Distance: {dist:.4f}): '{doc}'")
