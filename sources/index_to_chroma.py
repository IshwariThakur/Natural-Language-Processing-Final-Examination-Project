import json
import glob
import os

import chromadb
from common import CLEAN_DIR, CHROMA_DIR        

COLLECTION_NAME = "airbus_intel"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE    = 300   
CHUNK_OVERLAP = 50    

METADATA_FIELDS = ["source", "source_type", "title", "url", "published", "parent_id"]


def load_documents():
    docs, seen = [], set()
    for path in sorted(glob.glob(str(CLEAN_DIR / "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_id, text = d.get("id"), (d.get("text") or "").strip()
                if not doc_id or not text or doc_id in seen:
                    continue
                seen.add(doc_id)
                docs.append(d)
    return docs


def sanitize_metadata(doc):
    return {f: (doc.get(f) or "") for f in METADATA_FIELDS}


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [text]
    step   = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):   
            break
    return chunks


def build_chunks(docs):
    records = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"])):
            records.append({
                "id":          f"{doc['id']}_{i}",
                "text":        piece,
                "source":      doc.get("source", ""),
                "source_type": doc.get("source_type", ""),
                "title":       doc.get("title", ""),
                "url":         doc.get("url", ""),
                "published":   doc.get("published"),
                "parent_id":   doc["id"],
            })
    return records


from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder(model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print(f"  Model not cached locally. Downloading {model_name} …")
        return SentenceTransformer(model_name, local_files_only=False)


def embed_texts(model, texts):
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]


def get_collection(reset=False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_documents(collection, docs, embeddings):
    collection.upsert(
        ids        =[d["id"]   for d in docs],
        documents  =[d["text"] for d in docs],
        embeddings =embeddings,
        metadatas  =[sanitize_metadata(d) for d in docs],
    )


def index():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    print(f"Loaded {len(docs)} unique document(s) from {CLEAN_DIR}")
    if not docs:
        print("No documents found. Run the collectors and clean step first.")
        return

    chunks = build_chunks(docs)
    print(f"Split into {len(chunks)} chunk(s)")

    print(f"Embedding with {EMBEDDING_MODEL} ...")
    model      = get_embedder()
    embeddings = embed_texts(model, [c["text"] for c in chunks])

    collection = get_collection(reset=True)   
    upsert_documents(collection, chunks, embeddings)
    print(
        f"Indexed {collection.count()} chunk(s) from {len(docs)} document(s) "
        f"into '{COLLECTION_NAME}'"
    )


if __name__ == "__main__":
    index()