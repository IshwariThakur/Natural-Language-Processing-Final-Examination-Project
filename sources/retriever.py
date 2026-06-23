import chromadb

from common import CHROMA_DIR
from index_to_chroma import COLLECTION_NAME, get_embedder

_model = None
_collection = None

def _embedder():
    global _model

    if _model is None:
        _model = get_embedder()

    return _model


def get_collection():

    global _collection

    if _collection is None:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        _collection = client.get_collection(
            COLLECTION_NAME
        )

    return _collection

def retrieve(question, k=5):

    model = _embedder()

    query_vector = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    collection = get_collection()

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k * 3
    )

    docs = []
    seen_titles = set()

    for text, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):

        title = meta.get("title", "").strip()

        if title in seen_titles:
            continue

        seen_titles.add(title)

        docs.append({
            "title": title,
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "text": text,
            "score": round(1 - distance, 3)
        })

        if len(docs) >= k:
            break

    return docs


def main():

    question = input(
        "Ask a question about Airbus: "
    ).strip()

    if not question:
        print("No question entered.")
        return

    results = retrieve(
        question,
        k=5
    )

    print(f"\nTop {len(results)} relevant documents:\n")

    for i, doc in enumerate(results, start=1):

        print(
            f"{i}. [{doc['source']}] {doc['title']}"
        )

        print(
            f"   similarity: {doc['score']:.3f}"
        )

        print(
            f"   {doc['text'][:200]}..."
        )

        print(
            f"   {doc['url']}\n"
        )


if __name__ == "__main__":
    main()