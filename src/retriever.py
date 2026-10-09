"""
Step 3: Embed a user question and retrieve the top-k most relevant chunks.

Pseudocode:
    retrieve_relevant_chunks(vector_store, question, k) ->
        embed the question with the same embedding model used for chunks
        run a similarity search against the vector store
        return the top-k chunks with their source metadata and similarity scores
"""
from embeddings import load_vector_store, build_vector_store
from pathlib import Path
from ingest import load_documents, split_into_chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "data/chroma"

def retrieve_relevant_chunks(vector_store, question: str, k: int = 4) -> list:
    """Return the top-k most relevant chunks (with metadata) for a question."""
    if vector_store is None:
        vector_store = load_vector_store()

    return vector_store.similarity_search(question, k=k)

if __name__ == "__main__":
    # Example usage
    if not Path(PERSIST_DIR).exists():
        documents = load_documents("documents")
        chunks = split_into_chunks(documents)
        store = build_vector_store(chunks)
        print(f"toTAL CHUNKS STORED IN : {store._collection.count()}")
    else:
        loaded_store = load_vector_store()
        print(f"Number of chunks in the loaded vector store: {loaded_store._collection.count()}")
        document_list = retrieve_relevant_chunks(loaded_store, "What is the relationship between wave speed, frequency, and wavelength?", k=4)
        for doc in document_list:
            print(f"Source: {doc.metadata['source']}")
            print(f"Page: {doc.metadata['page_label']}")
            print(f"Preview: {doc.page_content[:200]}")
            print("-" * 50)
        