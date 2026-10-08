"""
Step 2: Embed chunks with local embeddings and build/persist a Chroma vector store.

Pseudocode:
    build_vector_store(chunks, persist_directory) ->
        embed each chunk's text via local embeddings
        store (embedding, chunk_text, metadata) in a Chroma collection
        persist to disk at persist_directory
        return the Chroma store handle

    load_vector_store(persist_directory) ->
        load an already-persisted Chroma store from disk
"""
from pathlib import Path
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_documents, split_into_chunks

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "data/chroma"

def build_vector_store(chunks: list, persist_directory: str = PERSIST_DIR):
    """Embed chunks and build a persisted Chroma vector store; return the store handle."""
    local_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=local_embeddings,
        persist_directory= persist_directory,
    )
    

    return vector_store


def load_vector_store(persist_directory: str = PERSIST_DIR):
    """Load an existing persisted Chroma vector store from disk."""
    local_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    if not Path(persist_directory).exists():
        raise FileNotFoundError(f"Persist directory '{persist_directory}' does not exist.")

    return Chroma(
    persist_directory=persist_directory,
    embedding_function=local_embeddings,
)


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
