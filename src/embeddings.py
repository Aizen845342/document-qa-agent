"""
Step 2: Embed chunks with OpenAI embeddings and build/persist a Chroma vector store.

Pseudocode:
    build_vector_store(chunks, persist_directory) ->
        embed each chunk's text via OpenAIEmbeddings
        store (embedding, chunk_text, metadata) in a Chroma collection
        persist to disk at persist_directory
        return the Chroma store handle

    load_vector_store(persist_directory) ->
        load an already-persisted Chroma store from disk
"""


def build_vector_store(chunks: list, persist_directory: str = "data/chroma"):
    """Embed chunks and build a persisted Chroma vector store; return the store handle."""
    raise NotImplementedError


def load_vector_store(persist_directory: str = "data/chroma"):
    """Load an existing persisted Chroma vector store from disk."""
    raise NotImplementedError
