"""
Step 3: Embed a user question and retrieve the top-k most relevant chunks.

Pseudocode:
    retrieve_relevant_chunks(vector_store, question, k) ->
        embed the question with the same embedding model used for chunks
        run a similarity search against the vector store
        return the top-k chunks with their source metadata and similarity scores
"""


def retrieve_relevant_chunks(vector_store, question: str, k: int = 4) -> list:
    """Return the top-k most relevant chunks (with metadata) for a question."""
    raise NotImplementedError
