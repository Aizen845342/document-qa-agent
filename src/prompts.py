"""
Step 4: Prompt templates that force the model to answer ONLY from retrieved context
and to cite its sources, rather than guessing when the context is insufficient.

Pseudocode:
    build_qa_prompt(question, retrieved_chunks) ->
        format retrieved chunks with their source labels (doc name + page)
        instruct the model: "answer only using the context below; if the context
        does not contain the answer, say 'I don't know'; cite the source(s) used"
        return the assembled prompt string
"""
from retriever import retrieve_relevant_chunks
from embeddings import load_vector_store
from pathlib import Path



QA_SYSTEM_INSTRUCTIONS = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say 'I don't know' rather than guessing. "
    "Cite the source document and page number(s) you used for your answer."
)


def build_qa_prompt(question: str, retrieved_chunks: list) -> str:
    """Assemble the grounded, citation-enforcing prompt from a question and its retrieved chunks."""
    assembled_prompt = []
    for chunk in retrieved_chunks:
        formatted_chunk = f"Source: {Path(chunk.metadata['source']).name}\nPage: {chunk.metadata.get('page_label', chunk.metadata.get('page', 'unknown'))}\nContent: {chunk.page_content}"
        assembled_prompt.append(formatted_chunk)

    context = "\n\n".join(assembled_prompt)
    final_prompt = f"{QA_SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"
    return final_prompt

if __name__ == "__main__":
    # Example usage
    store = load_vector_store()
    chunks = retrieve_relevant_chunks(store, "What is the relationship between wave speed, frequency, and wavelength?", k=4)
    prompt = build_qa_prompt("What is the relationship between wave speed, frequency, and wavelength?", chunks)
    print(prompt)



