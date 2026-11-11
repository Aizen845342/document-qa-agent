
"""Prompt construction for grounded document question answering."""

from pathlib import Path


QA_SYSTEM_INSTRUCTIONS = """
You are a document question-answering assistant.

Answer the user's question using only the provided document passages.

Rules:
1. Read all retrieved passages carefully before answering.
2. Find the passage that directly answers the question.
3. The relevant information may appear in any passage.
4. If the answer is present, provide it directly and clearly.
5. Do not confuse campaign target audiences with creator selection criteria.
6. Do not invent facts or use outside knowledge.
7. Only say "I don't know" when none of the passages contain enough information.
8. Keep the answer concise and factual.
9. Cite the source document and page number.

If the information is explicitly present in the context,
answer the question instead of saying "I don't know".
""".strip()


def build_qa_prompt(question: str, retrieved_chunks: list) -> str:
    """Build a grounded prompt from retrieved document chunks."""

    formatted_passages = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source = Path(
            chunk.metadata.get("source", "Unknown document")
        ).name

        page = chunk.metadata.get(
            "page_label",
            chunk.metadata.get("page", "Unknown")
        )

        formatted_passages.append(
            f"Passage {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text:\n{chunk.page_content}"
        )

    context = "\n\n---\n\n".join(formatted_passages)

    if not context:
        context = "No relevant document passages were found."

    return (
        f"{QA_SYSTEM_INSTRUCTIONS}\n\n"
        f"DOCUMENT PASSAGES:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    )
