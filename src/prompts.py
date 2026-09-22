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


QA_SYSTEM_INSTRUCTIONS = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say 'I don't know' rather than guessing. "
    "Cite the source document and page number(s) you used for your answer."
)


def build_qa_prompt(question: str, retrieved_chunks: list) -> str:
    """Assemble the grounded, citation-enforcing prompt from a question and its retrieved chunks."""
    raise NotImplementedError
