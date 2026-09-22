"""
Step 5: The LangChain RAG chain - ties retrieval and generation together into
a single callable agent.

Pseudocode:
    build_qa_chain(vector_store) ->
        construct a LangChain RetrievalQA-style chain using:
            - the vector store's retriever (Step 3)
            - the grounded prompt template (Step 4)
            - an OpenAI chat model for generation
        return the chain

    answer_question(chain, question) ->
        run the chain on the question
        return { answer, sources: [ {document, page}, ... ] }
"""


def build_qa_chain(vector_store):
    """Construct and return the end-to-end LangChain RAG chain."""
    raise NotImplementedError


def answer_question(chain, question: str) -> dict:
    """Run the chain on a question; return the answer plus cited sources."""
    raise NotImplementedError
